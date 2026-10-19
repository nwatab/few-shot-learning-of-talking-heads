"""Render frames for a sequence of landmark images.

Fine-tuned identity (output of fewshot_learning.py):
    python prediction.py --finetuned trained_models/monalisa --landmarks lm/*.jpg --out results

Meta-learned model only (embed the given source frames, no fine-tuning):
    python prediction.py --meta-dir trained_models --meta-tag latest \
        --source-frames f0.jpg f1.jpg --source-landmarks l0.jpg l1.jpg --landmarks lm/*.jpg --out results
"""
import argparse
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
from PIL import Image

from data_loader import load_image
from models import Config, build_embedder, build_generator, build_projector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--landmarks', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--finetuned', default=None)
    ap.add_argument('--meta-dir', default=None)
    ap.add_argument('--meta-tag', default='latest')
    ap.add_argument('--source-frames', nargs='*', default=[])
    ap.add_argument('--source-landmarks', nargs='*', default=[])
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--legacy', action='store_true')
    a = ap.parse_args()

    cfg = (Config.legacy if a.legacy else Config.fixed)(image_size=a.size, width=a.width)
    G = build_generator(cfg)
    lmks = np.stack([load_image(p, a.size) for p in a.landmarks])
    if a.finetuned:
        G.load_weights(os.path.join(a.finetuned, 'G.weights.h5'))
        psi = np.load(os.path.join(a.finetuned, 'psi.npy'))[:1]
    else:
        assert a.meta_dir and a.source_frames and len(a.source_frames) == len(a.source_landmarks)
        E, P = build_embedder(cfg), build_projector(cfg)
        for name, m in (('E', E), ('P', P), ('G', G)):
            m.load_weights(os.path.join(a.meta_dir, f'{a.meta_tag}_{name}.weights.h5'))
        style = np.stack([np.concatenate([load_image(l, a.size), load_image(f, a.size)], -1)
                          for f, l in zip(a.source_frames, a.source_landmarks)])
        psi = P.predict(E.predict(style, verbose=0).mean(0, keepdims=True), verbose=0)

    fake = G.predict([lmks, np.repeat(psi, len(lmks), axis=0)], verbose=0)
    os.makedirs(a.out, exist_ok=True)
    for path, img in zip(a.landmarks, fake):
        img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(a.out, os.path.basename(path)))


if __name__ == '__main__':
    main()
