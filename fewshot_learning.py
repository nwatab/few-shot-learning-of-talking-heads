"""Few-shot fine-tuning on a new identity (Section 3.3).

    python fewshot_learning.py --data datasets/fewshot/monalisa \
        --meta-dir trained_models --meta-tag latest --out trained_models/monalisa

--data must contain frames/ and lndmks/ with the same file names (T images).
"""
import argparse
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf

from data_loader import index_videos, load_image
from models import Config
from trainer import FewShotFinetuner, TalkingHeadGAN


def load_person(root, size):
    (lmk_dir, frm_dir, files), = index_videos(root)
    frames = np.stack([load_image(os.path.join(frm_dir, f), size) for f in files])
    lmks = np.stack([load_image(os.path.join(lmk_dir, f), size) for f in files])
    return frames, lmks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--meta-dir', required=True)
    ap.add_argument('--meta-tag', default='latest')
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--lr-g', type=float, default=5e-5)
    ap.add_argument('--lr-d', type=float, default=2e-4)
    ap.add_argument('--vgg19-weights', default=None)
    ap.add_argument('--vggface-weights', default=None)
    ap.add_argument('--legacy', action='store_true')
    a = ap.parse_args()

    frames, lmks = load_person(a.data, a.size)
    t = len(frames)
    cfg = (Config.legacy if a.legacy else Config.fixed)(image_size=a.size, width=a.width, k=t)
    gan = TalkingHeadGAN(cfg, num_videos=1, vgg19_weights=a.vgg19_weights, vggface_weights=a.vggface_weights)
    gan.load(a.meta_dir, a.meta_tag, load_W=False)

    styles = np.concatenate([lmks, frames], axis=-1)[None]          # (1, T, S, S, 6)
    ft = FewShotFinetuner(gan, tf.constant(styles), lr_g=a.lr_g, lr_d=a.lr_d)
    rng = np.random.default_rng(0)
    for epoch in range(a.epochs):
        order = rng.permutation(t)
        for i in range(0, t - a.batch + 1, a.batch):
            idx = order[i:i + a.batch]
            m = ft.train_step(tf.constant(frames[idx]), tf.constant(lmks[idx]))
        print(epoch, {k: round(float(v), 4) for k, v in m.items()}, flush=True)

    os.makedirs(a.out, exist_ok=True)
    gan.G.save_weights(os.path.join(a.out, 'G.weights.h5'))
    gan.D.save_weights(os.path.join(a.out, 'D.weights.h5'))
    np.save(os.path.join(a.out, 'psi.npy'), ft.psi.numpy())
    np.save(os.path.join(a.out, 'w_prime.npy'), ft.w_prime.numpy())


if __name__ == '__main__':
    main()
