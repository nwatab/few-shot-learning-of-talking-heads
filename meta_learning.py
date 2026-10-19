"""Meta-learning stage (Section 3.2) on VoxCeleb-style data.

    python meta_learning.py --data datasets/voxceleb2-9f/train --out trained_models
"""
import argparse
import json
import logging
import os
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow as tf

from data_loader import EpisodeSampler
from models import Config
from trainer import TalkingHeadGAN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='directory containing frames/ and lndmks/')
    ap.add_argument('--out', default='trained_models')
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--epochs', type=int, default=75)
    ap.add_argument('--lr-g', type=float, default=5e-5)
    ap.add_argument('--lr-d', type=float, default=2e-4)
    ap.add_argument('--save-every', type=int, default=1000, help='steps')
    ap.add_argument('--log-every', type=int, default=50, help='steps')
    ap.add_argument('--resume', default=None, help='checkpoint tag in --out to resume from')
    ap.add_argument('--vgg19-weights', default=None)
    ap.add_argument('--vggface-weights', default=None)
    ap.add_argument('--legacy', action='store_true', help='reproduce the original (broken) behaviour')
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    logging.basicConfig(filename=os.path.join(a.out, 'meta_learning.log'), level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    cfg = (Config.legacy if a.legacy else Config.fixed)(image_size=a.size, width=a.width, k=a.k)
    sampler = EpisodeSampler(a.data, a.size, a.k, a.batch, random_frames=cfg.random_frames)
    logging.info('config %s; %d videos', cfg, len(sampler))
    gan = TalkingHeadGAN(cfg, len(sampler), lr_g=a.lr_g, lr_d=a.lr_d,
                         vgg19_weights=a.vgg19_weights, vggface_weights=a.vggface_weights)
    if a.resume:
        gan.load(a.out, a.resume)

    step, t0 = 0, time.time()
    for epoch in range(a.epochs):
        for frames, lmks, styles, idx in sampler.epoch():
            m = gan.train_step(tf.constant(frames), tf.constant(lmks), tf.constant(styles), tf.constant(idx))
            if step % a.log_every == 0:
                rec = {k: round(float(v), 5) for k, v in m.items()}
                rec.update(epoch=epoch, step=step, sec=round(time.time() - t0, 1))
                logging.info(json.dumps(rec))
            step += 1
            if step % a.save_every == 0:
                gan.save(a.out, 'latest')
        gan.save(a.out, f'epoch{epoch:03d}')
        gan.save(a.out, 'latest')


if __name__ == '__main__':
    main()
