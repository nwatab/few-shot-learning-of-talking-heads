"""CPU smoke test: original (legacy) vs fixed implementation on synthetic heads.

    python smoke_test.py --arm fixed  --steps 1500 --out runs/fixed
    python smoke_test.py --arm legacy --steps 1500 --out runs/legacy

The test is meant to show that the failure mode of the original code
(AdaIN scales collapse with the embedding, residual branches die, output
degenerates to nearest-upsampled bottleneck blocks) is gone. It is NOT a
substitute for training on VoxCeleb2.

Evaluation uses poses never used for training: every training video has 12
rendered frames, frames 0-8 are used for training and 9-11 only as
evaluation targets (with the embedder fed frames 0-7 of the same video).
Identities 24-29 are never seen in training (zero-shot, no fine-tuning).
"""
import argparse
import json
import os
import shutil
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import tensorflow as tf
from PIL import Image

from data_loader import EpisodeSampler, load_image
from models import Config, adain_param_channels
from synthetic_data import write_dataset
from trainer import TalkingHeadGAN

ARMS = {
    'fixed': lambda **kw: Config.fixed(**kw),
    'legacy': lambda **kw: Config.legacy(**kw),
    # fixed everything except the AdaIN parameterisation and the output head
    'fixed_raw_adain': lambda **kw: Config.fixed(adain_residual_scale=False, fixed_output_head=False, **kw),
}
N_TRAIN_PERSONS, N_TEST_PERSONS, VIDEOS, FRAMES, TRAIN_FRAMES = 24, 6, 2, 12, 9


def prepare_data(root, size):
    full = os.path.join(root, 'full')
    train = os.path.join(root, 'train')
    if os.path.isdir(train):
        return full, train
    write_dataset(full, size, N_TRAIN_PERSONS + N_TEST_PERSONS, VIDEOS, FRAMES, seed=0)
    for kind in ('frames', 'lndmks'):
        for p in range(N_TRAIN_PERSONS):
            for v in range(VIDEOS):
                rel = os.path.join(kind, f'id{p:05d}', f'v{v:02d}')
                os.makedirs(os.path.join(train, rel), exist_ok=True)
                for f in range(TRAIN_FRAMES):
                    shutil.copy(os.path.join(full, rel, f'{f}.png'), os.path.join(train, rel, f'{f}.png'))
    return full, train


def eval_set(full, size, k, persons):
    frames, lmks, styles = [], [], []
    for p in persons:
        for v in range(VIDEOS):
            d = lambda kind, f: os.path.join(full, kind, f'id{p:05d}', f'v{v:02d}', f'{f}.png')
            st = np.stack([np.concatenate([load_image(d('lndmks', f), size), load_image(d('frames', f), size)], -1)
                           for f in range(k)])
            for t in range(TRAIN_FRAMES, FRAMES):
                frames.append(load_image(d('frames', t), size))
                lmks.append(load_image(d('lndmks', t), size))
                styles.append(st)
    return np.stack(frames), np.stack(lmks), np.stack(styles)


def blockiness(img, period):
    """Mean |horizontal + vertical gradient| on lines that are multiples of
    ``period`` divided by the mean on all other lines (1.0 = no grid)."""
    g = np.abs(np.diff(img.mean(-1), axis=-1)).mean(axis=(0, 1))      # over x
    h = np.abs(np.diff(img.mean(-1), axis=-2)).mean(axis=(0, 2))      # over y
    ratios = []
    for d in (g, h):
        on = (np.arange(len(d)) % period) == period - 1
        ratios.append(d[on].mean() / max(d[~on].mean(), 1e-8))
    return float(np.mean(ratios))


def adain_scale_stats(gan, e_hat):
    """Mean |effective AdaIN scale| for (residual+upsampling layers, output layer)."""
    psi = gan.P(e_hat, training=False).numpy()
    scales, o = [], 0
    for c in adain_param_channels(gan.cfg):
        s = psi[:, o + c:o + 2 * c]
        o += 2 * c
        scales.append(np.abs(1.0 + s if gan.cfg.adain_residual_scale else s).mean())
    return float(np.mean(scales[:-1])), float(scales[-1])


def generate(gan, lmks, styles, bs=6):
    out = []
    for i in range(0, len(lmks), bs):
        _, e_hat = gan.embed(styles[i:i + bs], training=False)
        out.append(gan.generate(lmks[i:i + bs], e_hat, training=False).numpy())
    return np.concatenate(out)


def evaluate(gan, ev, size):
    frames, lmks, styles = ev
    fake = generate(gan, lmks, styles)
    _, e_hat = gan.embed(styles[:12], training=False)
    s_body, s_out = adain_scale_stats(gan, e_hat)
    to01 = lambda x: (x + 1) / 2
    return dict(l1=float(np.abs(to01(fake) - to01(frames)).mean()),
                block16_fake=blockiness(to01(fake), size // 4),
                block16_real=blockiness(to01(frames), size // 4),
                fake_std=float(to01(fake).std(axis=(1, 2)).mean()),
                adain_scale_body=s_body, adain_scale_out=s_out), fake


def baselines(ev):
    frames, _, styles = ev
    to01 = lambda x: (x + 1) / 2
    src_mean = styles[..., 3:].mean(axis=1)
    return dict(l1_mean_of_sources=float(np.abs(to01(src_mean) - to01(frames)).mean()),
                l1_dataset_mean=float(np.abs(to01(frames.mean(0, keepdims=True)) - to01(frames)).mean()))


def save_grid(path, ev, fake, n=6):
    frames, lmks, _ = ev
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    rows = [np.concatenate([lmks[i], frames[i], fake[i]], axis=1) for i in idx]
    img = ((np.concatenate(rows, axis=0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=sorted(ARMS), required=True)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--eval-every', type=int, default=250)
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--width', type=int, default=16)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--lr-g', type=float, default=1e-4)
    ap.add_argument('--lr-d', type=float, default=4e-4)
    ap.add_argument('--data', default='datasets/synthetic')
    ap.add_argument('--out', required=True)
    ap.add_argument('--vgg19-weights', default=None)
    ap.add_argument('--vggface-weights', default=None)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    tf.random.set_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    cfg = ARMS[a.arm](image_size=a.size, width=a.width, k=a.k)
    full, train = prepare_data(a.data, a.size)
    sampler = EpisodeSampler(train, a.size, a.k, a.batch, random_frames=cfg.random_frames,
                             seed=a.seed, cache=True)
    ev_seen = eval_set(full, a.size, a.k, range(N_TRAIN_PERSONS))
    ev_unseen = eval_set(full, a.size, a.k, range(N_TRAIN_PERSONS, N_TRAIN_PERSONS + N_TEST_PERSONS))
    gan = TalkingHeadGAN(cfg, len(sampler), lr_g=a.lr_g, lr_d=a.lr_d,
                         vgg19_weights=a.vgg19_weights, vggface_weights=a.vggface_weights)

    log = open(os.path.join(a.out, 'log.jsonl'), 'w')
    summary = dict(arm=a.arm, config=cfg.__dict__, args=vars(a),
                   baselines_seen=baselines(ev_seen), baselines_unseen=baselines(ev_unseen), evals=[])
    it = sampler.forever()
    t0 = time.time()
    for step in range(a.steps + 1):
        if step % a.eval_every == 0:
            m_seen, fake_seen = evaluate(gan, ev_seen, a.size)
            m_unseen, fake_unseen = evaluate(gan, ev_unseen, a.size)
            rec = dict(step=step, seen=m_seen, unseen=m_unseen, elapsed=time.time() - t0)
            summary['evals'].append(rec)
            print(json.dumps(rec), flush=True)
            save_grid(os.path.join(a.out, f'seen_{step:05d}.png'), ev_seen, fake_seen)
            save_grid(os.path.join(a.out, f'unseen_{step:05d}.png'), ev_unseen, fake_unseen)
            with open(os.path.join(a.out, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=1)
        if step == a.steps:
            break
        frames, lmks, styles, idx = next(it)
        m = gan.train_step(tf.constant(frames), tf.constant(lmks), tf.constant(styles), tf.constant(idx))
        if step % 25 == 0:
            rec = {k: float(v) for k, v in m.items()}
            rec.update(step=step, elapsed=time.time() - t0)
            log.write(json.dumps(rec) + '\n')
            log.flush()
    gan.save(os.path.join(a.out, 'weights'), 'final')


if __name__ == '__main__':
    main()
