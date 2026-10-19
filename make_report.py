"""Collect smoke_test.py runs into comparison figures and a markdown table.

    python make_report.py --runs runs --out docs/smoke_test --arms legacy fixed_raw_adain fixed
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

LABELS = {'legacy': 'legacy (original code)', 'fixed_raw_adain': 'fixed, but raw AdaIN + old head',
          'fixed': 'fixed'}


def last_grid(run, split):
    return sorted(glob.glob(os.path.join(run, f'{split}_*.png')))[-1]


def compare_figure(runs, arms, split, out):
    """Columns: landmarks | ground truth | one column per arm (final step)."""
    cols = None
    for arm in arms:
        g = np.asarray(Image.open(last_grid(os.path.join(runs, arm), split)))
        w = g.shape[1] // 3
        if cols is None:
            cols = [g[:, :w], g[:, w:2 * w]]
        cols.append(g[:, 2 * w:])
    img = np.concatenate(cols, axis=1)
    head = 22
    canvas = Image.new('RGB', (img.shape[1], img.shape[0] + head), (255, 255, 255))
    canvas.paste(Image.fromarray(img), (0, head))
    d = ImageDraw.Draw(canvas)
    for i, t in enumerate(['landmarks', 'ground truth'] + arms):
        d.text((i * w + 4, 5), t, fill=(0, 0, 0))
    canvas.save(out)


def progress_figure(runs, arm, split, out, steps=(0, 250, 500, 1000, 2000, 3000)):
    """Rows of the first 3 examples over training steps for one arm."""
    cols, w = [], None
    for s in steps:
        p = os.path.join(runs, arm, f'{split}_{s:05d}.png')
        if not os.path.exists(p):
            continue
        g = np.asarray(Image.open(p))
        w = g.shape[1] // 3
        h = g.shape[0] // 2
        if not cols:
            cols += [g[:h, :w], g[:h, w:2 * w]]
        cols.append(g[:h, 2 * w:])
    img = np.concatenate(cols, axis=1)
    head = 22
    canvas = Image.new('RGB', (img.shape[1], img.shape[0] + head), (255, 255, 255))
    canvas.paste(Image.fromarray(img), (0, head))
    d = ImageDraw.Draw(canvas)
    labels = ['landmarks', 'ground truth'] + [f'step {s}' for s in steps
                                              if os.path.exists(os.path.join(runs, arm, f'{split}_{s:05d}.png'))]
    for i, t in enumerate(labels):
        d.text((i * w + 4, 5), t, fill=(0, 0, 0))
    canvas.save(out)


def curves(summaries, out):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for arm, s in summaries.items():
        st = [e['step'] for e in s['evals']]
        ax[0].plot(st, [e['seen']['l1'] for e in s['evals']], label=LABELS.get(arm, arm))
        ax[1].plot(st, [e['seen']['adain_scale_out'] for e in s['evals']], label=LABELS.get(arm, arm))
        ax[2].plot(st, [e['seen']['fake_std'] for e in s['evals']], label=LABELS.get(arm, arm))
    b = next(iter(summaries.values()))['baselines_seen']
    ax[0].axhline(b['l1_mean_of_sources'], ls='--', c='gray', lw=1, label='baseline: mean of K source frames')
    ax[0].axhline(b['l1_dataset_mean'], ls=':', c='gray', lw=1, label='baseline: dataset mean image')
    ax[0].set(title='L1 to ground truth (held-out poses)', xlabel='step', ylim=(0, None))
    ax[1].set(title='|effective AdaIN scale|, output layer', xlabel='step', yscale='log')
    ax[2].set(title='per-image pixel std of output', xlabel='step', ylim=(0, None))
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def table(summaries):
    rows = ['| arm | L1 seen ids | L1 unseen ids | 16px block ratio | output pixel std | AdaIN scale (body / out) |',
            '|---|---|---|---|---|---|']
    for arm, s in summaries.items():
        e = s['evals'][-1]
        rows.append('| {} | {:.3f} | {:.3f} | {:.2f} | {:.3f} | {:.3f} / {:.3f} |'.format(
            LABELS.get(arm, arm), e['seen']['l1'], e['unseen']['l1'], e['seen']['block16_fake'],
            e['seen']['fake_std'], e['seen']['adain_scale_body'], e['seen']['adain_scale_out']))
    s = next(iter(summaries.values()))
    e = s['evals'][-1]
    rows.append('| *ground truth* | – | – | {:.2f} | – | – |'.format(e['seen']['block16_real']))
    rows.append('| *baseline: mean of K source frames* | {:.3f} | {:.3f} | | | |'.format(
        s['baselines_seen']['l1_mean_of_sources'], s['baselines_unseen']['l1_mean_of_sources']))
    rows.append('| *baseline: dataset mean image* | {:.3f} | {:.3f} | | | |'.format(
        s['baselines_seen']['l1_dataset_mean'], s['baselines_unseen']['l1_dataset_mean']))
    return '\n'.join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--out', default='docs/smoke_test')
    ap.add_argument('--arms', nargs='+', default=['legacy', 'fixed_raw_adain', 'fixed'])
    a = ap.parse_args()
    arms = [x for x in a.arms if os.path.exists(os.path.join(a.runs, x, 'summary.json'))]
    os.makedirs(a.out, exist_ok=True)
    summaries = {arm: json.load(open(os.path.join(a.runs, arm, 'summary.json'))) for arm in arms}
    for split in ('seen', 'unseen'):
        compare_figure(a.runs, arms, split, os.path.join(a.out, f'compare_{split}.png'))
    for arm in arms:
        progress_figure(a.runs, arm, 'seen', os.path.join(a.out, f'progress_{arm}.png'))
    curves(summaries, os.path.join(a.out, 'curves.png'))
    md = table(summaries)
    open(os.path.join(a.out, 'results.md'), 'w').write(md + '\n')
    print(md)


if __name__ == '__main__':
    main()
