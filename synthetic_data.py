"""Procedurally generated "talking heads" for CPU smoke tests.

Each identity has its own colours (skin, hair, eyes, background, clothes),
face shape and hair style; each frame has its own pose (translation,
rotation, scale, yaw-like shift of the inner face) and expression (mouth and
eye opening). The landmark image encodes only geometry, the frame encodes
geometry + identity appearance, so the task has the same structure as
VoxCeleb: G must take geometry from the landmarks and appearance from the
embedder.

    python synthetic_data.py --out datasets/synthetic --size 64
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

SS = 4  # supersampling factor


def _ellipse(cx, cy, rx, ry, n=48, t0=0.0, t1=2 * np.pi):
    t = np.linspace(t0, t1, n)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=1)


def sample_identity(rng):
    def col(lo=0, hi=255):
        return tuple(int(c) for c in rng.integers(lo, hi, 3))
    return dict(skin=col(90, 255), hair=col(0, 200), eye=col(0, 255), bg=col(0, 255),
                shirt=col(0, 255), rx=rng.uniform(0.20, 0.27), ry=rng.uniform(0.27, 0.34),
                hair_style=int(rng.integers(0, 3)))


def sample_pose(rng):
    return dict(dx=rng.uniform(-0.07, 0.07), dy=rng.uniform(-0.05, 0.05),
                rot=rng.uniform(-0.35, 0.35), scale=rng.uniform(0.9, 1.1),
                yaw=rng.uniform(-0.6, 0.6), mouth=rng.uniform(0.0, 1.0), eyes=rng.uniform(0.15, 1.0))


def _geometry(idt, pose):
    """Facial parts as polygons in unit coordinates (centre 0, 0)."""
    rx, ry, yaw = idt['rx'], idt['ry'], pose['yaw']
    sh = yaw * rx * 0.35                                 # inner-face shift (cheap yaw)
    g = {}
    g['face'] = _ellipse(0, 0, rx, ry)
    g['jaw'] = _ellipse(0, 0, rx, ry, 24, 0.1 * np.pi, 0.9 * np.pi)
    ex, ey = 0.42 * rx, -0.18 * ry
    eh = 0.06 * ry * pose['eyes']
    for s, name in ((-1, 'eye_l'), (1, 'eye_r')):
        g[name] = _ellipse(s * ex + sh, ey, 0.22 * rx * (1 - 0.25 * s * yaw), max(eh, 0.006), 20)
        g['brow' + name[3:]] = np.array([[s * ex + sh - 0.22 * rx, ey - 0.16 * ry],
                                         [s * ex + sh, ey - 0.21 * ry],
                                         [s * ex + sh + 0.22 * rx, ey - 0.16 * ry]])
    g['nose'] = np.array([[sh * 1.3, ey + 0.05 * ry], [sh * 1.6 + 0.04 * rx, 0.22 * ry],
                          [sh * 1.3 - 0.06 * rx, 0.25 * ry]])
    mh = 0.02 * ry + 0.14 * ry * pose['mouth']
    g['mouth'] = _ellipse(sh, 0.52 * ry, 0.34 * rx, mh, 24)
    g['hair'] = _ellipse(0, -0.12 * ry, rx * 1.12, ry * [0.95, 1.05, 1.2][idt['hair_style']], 48,
                         np.pi * [1.0, 0.9, 0.75][idt['hair_style']],
                         np.pi * [2.0, 2.1, 2.25][idt['hair_style']])
    return g


def _to_px(poly, pose, size):
    c, s = np.cos(pose['rot']), np.sin(pose['rot'])
    p = poly @ np.array([[c, s], [-s, c]]) * pose['scale']
    p = (p + [0.5 + pose['dx'], 0.47 + pose['dy']]) * size * SS
    return [tuple(q) for q in p]


def render(idt, pose, size):
    """Returns (frame, landmark) as uint8 RGB arrays of shape (size, size, 3)."""
    big = size * SS
    g = _geometry(idt, pose)
    px = {k: _to_px(v, pose, size) for k, v in g.items()}

    frame = Image.new('RGB', (big, big), idt['bg'])
    d = ImageDraw.Draw(frame)
    neck = _to_px(np.array([[-0.12, 0.2], [0.12, 0.2], [0.14, 0.45], [-0.14, 0.45]]), pose, size)
    d.rectangle([0, int(0.86 * big), big, big], fill=idt['shirt'])
    d.polygon(neck, fill=tuple(int(0.8 * c) for c in idt['skin']))
    d.polygon(px['hair'], fill=idt['hair'])
    d.polygon(px['face'], fill=idt['skin'])
    for e in ('eye_l', 'eye_r'):
        d.polygon(px[e], fill=(245, 245, 245))
        cx, cy = np.mean(px[e], axis=0)
        r = 0.035 * size * SS * pose['scale'] * max(pose['eyes'], 0.3)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=idt['eye'])
    for b in ('brow_l', 'brow_r'):
        d.line(px[b], fill=idt['hair'], width=SS * 2)
    d.line(px['nose'], fill=tuple(int(0.6 * c) for c in idt['skin']), width=SS)
    d.polygon(px['mouth'], fill=(120, 20, 30))

    lmk = Image.new('RGB', (big, big), (0, 0, 0))
    d = ImageDraw.Draw(lmk)
    w = SS
    d.line(px['jaw'], fill=(255, 255, 255), width=w)
    d.line(px['brow_l'] + [], fill=(255, 128, 0), width=w)
    d.line(px['brow_r'], fill=(255, 128, 0), width=w)
    d.line(px['eye_l'], fill=(0, 255, 0), width=w)
    d.line(px['eye_r'], fill=(0, 255, 0), width=w)
    d.line(px['nose'], fill=(0, 128, 255), width=w)
    d.line(px['mouth'], fill=(255, 0, 255), width=w)

    frame = frame.resize((size, size), Image.LANCZOS)
    lmk = lmk.resize((size, size), Image.LANCZOS)
    return np.asarray(frame), np.asarray(lmk)


def write_dataset(out, size=64, persons=24, videos_per_person=2, frames_per_video=9, seed=0):
    rng = np.random.default_rng(seed)
    for p in range(persons):
        idt = sample_identity(rng)
        for v in range(videos_per_person):
            for kind in ('frames', 'lndmks'):
                os.makedirs(os.path.join(out, kind, f'id{p:05d}', f'v{v:02d}'), exist_ok=True)
            for f in range(frames_per_video):
                frame, lmk = render(idt, sample_pose(rng), size)
                Image.fromarray(frame).save(os.path.join(out, 'frames', f'id{p:05d}', f'v{v:02d}', f'{f}.png'))
                Image.fromarray(lmk).save(os.path.join(out, 'lndmks', f'id{p:05d}', f'v{v:02d}', f'{f}.png'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='datasets/synthetic')
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--persons', type=int, default=24)
    ap.add_argument('--videos-per-person', type=int, default=2)
    ap.add_argument('--frames-per-video', type=int, default=9)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    write_dataset(a.out, a.size, a.persons, a.videos_per_person, a.frames_per_video, a.seed)
