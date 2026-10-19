"""Episode sampler for meta-learning.

Expected layout (same as the original code):
    <root>/frames/<person_id>/<video_id>/<n>.jpg
    <root>/lndmks/<person_id>/<video_id>/<n>.jpg   (rasterised landmarks, same file names)

Every batch element is one episode from one video: a target frame/landmark
pair and K (frame, landmark) pairs for the embedder.
"""
import os

import numpy as np
from PIL import Image

IMG_EXT = ('.jpg', '.jpeg', '.png')


def load_image(path, size):
    """Read an image as float32 in [-1, 1].

    The original code called ``skimage.transform.resize`` without
    ``preserve_range``, which returned [0, 1] floats; dividing those by 127.5
    put every pixel in [-1, -0.992], i.e. black.
    """
    img = Image.open(path).convert('RGB')
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 127.5 - 1.0


def index_videos(root):
    """Sorted list of (landmark_dir, frame_dir, [file names]) for every video."""
    lmk_root = os.path.join(root, 'lndmks')
    if not os.path.isdir(lmk_root):
        raise FileNotFoundError(lmk_root)
    videos = []
    for cur, dirs, files in os.walk(lmk_root):
        dirs.sort()
        files = sorted(f for f in files if f.lower().endswith(IMG_EXT))
        if files:
            rel = os.path.relpath(cur, lmk_root)
            videos.append((cur, os.path.join(root, 'frames', rel), files))
    return videos


class EpisodeSampler:
    """Yields (frames, landmarks, styles, video_idx) batches.

    frames, landmarks: (B, S, S, 3); styles: (B, K, S, S, 6) with
    [..., :3] = landmark and [..., 3:] = frame; video_idx: (B,) int32.

    random_frames=True (paper): videos are shuffled every epoch and the target
    and the K source frames are drawn at random every time a video is used.
    random_frames=False (original code): videos in directory order, the
    target is always the first file and the sources are the next K files.
    If a video has fewer than K + 1 frames, horizontally flipped copies are
    used to fill up (as in the original code).
    """

    def __init__(self, root, image_size, k, batch_size, random_frames=True, seed=0,
                 min_frames=None, cache=False):
        self.size, self.k, self.batch_size = image_size, k, batch_size
        self.random_frames = random_frames
        self.rng = np.random.default_rng(seed)
        min_frames = (k // 2 + 1) if min_frames is None else min_frames
        self.videos = [v for v in index_videos(root) if len(v[2]) >= min_frames]
        if not self.videos:
            raise ValueError(f'no video with at least {min_frames} frames under {root}')
        self.cache = {} if cache else None

    def __len__(self):
        return len(self.videos)

    def _img(self, path):
        if self.cache is None:
            return load_image(path, self.size)
        if path not in self.cache:
            self.cache[path] = load_image(path, self.size)
        return self.cache[path]

    def _pair(self, vid, name, flip):
        lmk_dir, frm_dir, _ = self.videos[vid]
        f = self._img(os.path.join(frm_dir, name))
        l = self._img(os.path.join(lmk_dir, name))
        if flip:
            f, l = f[:, ::-1], l[:, ::-1]
        return f, l

    def episode(self, vid):
        files = self.videos[vid][2]
        if self.random_frames:
            order = list(self.rng.permutation(len(files)))
            target, sources = order[0], order[1:]
        else:
            target, sources = 0, list(range(1, len(files)))
        src = [(files[i], False) for i in sources]
        if len(src) < self.k:
            src = src + [(n, True) for n, _ in src]
        while len(src) < self.k:
            src = src + src
        src = src[:self.k]
        frame, lmk = self._pair(vid, files[target], False)
        style = [np.concatenate(self._pair(vid, n, fl)[::-1], axis=-1) for n, fl in src]
        return frame, lmk, np.stack(style)

    def epoch(self):
        order = self.rng.permutation(len(self.videos)) if self.random_frames else np.arange(len(self.videos))
        for start in range(0, len(order) - self.batch_size + 1, self.batch_size):
            idx = order[start:start + self.batch_size]
            frames, lmks, styles = zip(*(self.episode(int(i)) for i in idx))
            yield (np.stack(frames), np.stack(lmks), np.stack(styles), idx.astype(np.int32))

    def forever(self):
        while True:
            yield from self.epoch()
