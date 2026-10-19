"""Networks of Zakharov et al. 2019, "Few-Shot Adversarial Learning of
Realistic Neural Talking Head Models" (Keras 3 / TensorFlow backend).

``Config`` collects every switch that differs between the original
implementation of this repository (``Config.legacy()``) and the fixed one
(``Config.fixed()``, the default). The legacy switches exist only so that the
failure of the original code can be reproduced side by side (see
``smoke_test.py``); they are not meant to be used for real training.
"""
from dataclasses import dataclass, replace

import numpy as np
import keras
from keras import layers, ops

from utils import AdaIN, GlobalSumPooling2D, SelfAttention, conv_sn, dense_sn, instance_norm


@dataclass(frozen=True)
class Config:
    image_size: int = 256
    width: int = 64            # channels: w, 2w, 4w, 8w (paper: w = 64 -> 512-d embedding)
    k: int = 8                 # number of frames given to the embedder

    # --- switches between the original code (legacy) and the fixed code ---
    adain_residual_scale: bool = True   # AdaIN scale = 1 + P e   (legacy: P e)
    fixed_output_head: bool = True      # AdaIN->ReLU->Conv->tanh (legacy: Conv->ReLU->AdaIN->tanh)
    first_block_preact: bool = False    # ReLU on the raw image in the first block (legacy: True)
    g_adv_loss: str = 'neg'             # 'neg': -D(x_hat) (paper); 'hinge': max(0, 1 - D(x_hat))
    vgg_preprocess: bool = True         # caffe preprocessing (BGR, 0-255, mean subtraction)
    vgg_paper_layers: bool = True       # relu{1..5}_1 (legacy: conv1_2, conv2_2, conv3_4/3_3, ...)
    mch_on_mean: bool = True            # L1(W_i, mean_k e_k) * 80 (legacy: sum_k L1(W_i, e_k) * 10)
    random_frames: bool = True          # shuffle videos, random target/source frames each step

    # loss weights (paper values)
    w_vgg19: float = 1.5e-1
    w_vggface: float = 2.5e-2
    w_fm: float = 10.0
    w_mch: float = 80.0

    @property
    def emb_dim(self):
        return 8 * self.width

    @staticmethod
    def fixed(**kw):
        return Config(**kw)

    @staticmethod
    def legacy(**kw):
        base = Config(adain_residual_scale=False, fixed_output_head=False, first_block_preact=True,
                      g_adv_loss='hinge', vgg_preprocess=False, vgg_paper_layers=False,
                      mch_on_mean=False, random_frames=False)
        return replace(base, **kw)


# ---------------------------------------------------------------- blocks

def downsample(x, channels, norm=False, preact=True, fm_out=None):
    """BigGAN-style residual down block. ``fm_out`` collects the pre-activation
    ReLU output (used for the feature-matching loss in the discriminator)."""
    shortcut = x
    if preact:
        x = layers.ReLU()(x)
        if fm_out is not None:
            fm_out.append(x)
    x = conv_sn(channels, 3)(x)
    if norm:
        x = instance_norm()(x)
    x = layers.ReLU()(x)
    x = conv_sn(channels, 3)(x)
    if norm:
        x = instance_norm()(x)
    x = layers.AveragePooling2D(2)(x)
    shortcut = conv_sn(channels, 1)(shortcut)
    shortcut = layers.AveragePooling2D(2)(shortcut)
    return layers.Add()([x, shortcut])


class _ParamSlicer:
    """Hands out consecutive (shift, scale) slices of the flat AdaIN parameter vector."""

    def __init__(self, params):
        self.params = params
        self.offset = 0

    def take(self, channels):
        o = self.offset
        shift = layers.Lambda(lambda p, a=o, b=o + channels: p[:, a:b])(self.params)
        scale = layers.Lambda(lambda p, a=o + channels, b=o + 2 * channels: p[:, a:b])(self.params)
        self.offset += 2 * channels
        return shift, scale


def adain_param_channels(cfg):
    """Channel count of every AdaIN layer in the generator, in order."""
    w = cfg.width
    chans = [8 * w] * 8                                   # 4 residual blocks x 2
    for c_in, c_out in [(8 * w, 8 * w), (8 * w, 4 * w), (4 * w, 2 * w), (2 * w, w)]:
        chans += [c_in, c_out]                            # 4 up blocks x 2
    chans.append(w if cfg.fixed_output_head else 3)       # output head
    return chans


def num_adain_params(cfg):
    return 2 * sum(adain_param_channels(cfg))


# ---------------------------------------------------------------- networks

def build_embedder(cfg, name='embedder'):
    s, w = cfg.image_size, cfg.width
    inp = keras.Input((s, s, 6), name='landmark_frame')   # [..., :3] landmark, [..., 3:] frame
    h = downsample(inp, w, preact=cfg.first_block_preact)
    h = downsample(h, 2 * w)
    h = downsample(h, 4 * w)
    h = SelfAttention(4 * w)(h)
    h = downsample(h, 8 * w)
    h = downsample(h, 8 * w)
    h = downsample(h, 8 * w)
    h = layers.ReLU()(h)
    e = GlobalSumPooling2D()(h)
    return keras.Model(inp, e, name=name)


def build_projector(cfg, name='projector'):
    """Projection matrix P: embedding -> AdaIN parameters (psi_hat = P e_hat)."""
    e = keras.Input((cfg.emb_dim,), name='embedding')
    p = dense_sn(num_adain_params(cfg))(e)
    return keras.Model(e, p, name=name)


def build_generator(cfg, name='generator'):
    """Generator G(landmarks, psi). psi is the flat AdaIN parameter vector,
    so that it can be fine-tuned directly in the few-shot stage (psi')."""
    s, w = cfg.image_size, cfg.width
    rs = cfg.adain_residual_scale
    lmk = keras.Input((s, s, 3), name='landmarks')
    psi = keras.Input((num_adain_params(cfg),), name='adain_params')
    sl = _ParamSlicer(psi)

    h = downsample(lmk, w, norm=True, preact=cfg.first_block_preact)
    h = downsample(h, 2 * w, norm=True)
    h = downsample(h, 4 * w, norm=True)
    h = SelfAttention(4 * w)(h)
    h = downsample(h, 8 * w, norm=True)

    for _ in range(4):                       # residual blocks at the bottleneck
        sc = h
        h = conv_sn(8 * w, 3)(h)
        h = AdaIN(rs)([h, *sl.take(8 * w)])
        h = layers.ReLU()(h)
        h = conv_sn(8 * w, 3)(h)
        h = AdaIN(rs)([h, *sl.take(8 * w)])
        h = layers.Add()([h, sc])

    for i, (c_in, c_out) in enumerate([(8 * w, 8 * w), (8 * w, 4 * w), (4 * w, 2 * w), (2 * w, w)]):
        sc = h
        h = AdaIN(rs)([h, *sl.take(c_in)])
        h = layers.ReLU()(h)
        h = layers.UpSampling2D(2)(h)
        h = conv_sn(c_out, 3)(h)
        h = AdaIN(rs)([h, *sl.take(c_out)])
        h = layers.ReLU()(h)
        h = conv_sn(c_out, 3)(h)
        sc = layers.UpSampling2D(2)(sc)
        sc = conv_sn(c_out, 1)(sc)
        h = layers.Add()([h, sc])
        if i == 1:
            h = SelfAttention(4 * w)(h)

    if cfg.fixed_output_head:
        h = AdaIN(rs)([h, *sl.take(w)])
        h = layers.ReLU()(h)
        h = conv_sn(3, 3)(h)
    else:  # original: Conv(3) -> ReLU -> AdaIN(3 channels) -> tanh
        h = conv_sn(3, 1)(h)
        h = layers.ReLU()(h)
        h = AdaIN(rs)([h, *sl.take(3)])
    out = layers.Activation('tanh', name='fake_frame')(h)
    assert sl.offset == num_adain_params(cfg)
    return keras.Model([lmk, psi], out, name=name)


@keras.saving.register_keras_serializable(package='talking_heads')
class ProjectionHead(layers.Layer):
    """r = v . (W_i + w_0) + b"""

    def build(self, input_shape):
        d = input_shape[0][-1]
        self.w0 = self.add_weight(name='w0', shape=(d,), initializer='zeros')
        self.b = self.add_weight(name='b', shape=(1,), initializer='zeros')

    def call(self, inputs):
        v, w_i = inputs
        return ops.sum(v * (w_i + self.w0), axis=-1, keepdims=True) + self.b


def build_discriminator(cfg, name='discriminator'):
    """Returns a model mapping [frame, landmarks, W_i] -> [score, *features]."""
    s, w = cfg.image_size, cfg.width
    frame = keras.Input((s, s, 3), name='frame')
    lmk = keras.Input((s, s, 3), name='landmarks')
    w_i = keras.Input((cfg.emb_dim,), name='w_i')
    fms = []
    h = layers.Concatenate()([frame, lmk])
    h = downsample(h, w, preact=cfg.first_block_preact)
    h = downsample(h, 2 * w, fm_out=fms)
    h = downsample(h, 4 * w, fm_out=fms)
    h = SelfAttention(4 * w)(h)
    h = downsample(h, 8 * w, fm_out=fms)
    h = downsample(h, 8 * w, fm_out=fms)
    h = downsample(h, 8 * w, fm_out=fms)
    sc = h
    h = layers.ReLU()(h)
    fms.append(h)
    h = conv_sn(8 * w, 3)(h)
    h = layers.ReLU()(h)
    h = conv_sn(8 * w, 3)(h)
    h = layers.Add()([h, sc])
    h = layers.ReLU()(h)
    fms.append(h)
    v = GlobalSumPooling2D()(h)
    score = ProjectionHead()([v, w_i])
    return keras.Model([frame, lmk, w_i], [score] + fms, name=name)


# ---------------------------------------------------------------- perceptual networks

VGG19_URL = ('https://github.com/fchollet/deep-learning-models/releases/download/v0.1/'
             'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
VGGFACE_URL = ('https://github.com/rcmalli/keras-vggface/releases/download/v2.0/'
               'rcmalli_vggface_tf_notop_vgg16.h5')

VGG19_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype='float32')
VGGFACE_MEAN_BGR = np.array([93.5940, 104.7624, 129.1863], dtype='float32')

# Paper: "conv1, 6, 11, 20, 29" of VGG19 and "conv1, 6, 11, 18, 25" of VGGFace.
# These are torchvision module indices and point at relu1_1 ... relu5_1.
# (Keras Conv2D layers include the ReLU.)
VGG19_PAPER = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
VGG19_LEGACY = ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4']
VGGFACE_PAPER = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
VGGFACE_LEGACY = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3']


def _get_weights(path, url):
    if path:
        return path
    return keras.utils.get_file(url.rsplit('/', 1)[-1], url, cache_subdir='models')


def _vggface_vgg16(size):
    inp = keras.Input((size, size, 3))
    h = inp
    for b, (n, c) in enumerate([(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)], start=1):
        for j in range(1, n + 1):
            h = layers.Conv2D(c, 3, padding='same', activation='relu', name=f'conv{b}_{j}')(h)
        h = layers.MaxPooling2D(2, name=f'pool{b}')(h)
    return keras.Model(inp, h, name='vggface_vgg16')


def build_perceptual(cfg, which, weights_path=None):
    """Frozen feature extractor taking images in [-1, 1] (RGB)."""
    s = cfg.image_size
    if which == 'vgg19':
        base = keras.applications.VGG19(include_top=False, weights=None, input_shape=(s, s, 3))
        base.load_weights(_get_weights(weights_path, VGG19_URL))
        names, mean = (VGG19_PAPER if cfg.vgg_paper_layers else VGG19_LEGACY), VGG19_MEAN_BGR
    elif which == 'vggface':
        base = _vggface_vgg16(s)
        base.load_weights(_get_weights(weights_path, VGGFACE_URL))
        names, mean = (VGGFACE_PAPER if cfg.vgg_paper_layers else VGGFACE_LEGACY), VGGFACE_MEAN_BGR
    else:
        raise ValueError(which)
    base.trainable = False
    feat = keras.Model(base.input, [base.get_layer(n).output for n in names])
    feat.trainable = False

    inp = keras.Input((s, s, 3))
    x = inp
    if cfg.vgg_preprocess:
        x = layers.Lambda(lambda t: (t + 1.0) * 127.5)(x)        # [-1, 1] -> [0, 255]
        x = layers.Lambda(lambda t: t[..., ::-1] - mean)(x)       # RGB -> BGR, subtract mean
    model = keras.Model(inp, feat(x), name=f'perceptual_{which}')
    model.trainable = False
    return model
