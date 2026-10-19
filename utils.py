"""Custom layers (Keras 3, TensorFlow backend)."""
import keras
from keras import layers, ops


def conv_sn(filters, kernel_size, **kwargs):
    """Spectrally normalised Conv2D (Miyato et al. 2018)."""
    kwargs.setdefault('padding', 'same')
    kwargs.setdefault('kernel_initializer', 'he_normal')
    return layers.SpectralNormalization(layers.Conv2D(filters, kernel_size, **kwargs))


def dense_sn(units, **kwargs):
    return layers.SpectralNormalization(layers.Dense(units, **kwargs))


def instance_norm():
    # GroupNormalization with groups=-1 is InstanceNormalization (affine).
    return layers.GroupNormalization(groups=-1, epsilon=1e-5)


@keras.saving.register_keras_serializable(package='talking_heads')
class AdaIN(layers.Layer):
    """Adaptive instance normalisation.

    y = scale * (x - mean_hw(x)) / sqrt(var_hw(x) + eps) + shift

    ``scale`` and ``shift`` have shape (B, C) and are produced from the
    identity embedding by the generator's projection matrix P.

    If ``residual_scale`` is True the layer applies ``1 + scale`` instead of
    ``scale``. With the raw parameterisation an embedding near zero makes the
    scale near zero, which zeroes every residual branch and the gradients that
    flow through it (this was the failure mode of the original
    implementation). With ``1 + scale`` an embedding near zero degrades
    gracefully to plain instance normalisation.
    """

    def __init__(self, residual_scale=True, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.residual_scale = residual_scale
        self.epsilon = epsilon

    def call(self, inputs):
        x, shift, scale = inputs
        mean = ops.mean(x, axis=(1, 2), keepdims=True)
        var = ops.var(x, axis=(1, 2), keepdims=True)
        x = (x - mean) / ops.sqrt(var + self.epsilon)
        scale = ops.expand_dims(ops.expand_dims(scale, 1), 1)
        shift = ops.expand_dims(ops.expand_dims(shift, 1), 1)
        if self.residual_scale:
            scale = 1.0 + scale
        return scale * x + shift

    def get_config(self):
        cfg = super().get_config()
        cfg.update(residual_scale=self.residual_scale, epsilon=self.epsilon)
        return cfg


@keras.saving.register_keras_serializable(package='talking_heads')
class SelfAttention(layers.Layer):
    """SAGAN self-attention block; gamma starts at 0 so the block is an identity at init."""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.f = layers.Conv2D(max(channels // 8, 1), 1)
        self.g = layers.Conv2D(max(channels // 8, 1), 1)
        self.h = layers.Conv2D(channels, 1)

    def build(self, input_shape):
        self.f.build(input_shape)
        self.g.build(input_shape)
        self.h.build(input_shape)
        self.gamma = self.add_weight(name='gamma', shape=(1,), initializer='zeros')

    def call(self, x):
        b = ops.shape(x)[0]
        hh, ww, c = x.shape[1], x.shape[2], x.shape[3]
        n = hh * ww
        f = ops.reshape(self.f(x), (b, n, -1))
        g = ops.reshape(self.g(x), (b, n, -1))
        h = ops.reshape(self.h(x), (b, n, c))
        beta = ops.softmax(ops.matmul(g, ops.transpose(f, (0, 2, 1))), axis=-1)
        o = ops.reshape(ops.matmul(beta, h), (b, hh, ww, c))
        return x + self.gamma * o

    def get_config(self):
        cfg = super().get_config()
        cfg.update(channels=self.channels)
        return cfg


@keras.saving.register_keras_serializable(package='talking_heads')
class GlobalSumPooling2D(layers.Layer):
    def call(self, x):
        return ops.sum(x, axis=(1, 2))
