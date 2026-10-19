"""Meta-learning and few-shot fine-tuning steps (custom training loop).

Losses follow Section 3.2 of the paper:
  L(E, G) = L_CNT + L_ADV + L_MCH
    L_CNT = w_vgg19 * sum_l |VGG19_l(x) - VGG19_l(x_hat)|_1 + w_vggface * (same for VGGFace)
    L_ADV = -D(x_hat) + w_fm * sum_l |D_l(x) - D_l(x_hat)|_1
    L_MCH = w_mch * |W_i - e_hat_i|_1
  L(D)    = max(0, 1 + D(x_hat)) + max(0, 1 - D(x))          (hinge)
"""
import os

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from models import (build_discriminator, build_embedder, build_generator, build_perceptual,
                    build_projector)


def l1(a, b):
    return tf.reduce_mean(tf.abs(a - b))


class TalkingHeadGAN:
    def __init__(self, cfg, num_videos, lr_g=5e-5, lr_d=2e-4, vgg19_weights=None,
                 vggface_weights=None, w_init_stddev=0.02):
        self.cfg = cfg
        self.E = build_embedder(cfg)
        self.P = build_projector(cfg)
        self.G = build_generator(cfg)
        self.D = build_discriminator(cfg)
        # W: one embedding vector per training video (the paper's matrix W)
        self.W = layers.Embedding(num_videos, cfg.emb_dim,
                                  embeddings_initializer=keras.initializers.RandomNormal(stddev=w_init_stddev),
                                  name='W')
        self.W.build((None,))
        self.vgg19 = build_perceptual(cfg, 'vgg19', vgg19_weights)
        self.vggface = build_perceptual(cfg, 'vggface', vggface_weights)
        # beta_1 = 0 as in SAGAN / the paper (the original code used 1e-5)
        self.opt_g = keras.optimizers.Adam(lr_g, beta_1=0.0, beta_2=0.999)
        self.opt_d = keras.optimizers.Adam(lr_d, beta_1=0.0, beta_2=0.999)
        self.g_vars = self.E.trainable_variables + self.P.trainable_variables + self.G.trainable_variables
        self.d_vars = self.D.trainable_variables + self.W.trainable_variables

    # ------------------------------------------------------------ helpers
    def embed(self, styles, training):
        """styles: (B, K, H, W, 6) -> per-frame embeddings (B, K, d) and their mean (B, d)."""
        styles = tf.convert_to_tensor(styles)
        b = tf.shape(styles)[0]
        k = styles.shape[1]
        s = self.cfg.image_size
        e = self.E(tf.reshape(styles, (-1, s, s, 6)), training=training)
        e = tf.reshape(e, (b, k, self.cfg.emb_dim))
        return e, tf.reduce_mean(e, axis=1)

    def generate(self, landmarks, e_hat, training):
        landmarks = tf.convert_to_tensor(landmarks)
        return self.G([landmarks, self.P(e_hat, training=training)], training=training)

    def content_loss(self, x, x_hat):
        cfg = self.cfg
        loss = 0.0
        for net, w in ((self.vgg19, cfg.w_vgg19), (self.vggface, cfg.w_vggface)):
            real = net(x, training=False)
            fake = net(x_hat, training=False)
            loss += w * tf.add_n([l1(tf.stop_gradient(r), f) for r, f in zip(real, fake)])
        return loss

    def g_adv_loss(self, score_fake):
        if self.cfg.g_adv_loss == 'neg':
            return -tf.reduce_mean(score_fake)
        return tf.reduce_mean(tf.nn.relu(1.0 - score_fake))   # keras 'hinge' with y = +1

    # ------------------------------------------------------------ meta-learning
    @tf.function
    def train_step(self, frames, landmarks, styles, video_idx):
        cfg = self.cfg
        w_i = self.W(video_idx)
        with tf.GradientTape() as tape:
            e_k, e_hat = self.embed(styles, training=True)
            x_hat = self.generate(landmarks, e_hat, training=True)
            out_fake = self.D([x_hat, landmarks, tf.stop_gradient(w_i)], training=True)
            out_real = self.D([frames, landmarks, tf.stop_gradient(w_i)], training=True)
            l_cnt = self.content_loss(frames, x_hat)
            l_adv = self.g_adv_loss(out_fake[0])
            l_fm = cfg.w_fm * tf.add_n([l1(tf.stop_gradient(r), f)
                                        for r, f in zip(out_real[1:], out_fake[1:])])
            if cfg.mch_on_mean:
                l_mch = cfg.w_mch * l1(tf.stop_gradient(w_i), e_hat)
            else:  # original: 10 * sum_k |W_i - e_k|
                l_mch = (cfg.w_mch / cfg.k) * tf.add_n(
                    [l1(tf.stop_gradient(w_i), e_k[:, j]) for j in range(cfg.k)])
            loss_g = l_cnt + l_adv + l_fm + l_mch
        grads = tape.gradient(loss_g, self.g_vars)
        self.opt_g.apply_gradients(zip(grads, self.g_vars))

        x_hat = tf.stop_gradient(x_hat)
        with tf.GradientTape() as tape:
            w_i = self.W(video_idx)
            s_real = self.D([frames, landmarks, w_i], training=True)[0]
            s_fake = self.D([x_hat, landmarks, w_i], training=True)[0]
            loss_d = tf.reduce_mean(tf.nn.relu(1.0 - s_real)) + tf.reduce_mean(tf.nn.relu(1.0 + s_fake))
        grads = tape.gradient(loss_d, self.d_vars)
        self.opt_d.apply_gradients(zip(grads, self.d_vars))

        return dict(loss_g=loss_g, l_cnt=l_cnt, l_adv=l_adv, l_fm=l_fm, l_mch=l_mch, loss_d=loss_d,
                    d_real=tf.reduce_mean(s_real), d_fake=tf.reduce_mean(s_fake),
                    e_norm=tf.reduce_mean(tf.norm(e_hat, axis=-1)),
                    w_norm=tf.reduce_mean(tf.norm(w_i, axis=-1)))

    # ------------------------------------------------------------ io
    def save(self, directory, tag):
        os.makedirs(directory, exist_ok=True)
        for name in ('E', 'P', 'G', 'D'):
            getattr(self, name).save_weights(os.path.join(directory, f'{tag}_{name}.weights.h5'))
        np.save(os.path.join(directory, f'{tag}_W.npy'), self.W.embeddings.numpy())

    def load(self, directory, tag, load_W=True):
        for name in ('E', 'P', 'G', 'D'):
            getattr(self, name).load_weights(os.path.join(directory, f'{tag}_{name}.weights.h5'))
        if load_W:
            self.W.embeddings.assign(np.load(os.path.join(directory, f'{tag}_W.npy')))


class FewShotFinetuner:
    """Fine-tuning on a new identity (Section 3.3).

    psi' (generator AdaIN parameters) is initialised to P e_NEW and W' to
    e_NEW; both are then optimised together with G and D. The embedder is not
    used after initialisation.
    """

    def __init__(self, gan, styles, lr_g=5e-5, lr_d=2e-4):
        self.gan = gan
        _, e_new = gan.embed(styles, training=False)
        self.psi = tf.Variable(gan.P(e_new, training=False), name='psi_prime')
        self.w_prime = tf.Variable(e_new, name='w_prime')
        self.opt_g = keras.optimizers.Adam(lr_g, beta_1=0.0, beta_2=0.999)
        self.opt_d = keras.optimizers.Adam(lr_d, beta_1=0.0, beta_2=0.999)
        self.g_vars = gan.G.trainable_variables + [self.psi]
        self.d_vars = gan.D.trainable_variables + [self.w_prime]

    def generate(self, landmarks, training=False):
        b = tf.shape(landmarks)[0]
        psi = tf.repeat(self.psi[:1], b, axis=0)
        return self.gan.G([landmarks, psi], training=training)

    @tf.function
    def train_step(self, frames, landmarks):
        gan, cfg = self.gan, self.gan.cfg
        b = tf.shape(frames)[0]
        w = tf.repeat(self.w_prime[:1], b, axis=0)
        with tf.GradientTape() as tape:
            x_hat = self.generate(landmarks, training=True)
            out_fake = gan.D([x_hat, landmarks, tf.stop_gradient(w)], training=True)
            out_real = gan.D([frames, landmarks, tf.stop_gradient(w)], training=True)
            loss_g = (gan.content_loss(frames, x_hat) + gan.g_adv_loss(out_fake[0])
                      + cfg.w_fm * tf.add_n([l1(tf.stop_gradient(r), f)
                                             for r, f in zip(out_real[1:], out_fake[1:])]))
        grads = tape.gradient(loss_g, self.g_vars)
        self.opt_g.apply_gradients(zip(grads, self.g_vars))
        x_hat = tf.stop_gradient(x_hat)
        with tf.GradientTape() as tape:
            w = tf.repeat(self.w_prime[:1], b, axis=0)
            s_real = gan.D([frames, landmarks, w], training=True)[0]
            s_fake = gan.D([x_hat, landmarks, w], training=True)[0]
            loss_d = tf.reduce_mean(tf.nn.relu(1.0 - s_real)) + tf.reduce_mean(tf.nn.relu(1.0 + s_fake))
        grads = tape.gradient(loss_d, self.d_vars)
        self.opt_d.apply_gradients(zip(grads, self.d_vars))
        return dict(loss_g=loss_g, loss_d=loss_d)
