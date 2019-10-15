import keras.backend as K
from keras.layers import Add, Input, LeakyReLU, AveragePooling2D, GlobalAveragePooling1D, Dot, Concatenate, Lambda, UpSampling2D, Reshape, ZeroPadding2D, Cropping2D
from keras.layers.core import Activation
from keras.models import Model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import tensorflow as tf

from utils import GlobalSumPooling2D, ConvSN2D, DenseSN, AdaIN, Bias, SelfAttention
from utils import AdaInstanceNormalization


class GAN:
    def __init__(self, input_shape, num_videos, k):
        """
        input_shape: (H, W, 3)
        k: The number of image pairs into embedder
        """
        self.h, self.w, self.c = self.input_shape = input_shape
        self.num_videos = num_videos
        self.k = k

    def downsample(self, x, channels, instance_normalization=False, act_name=None):
        """  Downsampling is similar to an implementation in BigGAN """
        shortcut = x
 
        x = LeakyReLU(alpha=0.2)(x)
        x = ConvSN2D(channels, (3,3), strides=(1, 1), padding='same',)(x)
        if instance_normalization:
            x = InstanceNormalization(axis=-1)(x)  # might be unnecessary
        x = LeakyReLU(alpha=0.2, name=act_name)(x)
        act = x
        x = ConvSN2D(channels, (3,3), strides=(1, 1), padding='same', kernel_initializer = 'he_normal')(x)
        if instance_normalization:
            x = InstanceNormalization(axis=-1)(x)  # might be unnecessary
        x = AveragePooling2D(pool_size=(2, 2))(x)

        shortcut = ConvSN2D(channels, (1, 1), padding='same', kernel_initializer = 'he_normal')(shortcut)
        shortcut = AveragePooling2D(pool_size=(2, 2))(shortcut)

        x = Add()([x, shortcut])
        if act_name:
            return x, act
        return x

    def resblock(self, x, channels):
        shortcut = x

        x = ConvSN2D(channels, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = InstanceNormalization(axis=-1)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = ConvSN2D(channels, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = InstanceNormalization(axis=-1)(x)

        if shortcut.shape[-1] != channels:
            shortcut = ConvSN2D(channels, (1, 1), padding='same', kernel_initializer='he_normal')(shortcut)

        x = Add()([x, shortcut])
        return x

    def upsample(self, x, channels, style_embedding, instance_normalization=False, name='0'):
        def adain(inputs):
            '''
            Borrowed from https://github.com/jonrei/tf-AdaIN
            Normalizes the `content_features` with scaling and offset from `style_features`.
            See "5. Adaptive Instance Normalization" in https://arxiv.org/abs/1703.06868 for details.
            '''
            epsilon = 1e-5
            content_features, style_mean, style_std = inputs
            content_mean, content_variance = tf.nn.moments(content_features, [1, 2], keep_dims=True)
            normalized_content_features = tf.nn.batch_normalization(content_features,
                                                                    content_mean,
                                                                    content_variance,
                                                                    style_mean, 
                                                                    style_std,
                                                                    epsilon)
            return normalized_content_features

        shortcut = x
        style_embedding = DenseSN(512, kernel_initializer='he_normal')(style_embedding)
        style_embedding = DenseSN(512, kernel_initializer='he_normal')(style_embedding)
        style_mean = DenseSN(channels, name='style_mean'+name, kernel_initializer='he_normal')(style_embedding)
        style_std  = DenseSN(channels, name='style_std'+ name, kernel_initializer='he_normal')(style_embedding)
        style_mean = Reshape((1, 1, channels))(style_mean)
        style_std  = Reshape((1, 1, channels))(style_std)

        if instance_normalization:
            x = InstanceNormalization(axis=-1)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = ConvSN2D(channels, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        if instance_normalization:
            x = InstanceNormalization(axis=-1)(x)
#        x = Lambda(adain)([x, style_mean, style_std])
#        x = AdaIN()([x, style_mean, style_std])
        x = AdaInstanceNormalization()([x, style_mean, style_std])
        x = LeakyReLU(alpha=0.2)(x)
        x = ConvSN2D(channels, (3, 3), padding='same', kernel_initializer='he_normal')(x)

        shortcut = UpSampling2D(size=(2, 2))(shortcut)
        shortcut = ConvSN2D(channels, (1, 1), kernel_initializer='he_normal')(shortcut)

        x = Add()([x, shortcut])
        return x
        

    def embed(self, name='embed'):
        input_frame = Input(shape=self.input_shape)
        input_landmark = Input(shape=self.input_shape)
        inputs = Concatenate()([input_frame, input_landmark])
        hid = self.downsample(inputs, 64)
        hid = self.downsample(hid, 128)
        hid = self.downsample(hid, 512)
        hid = SelfAttention(512)(hid)
        hid = self.downsample(hid, 512)
        hid = self.downsample(hid, 512)
        hid = self.downsample(hid, 512)
        hid = Activation('relu')(hid)
        e = GlobalSumPooling2D()(hid)

        embedder = Model(inputs=[input_frame, input_landmark], outputs=e, name=name)
        return embedder

    def build_embedder(self):
        """ k frames from the same sequence """
        input_frames = Input(shape=(self.h, self.w, self.c * self.k))
        input_landmarks = Input(shape=(self.h, self.w, self.c * self.k))

        input_frames_splt = [Lambda(lambda x: x[:, :, :, 3*i:3*(i+1)])(input_frames) for i in range(self.k)]
        input_landmarks_splt = [Lambda(lambda x: x[:,:, :, 3*i:3*(i+1)])(input_landmarks) for i in range(self.k)]
        single_embedder = self.embed(name='single_embedder')
        embedding_vectors = [single_embedder([frame, landmark]) for frame, landmark in zip(input_frames_splt, input_landmarks_splt)] # List of (BATCH_SIZE, 512,)
        embedding_vectors = [Lambda(lambda x: K.expand_dims(x, axis=1))(vector) for vector in embedding_vectors]  # List of (BATCH_SIZE, 1, 512)
        embeddings = Concatenate(axis=1)(embedding_vectors)  # (BATCH_SIZE, k, 512)
        embedder_embedding = GlobalAveragePooling1D(name='embedder_embedding')(embeddings)
        
        embedder = Model(inputs=[input_frames, input_landmarks], outputs=embedder_embedding, name='embedder')
        return embedder

    def build_generator(self):
        landmarks = Input(shape=self.input_shape, name='landmarks')
        style_embedding = Input(shape=(512,), name='style_embedding')

        hid, fm7 = self.downsample(landmarks, 64, instance_normalization=True, act_name='fm7')
        hid, fm6 = self.downsample(hid, 128, instance_normalization=True, act_name='fm6')
        hid, fm5 = self.downsample(hid, 256, instance_normalization=True, act_name='fm5')
        hid = SelfAttention(256)(hid)
        hid, fm4 = self.downsample(hid, 256, instance_normalization=True, act_name='fm4')
        hid, fm3 = self.downsample(hid, 256, instance_normalization=True, act_name='fm3')
        hid, fm2 = self.downsample(hid, 256, instance_normalization=True, act_name='fm2')
        hid, fm1 = self.downsample(hid, 512, instance_normalization=True, act_name='fm1')
 #       hid = ZeroPadding2D(padding=((0, 1),(0, 1)))(hid)  # For input size is 224p
        
        hid = self.resblock(hid, 512)
        hid = self.resblock(hid, 512)
        
        hid = self.upsample(hid, 256, style_embedding, instance_normalization=True, name='4')
        hid = self.upsample(hid, 256, style_embedding, instance_normalization=True, name='8')
#        hid = Cropping2D(cropping=((0,1), (0,1)))(hid)  # For input size is 224p
        hid = self.upsample(hid, 256, style_embedding, instance_normalization=True, name='16')
        hid = self.upsample(hid, 256, style_embedding, instance_normalization=True, name='32')
        hid = self.upsample(hid, 128, style_embedding, instance_normalization=True, name='64')
        hid = SelfAttention(128)(hid)
        hid = self.upsample(hid, 64, style_embedding, instance_normalization=True, name='128')
        hid = self.upsample(hid, 64, style_embedding, instance_normalization=True, name='256')
        hid = ConvSN2D(3, (1, 1), padding='same', kernel_initializer='he_normal')(hid)
        fake_frame = Activation('tanh', name='fake_frame')(hid)

        generator = Model(
            inputs=[landmarks, style_embedding],
            outputs=[fake_frame, fm1, fm2, fm3, fm4, fm5, fm6, fm7],
            name='generator')
        return generator
    
    def build_discriminator(self, meta):
        """
        realicity = dot(v, w)
        w = Pc+w_0 (P: Projection Matrix; c: one-hot condition)
        """
        input_frame = Input(shape=self.input_shape, name='frames')
        input_landmark = Input(shape=self.input_shape, name='landmarks')
        inputs = Concatenate()([input_frame, input_landmark])
        hid, fm7 = self.downsample(inputs, 64, act_name='fm7')
        hid, fm6 = self.downsample(hid, 128, act_name='fm6')
        hid, fm5 = self.downsample(hid, 256, act_name='fm5')
        hid = SelfAttention(256)(hid)
        hid, fm4 = self.downsample(hid, 256, act_name='fm4')
        hid, fm3 = self.downsample(hid, 256, act_name='fm3')
        hid, fm2 = self.downsample(hid, 256, act_name='fm2')
        hid, fm1 = self.downsample(hid, 512, act_name='fm1')
        hid = Activation('relu')(hid)
        v = GlobalSumPooling2D()(hid)

        if meta:
            condition = Input(shape=(self.num_videos,), name='condition')
            W_i = DenseSN(512, use_bias=False, name='W_i')(condition)  # Projection Matrix, P
        else:
            W_i  = Input(shape=(512,), name='e_NEW')
        w = Bias(name='w')(W_i)  # w = W_i + w_0
        
        innerproduct = Dot(axes=-1)([v, w])
        realicity = Bias(name='realicity')(innerproduct)
        
        if meta:
            inputs = [input_frame, input_landmark, condition]
        else:
            inputs = [input_frame, input_landmark, W_i]
        discriminator = Model(
            inputs=inputs,
            outputs=[realicity],
            name='discriminator'
        )
        return discriminator

    def build_models(self, meta=True):
        embedder = self.build_embedder()
        generator = self.build_generator()
        discriminator = self.build_discriminator(meta)

        # Generator Input
        input_lndmk = Input(shape=self.input_shape, name='landmarks')
        input_embedder_frames = Input(shape=(self.h, self.w, self.c * self.k), name='input_embedder_frames')
        input_embedder_lndmks = Input(shape=(self.h, self.w, self.c * self.k), name='input_embedder_lndmks')
        
        embedder_embedding = embedder([input_embedder_frames, input_embedder_lndmks])
        fake_frame, g_fm1, g_fm2, g_fm3, g_fm4, g_fm5, g_fm6, g_fm7 = generator([input_lndmk, embedder_embedding])
        if meta:
            condition = Input(shape=(self.num_videos,), name='condition')
            realicity = discriminator([fake_frame, input_lndmk, condition])
            combined = Model(
                inputs=[input_lndmk, input_embedder_frames, input_embedder_lndmks, condition],
                outputs=[fake_frame, realicity, embedder_embedding, g_fm1, g_fm2, g_fm3, g_fm4, g_fm5, g_fm6, g_fm7],
                name='combined'
            )
        else:
            realicity = discriminator([fake_frame, input_lndmk, embedder_embedding])
            combined = Model(
                inputs=[input_lndmk, input_embedder_frames, input_embedder_lndmks],
                outputs=[fake_frame, realicity, g_fm1, g_fm2, g_fm3, g_fm4, g_fm5, g_fm6, g_fm7],
                name='combined'
            )

        return combined, discriminator

if __name__ == '__main__':
    import numpy as np
    from keras.optimizers import Adam
    g = GAN((256,256,3),100,8)
    gen = g.build_generator()
    emb = g.build_embedder()
    edg = emb.predict([np.random.uniform(-1, 1, (4, 256, 256, 24)), np.random.uniform(-1, 1, (4, 256, 256, 24))])
    gen.compile(loss='mse', optimizer=Adam())
    geninput = [np.random.uniform(-1, 1, (4, 256, 256, 3)), edg]
    genoutput = gen.predict_on_batch(geninput)
    print(genoutput)
    print(gen.train_on_batch(geninput, genoutput))
    import sys;sys.exit()
