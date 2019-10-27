from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.layers import Add, Input, LeakyReLU, AveragePooling2D, GlobalAveragePooling1D, Dot, Concatenate, Lambda, UpSampling2D, Reshape, ZeroPadding2D, Cropping2D
from keras.layers.core import Activation
from keras.models import Model
from keras.optimizers import Adam
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras_vggface.vggface import VGGFace
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
        x = ConvSN2D(channels, (3,3), strides=(1, 1), padding='same', kernel_initializer = 'he_normal')(x)
        if instance_normalization:
            x = InstanceNormalization(axis=-1)(x)  # might be unnecessary
        x = AveragePooling2D(pool_size=(2, 2))(x)

        shortcut = ConvSN2D(channels, (1, 1), padding='same', kernel_initializer = 'he_normal')(shortcut)
        shortcut = AveragePooling2D(pool_size=(2, 2))(shortcut)

        x = Add()([x, shortcut])
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
        if self.k == 1:
            embeddings = embedding_vectors[0]
        else:
            embeddings = Concatenate(axis=1)(embedding_vectors)  # (BATCH_SIZE, k, 512)
        embedder_embedding = GlobalAveragePooling1D(name='embedder_embedding')(embeddings)  # (BATCH_SIZE, 512)
        
        embedder = Model(inputs=[input_frames, input_landmarks], outputs=embedder_embedding, name='embedder')
        return embedder

    def build_generator(self):
        landmarks = Input(shape=self.input_shape, name='landmarks')
        style_embedding = Input(shape=(512,), name='style_embedding')

        hid = self.downsample(landmarks, 64, instance_normalization=True)
        hid = self.downsample(hid, 128, instance_normalization=True)
        hid = self.downsample(hid, 256, instance_normalization=True)
        hid = SelfAttention(256)(hid)
        hid = self.downsample(hid, 256, instance_normalization=True)
        hid = self.downsample(hid, 256, instance_normalization=True)
        hid = self.downsample(hid, 256, instance_normalization=True)
        hid = self.downsample(hid, 512, instance_normalization=True)
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
            outputs=[fake_frame],
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
        hid = self.downsample(inputs, 64, act_name='fm7')
        hid = self.downsample(hid, 128, act_name='fm6')
        hid = self.downsample(hid, 256, act_name='fm5')
        hid = SelfAttention(256)(hid)
        hid = self.downsample(hid, 256, act_name='fm4')
        hid = self.downsample(hid, 256, act_name='fm3')
        hid = self.downsample(hid, 256, act_name='fm2')
        hid = self.downsample(hid, 512, act_name='fm1')
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

    def _build_embedding_discriminator_model(self, discriminator):
        self.embedding_discriminator = Model(discriminator.get_layer('condition').input, discriminator.get_layer('W_i').output, name='embedding_discriminator')
        return self.embedding_discriminator

    def _build_intermediate_discriminator_model(self, discriminator):
        layer_names = ['fm{}'.format(i) for i in range(1, 8)]
        fm_outputs = [discriminator.get_layer(layer_name).output for layer_name in layer_names]
        self.intermediate_discriminator = Model(discriminator.input[:2], fm_outputs, name='intermediate_discriminator')
        return self.intermediate_discriminator
        
    def compile_models(self, meta, gpus=1):
        hinge_loss='mse'
        # Compile discriminator
        discriminator = self.build_discriminator(meta)
        discriminator.trainable = True
        if gpus > 1:
            parallel_discriminator = multi_gpu_model(discriminator, gpus=4)
            parallel_discriminator.compile(loss=hinge_loss, optimizer=Adam(lr=2e-4, beta_1=1e-5))
        else:
            discriminator.compile(loss=hinge_loss, optimizer=Adam(lr=2e-4, beta_1=1e-5))

        # Compile Combined model to train generator
        embedder = self.build_embedder()
        generator = self.build_generator()
        intermediate_discriminator = self._build_intermediate_discriminator_model(discriminator)
        self._build_embedding_discriminator_model(discriminator)
        intermediate_vgg19 = self.build_intermediate_vgg19_model()
        intermediate_vggface = self.build_intermediate_vggface_model()
        discriminator.trainable = False
        intermediate_discriminator.trainable = False
        intermediate_vgg19.trainable = False
        intermediate_vggface.trainable = False

        input_lndmk = Input(shape=self.input_shape, name='landmarks')
        input_embedder_frames = Input(shape=(self.h, self.w, self.c * self.k), name='input_embedder_frames')
        input_embedder_lndmks = Input(shape=(self.h, self.w, self.c * self.k), name='input_embedder_lndmks')
        
        embedder_embedding = embedder([input_embedder_frames, input_embedder_lndmks])
        fake_frame = generator([input_lndmk, embedder_embedding])
        intermediate_vgg19_outputs = intermediate_vgg19(fake_frame)
        intermediate_vggface_outputs = intermediate_vggface(fake_frame)
        intermediate_discriminator_outputs = intermediate_discriminator([fake_frame, input_lndmk])
        if meta:
            condition = Input(shape=(self.num_videos,), name='condition')
            realicity = discriminator([fake_frame, input_lndmk, condition])
            combined = Model(
                inputs = [input_lndmk, input_embedder_frames, input_embedder_lndmks, condition],
                outputs = intermediate_vgg19_outputs + intermediate_vggface_outputs + [realicity] + intermediate_discriminator_outputs + [embedder_embedding],
                name = 'combined'
                )
            loss_weights = [1.5e-1] * len(intermediate_vgg19_outputs) + [2.5e-2] * len(intermediate_vggface_outputs) + [10] + [10] * len(intermediate_discriminator_outputs) + [10]
        else:
            realicity = discriminator([fake_frame, input_lndmk, embedder_embedding])
            combined = Model(
                inputs = [input_lndmk, input_embedder_frames, input_embedder_lndmks],
                outputs = intermediate_vgg19_outputs + intermediate_vggface_outputs + [realicity] + intermediate_discriminator_outputs,
                name = 'combined'
            )
            loss_weights = [1.5e-1] * len(intermediate_vgg19_outputs) + [2.5e-2] * len(intermediate_vggface_outputs) + [10] + [10] * len(intermediate_discriminator_outputs)

        self.embedder = embedder
        self.generator = generator
        self.combined = combined
        self.discriminator = discriminator

        if gpus > 1:
            parallel_combined = multi_gpu_model(combined, gpus=4)
            parallel_combined.compile(
                loss='mae',
                loss_weights=loss_weights,
                optimizer=Adam(lr=5e-5, beta_1=1e-5)
            )

            self.parallel_combined = parallel_combined
            self.parallel_discriminator = parallel_discriminator

            return parallel_combined, parallel_discriminator, combined, discriminator
        else:
            combined.compile(
                loss='mae',
                loss_weights=loss_weights,
                optimizer=Adam(lr=5e-5, beta_1=1e-5)
            )
            return combined, combined, discriminator, discriminator

    def build_intermediate_vgg19_model(self):
        vgg19 = VGG19(input_shape=self.input_shape, weights='imagenet', include_top=False)
        vgg19.trainable = False
        # Paper says Conv1, 6, 11, 20, 29 VGG19 layers but it isn't clear which layer is which layer
        layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4']
        intermediate_outputs = [vgg19.get_layer(layer_name).output for layer_name in layer_names]
        self.intermediate_vgg19 = Model(vgg19.input, intermediate_outputs, name='intermediate_vgg19')
        return self.intermediate_vgg19

    def build_intermediate_vggface_model(self):
        vggface = VGGFace(input_shape=self.input_shape, weights='vggface', include_top=False)
        vggface.trainable = False
        layer_names = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3']
        intermediate_outputs = [vggface.get_layer(layer_name).output for layer_name in layer_names]
        self.intermediate_vggface = Model(vggface.input, intermediate_outputs, name='intermediate_vggface')
        return self.intermediate_vggface

if __name__ == '__main__':
    import numpy as np
    from keras.optimizers import Adam
    g = GAN((256,256,3),100,8)
#    g.build_intermediate_vgg19_model()
    g.compile_models()
