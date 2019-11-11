from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.layers import Dense, Add, Input, ReLU, AveragePooling2D, GlobalAveragePooling1D, Dot, Concatenate, Lambda, UpSampling2D, Reshape, ZeroPadding2D, Cropping2D
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

    def downsample(self, x, channels, i_norm=False, act_name=None):
        """  Downsampling is similar to an implementation in BigGAN """
        shortcut = x
 
        x = ReLU()(x)
        x = ConvSN2D(channels, (3,3), strides=(1, 1), padding='same',)(x)
        if i_norm:
            x = InstanceNormalization(axis=-1)(x)  # might be unnecessary
        x = ReLU(name=act_name)(x)
        x = ConvSN2D(channels, (3,3), strides=(1, 1), padding='same', kernel_initializer = 'he_normal')(x)
        if i_norm:
            x = InstanceNormalization(axis=-1)(x)  # might be unnecessary
        x = AveragePooling2D(pool_size=(2, 2))(x)

        shortcut = ConvSN2D(channels, (1, 1), padding='same', kernel_initializer = 'he_normal')(shortcut)
        shortcut = AveragePooling2D(pool_size=(2, 2))(shortcut)

        x = Add()([x, shortcut])
        return x

    def resblock(self, x, channels, mean0, std0, mean1, std1):
        shortcut = x

        x = ConvSN2D(channels, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = AdaInstanceNormalization()([x, mean0, std0])
        x = ReLU()(x)
        x = ConvSN2D(channels, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = AdaInstanceNormalization()([x, mean1, std1])

        if shortcut.shape[-1] != channels:
            shortcut = ConvSN2D(channels, (1, 1), padding='same', kernel_initializer='he_normal')(shortcut)

        x = Add()([x, shortcut])
        return x

    def upsample(self, x, channels, mean0, std0, mean1, std1):
        shortcut = x

        x = AdaInstanceNormalization()([x, mean0, std0])
        x = ReLU()(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = ConvSN2D(channels, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = AdaInstanceNormalization()([x, mean1, std1])
        x = ReLU()(x)
        x = ConvSN2D(channels, (3, 3), padding='same', kernel_initializer='he_normal')(x)

        shortcut = UpSampling2D(size=(2, 2))(shortcut)
        shortcut = ConvSN2D(channels, (1, 1), kernel_initializer='he_normal')(shortcut)

        x = Add()([x, shortcut])
        return x

    def build_embedder(self, name='embedder'):
        h, w, c = self.input_shape
        
        input_landmark_frame = Input(shape=(h, w, c * 2))  # [:,:,3:]: landmark; [:,:,4:]: frame
        hid = self.downsample(input_landmark_frame, 64)
        hid = self.downsample(hid, 128)
        hid = self.downsample(hid, 256)
        hid = SelfAttention(256)(hid)
        hid = self.downsample(hid, 512)
        hid = self.downsample(hid, 512)
        hid = self.downsample(hid, 512)
        hid = ReLU()(hid)
        embedding = GlobalSumPooling2D()(hid)

        embedder = Model(inputs=input_landmark_frame, outputs=embedding, name=name)
        return embedder

    def build_embedder_depricated(self):
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
        average_embedding = GlobalAveragePooling1D(name='average_embedding')(embeddings)  # (BATCH_SIZE, 512)
        
        embedder = Model(inputs=[input_frames, input_landmarks], outputs=average_embedding, name='embedder')
        return embedder

    def build_generator(self):
        landmarks = Input(shape=self.input_shape, name='landmarks')
        style_embedding = Input(shape=(512,), name='style_embedding')

        def get_adain_ix(l,):
            adain_channels = [
                # res block
                (512, 512),
                (512, 512),
                (512, 512),
                (512, 512),
                # res up
                (512, 256),
                (512, 512),
                (256, 128),
                (128, 64),
                (3, 3)
            ]
            prev_total=2*sum(list(sum(adain_channels, ()))[:2 * l])
            return [(prev_total, prev_total + adain_channels[l][0]),
                    (prev_total + adain_channels[l][0], prev_total +2 *  adain_channels[l][0]),
                    (prev_total + 2 * adain_channels[l][0], prev_total + 2 * adain_channels[l][0] + adain_channels[l][1]),
                    (prev_total + 2 * adain_channels[l][0] + adain_channels[l][1], prev_total + 2 * adain_channels[l][0] + 2 * adain_channels[l][1])
            ]
        adain_param_length = 512*22+256*4+128*4+64*2+6
        adain_params = DenseSN(adain_param_length)(style_embedding)
        adain_params = Reshape((1, 1, adain_param_length))(adain_params)

        # Slice AdaIN Parameters
        mean_b00 = Lambda(lambda x: x[:,:,:, 0    :512])(adain_params)
        std_b00  = Lambda(lambda x: x[:,:,:, 512  :512*2])(adain_params)
        mean_b01 = Lambda(lambda x: x[:,:,:, 512*2:512*3])(adain_params)
        std_b01  = Lambda(lambda x: x[:,:,:, 512*3:512*4])(adain_params)

        mean_b10 = Lambda(lambda x: x[:,:,:, 512*4:512*5])(adain_params)
        std_b10  = Lambda(lambda x: x[:,:,:, 512*5:512*6])(adain_params)
        mean_b11 = Lambda(lambda x: x[:,:,:, 512*6:512*7])(adain_params)
        std_b11  = Lambda(lambda x: x[:,:,:, 512*7:512*8])(adain_params)

        mean_b20 = Lambda(lambda x: x[:,:,:, 512*8:512*9])(adain_params)
        std_b20  = Lambda(lambda x: x[:,:,:, 512*9:512*10])(adain_params)
        mean_b21 = Lambda(lambda x: x[:,:,:, 512*10:512*11])(adain_params)
        std_b21  = Lambda(lambda x: x[:,:,:, 512*11:512*12])(adain_params)

        mean_b30 = Lambda(lambda x: x[:,:,:, 512*12:512*13])(adain_params)
        std_b30  = Lambda(lambda x: x[:,:,:, 512*13:512*14])(adain_params)
        mean_b31 = Lambda(lambda x: x[:,:,:, 512*14:512*15])(adain_params)
        std_b31  = Lambda(lambda x: x[:,:,:, 512*15:512*16])(adain_params)

        mean_u00 = Lambda(lambda x: x[:,:,:, 512*16:512*17])(adain_params)
        std_u00  = Lambda(lambda x: x[:,:,:, 512*17:512*18])(adain_params)
        mean_u01 = Lambda(lambda x: x[:,:,:, 512*18:512*19])(adain_params)
        std_u01  = Lambda(lambda x: x[:,:,:, 512*19:512*20])(adain_params)

        mean_u10 = Lambda(lambda x: x[:,:,:, 512*20    :512*21])(adain_params)
        std_u10  = Lambda(lambda x: x[:,:,:, 512*21    :512*22])(adain_params)
        mean_u11 = Lambda(lambda x: x[:,:,:, 512*22    :512*22+256])(adain_params)
        std_u11  = Lambda(lambda x: x[:,:,:, 512*22+256:512*22+256*2])(adain_params)

        mean_u20 = Lambda(lambda x: x[:,:,:, 512*22+256*2    :512*22+256*3])(adain_params)
        std_u20  = Lambda(lambda x: x[:,:,:, 512*22+256*3    :512*22+256*4])(adain_params)
        mean_u21 = Lambda(lambda x: x[:,:,:, 512*22+256*4    :512*22+256*4+128])(adain_params)
        std_u21  = Lambda(lambda x: x[:,:,:, 512*22+256*4+128:512*22+256*4+128*2])(adain_params)

        mean_u30 = Lambda(lambda x: x[:,:,:, 512*22+256*4+128*2   :512*22+256*4+128*3])(adain_params)
        std_u30  = Lambda(lambda x: x[:,:,:, 512*22+256*4+128*3   :512*22+256*4+128*4])(adain_params)
        mean_u31 = Lambda(lambda x: x[:,:,:, 512*22+256*4+128*4   :512*22+256*4+128*4+64])(adain_params)
        std_u31  = Lambda(lambda x: x[:,:,:, 512*22+256*4+128*4+64:512*22+256*4+128*4+64*2])(adain_params)

        mean_u4 = Lambda(lambda x: x[:,:,:, 512*22+256*4+128*4+64*2  :512*22+256*4+128*4+64*2+3])(adain_params)
        std_u4  = Lambda(lambda x: x[:,:,:, 512*22+256*4+128*4+64*2+3:512*22+256*4+128*4+64*2+6])(adain_params)

        # Main forward
        hid = self.downsample(landmarks, 64, i_norm=True)
        hid = self.downsample(hid, 128, i_norm=True)
        hid = self.downsample(hid, 256, i_norm=True)
#        hid = SelfAttention(256)(hid)
        hid = self.downsample(hid, 512, i_norm=True)
        
        hid = self.resblock(hid, 512, mean_b00, std_b00, mean_b01, std_b01)
        hid = self.resblock(hid, 512, mean_b10, std_b10, mean_b11, std_b11)
        hid = self.resblock(hid, 512, mean_b20, std_b20, mean_b21, std_b21)
        hid = self.resblock(hid, 512, mean_b30, std_b30, mean_b31, std_b31)
        
        hid = self.upsample(hid, 512, mean_u00, std_u00, mean_u01, std_u01)
        hid = self.upsample(hid, 256, mean_u10, std_u10, mean_u11, std_u11)
#        hid = SelfAttention(256)(hid)
        hid = self.upsample(hid, 128, mean_u20, std_u20, mean_u21, std_u21)
        hid = self.upsample(hid, 64,  mean_u30, std_u30, mean_u31, std_u31)
        hid = ConvSN2D(3, (1, 1), padding='same', kernel_initializer='he_normal')(hid)
        hid = ReLU()(hid)
        hid = AdaInstanceNormalization()([hid, mean_u4, std_u4])
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
            W_i = Dense(512, use_bias=False, name='W_i')(condition)  # Projection Matrix, P
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
        # Compile discriminator
        discriminator = self.build_discriminator(meta)
        discriminator.trainable = True
        if gpus > 1:
            parallel_discriminator = multi_gpu_model(discriminator, gpus=4)
            parallel_discriminator.compile(loss='hinge', optimizer=Adam(lr=2e-4, beta_1=1e-5))
        else:
            discriminator.compile(loss='hinge', optimizer=Adam(lr=2e-4, beta_1=1e-5))

        # Compile Combined model to train generator
        embedder = self.build_embedder()
        generator = self.build_generator()
        intermediate_discriminator = self._build_intermediate_discriminator_model(discriminator)
        intermediate_vgg19 = self.build_intermediate_vgg19_model()
        intermediate_vggface = self.build_intermediate_vggface_model()
        discriminator.trainable = False
        intermediate_discriminator.trainable = False
        intermediate_vgg19.trainable = False
        intermediate_vggface.trainable = False

        input_lndmk = Input(shape=self.input_shape, name='landmarks')
        condition = Input(shape=(self.num_videos,), name='condition')
        inputs_embedder = [Input((self.h, self.w, self.c * 2), name='style{}'.format(i)) for i in range(self.k)]  # (BATCH_SIZE, H, W, 6)
        
        embeddings = [embedder(em_input) for em_input in inputs_embedder]  # (BATCH_SIZE, 512)
        
        embeddings_expand = [Lambda(lambda x: K.expand_dims(x, axis=1))(embedding) for embedding in embeddings]  # (BATCH_SIZE, 1, 512)
        embedding_k = Concatenate(axis=1)(embeddings_expand)  # (BATCH_SIZE, K, 512)
        average_embedding = GlobalAveragePooling1D()(embedding_k)  # (BATCH_SIZE, 512)
        fake_frame = generator([input_lndmk, average_embedding])

        intermediate_vgg19_fakes = intermediate_vgg19(fake_frame)
        intermediate_vggface_fakes = intermediate_vggface(fake_frame)
        intermediate_discriminator_fakes = intermediate_discriminator([fake_frame, input_lndmk])
        
        if meta:
            self._build_embedding_discriminator_model(discriminator)  # Call embedding discriminator when meta learning
            realicity = discriminator([fake_frame, input_lndmk, condition])
            combined = Model(
                inputs = [input_lndmk] + inputs_embedder + [condition],
                outputs = intermediate_vgg19_fakes + intermediate_vggface_fakes + [realicity] + intermediate_discriminator_fakes + embeddings,
                name = 'combined'
                )
            loss = ['mae'] * len(intermediate_vgg19_fakes) + ['mae'] * len(intermediate_vggface_fakes) + ['hinge'] + ['mae'] * len(intermediate_discriminator_fakes) + ['mae'] * self.k
            loss_weights = [1.5e-1] * len(intermediate_vgg19_fakes) + [2.5e-2] * len(intermediate_vggface_fakes) + [10] + [10] * len(intermediate_discriminator_fakes) + [10] * self.k
        else:
            realicity = discriminator([fake_frame, input_lndmk, average_embedding])
            combined = Model(
                inputs = [input_lndmk] + inputs_embedder,
                outputs = intermediate_vgg19_fakes + intermediate_vggface_fakes + [realicity] + intermediate_discriminator_fakes,
                name = 'combined'
            )
            loss = ['mae'] * len(intermediate_vgg19_fakes) + ['mae'] * len(intermediate_vggface_fakes) + ['hinge'] + ['mae'] * len(intermediate_discriminator_fakes)
            loss_weights = [1.5e-1] * len(intermediate_vgg19_fakes) + [2.5e-2] * len(intermediate_vggface_fakes) + [10] + [10] * len(intermediate_discriminator_fakes)

        self.embedder = embedder
        self.generator = generator
        self.combined = combined
        self.discriminator = discriminator

        if gpus > 1:
            parallel_combined = multi_gpu_model(combined, gpus=gpus)
            parallel_combined.compile(
                loss=loss,
                loss_weights=loss_weights,
                optimizer=Adam(lr=5e-5, beta_1=1e-5)
            )

            self.parallel_combined = parallel_combined
            self.parallel_discriminator = parallel_discriminator

            return parallel_combined, parallel_discriminator, combined, discriminator

        combined.compile(
            loss=loss,
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
