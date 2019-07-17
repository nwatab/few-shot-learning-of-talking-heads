import keras.backend as K
from keras.layers import Add, Input, LeakyReLU, AveragePooling2D, GlobalAveragePooling1D, Dot, Concatenate, Lambda, UpSampling2D, Lambda, ZeroPadding2D, Cropping2D
from keras.layers.core import Activation
from keras.models import Model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import tensorflow as tf
from utils import GlobalSumPooling2D, ConvSN2D, DenseSN, AdaIN, Bias, SelfAttention


class GAN:
    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape

    def downsample(self, x, channels, instance_normalization=False, act_name=None):
        """  Downsampling is similar to an implementation in BigGAN """
        shortcut = x
 
        x = LeakyReLU(alpha=0.2)(x)
        x = ConvSN2D(channels, (3,3), strides=(1, 1), padding='same')(x)
        if instance_normalization:
            x = InstanceNormalization(axis=-1)(x)  # might be unnecessary
        x = LeakyReLU(alpha=0.2, name=act_name)(x)
        act = x
        x = ConvSN2D(channels, (3,3), strides=(1, 1), padding='same')(x)
        if instance_normalization:
            x = InstanceNormalization(axis=-1)(x)  # might be unnecessary
        x = AveragePooling2D(pool_size=(2, 2))(x)

        shortcut = ConvSN2D(channels, (1, 1), padding='same')(shortcut)
        shortcut = AveragePooling2D(pool_size=(2, 2))(shortcut)

        x = Add()([x, shortcut])
        if act_name:
            return x, act
        return x

    def resblock(self, x, channels):
        shortcut = x

        x = ConvSN2D(channels, (3, 3), padding='same')(x)
        x = InstanceNormalization(axis=-1)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = ConvSN2D(channels, (3, 3), padding='same')(x)
        x = InstanceNormalization(axis=-1)(x)

        if shortcut.shape[-1] != channels:
            shortcut = ConvSN2D(channels, (1, 1), padding='same')(shortcut)

        x = Add()([x, shortcut])
        return x

    def upsample(self, x, channels, mean, var, instance_normalization=False):
        shortcut = x

        if instance_normalization:
            x = InstanceNormalization(axis=-1)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = ConvSN2D(channels, (3, 3), padding='same')(x)
        if instance_normalization:
            x = InstanceNormalization(axis=-1)(x)
        x = AdaIN()([x, mean, var])
        x = LeakyReLU(alpha=0.2)(x)
        x = ConvSN2D(channels, (3, 3), padding='same')(x)

        shortcut = UpSampling2D(size=(2, 2))(shortcut)
        shortcut = ConvSN2D(channels, (1, 1))(shortcut)

        x = Add()([x, shortcut])
        return x

    def embed(self):
        input_frame = Input(shape=self.input_shape)
        input_landmark = Input(shape=self.input_shape)
        inputs = Concatenate()([input_frame, input_landmark])
        hid = self.downsample(inputs, 64)
        hid = self.downsample(hid, 128)
        hid = self.downsample(hid, 256)
        hid = SelfAttention(512)(hid)
        hid = self.downsample(hid, 512)
        hid = self.downsample(hid, 512)
        hid = self.downsample(hid, 512)
        hid = Activation('relu')(hid)
        e = GlobalSumPooling2D()(hid)

        embedder = Model(inputs=[input_frame, input_landmark], outputs=e)
        return embedder

    def build_embedder(self, k):
        """ k frames from the same sequence """
        h, w, c = self.input_shape
        input_frames = Input(shape=(h, w, c * k))
        input_landmarks = Input(shape=(h, w, c * k))

        input_frames_splt = [Lambda(lambda x: x[:, :, :, 3*i:3*(i+1)])(input_frames) for i in range(k)]
        input_landmarks_splt = [Lambda(lambda x: x[:,:, :, 3*i:3*(i+1)])(input_landmarks) for i in range(k)]

        embedder = self.embed()
        embedding_vectors = [embedder([frame, landmark]) for frame, landmark in zip(input_frames_splt, input_landmarks_splt)] # List of (BATCH_SIZE, 512,)
        embedding_vectors = [Lambda(lambda x: K.expand_dims(x, axis=1))(vector) for vector in embedding_vectors]  # List of (BATCH_SIZE, 1, 512)
        embeddings = Concatenate(axis=1)(embedding_vectors)  # (BATCH_SIZE, k, 512)
        average_embedding = GlobalAveragePooling1D(name='average_embedding')(embeddings)
        
        hid = DenseSN(512)(average_embedding)
        hid = DenseSN(512)(hid)
        mean = DenseSN(1, name='mean')(hid)
        stdev = DenseSN(1, name='stdev')(hid)

        embedder = Model(inputs=[input_frames, input_landmarks], outputs=[average_embedding, mean, stdev])
        return embedder

    def build_generator(self):
        inputs = Input(shape=self.input_shape, name='landmarks')
        mean = Input(shape=(1,), name='mean')
        var = Input(shape=(1,), name='var')

        hid, fm7 = self.downsample(inputs, 64, instance_normalization=True, act_name='fm7')
        hid, fm6 = self.downsample(hid, 128, instance_normalization=True, act_name='fm6')
        hid, fm5 = self.downsample(hid, 256, instance_normalization=True, act_name='fm5')
        hid = SelfAttention(256)(hid)
        hid, fm4 = self.downsample(hid, 256, instance_normalization=True, act_name='fm4')
        hid, fm3 = self.downsample(hid, 256, instance_normalization=True, act_name='fm3')
        hid, fm2 = self.downsample(hid, 256, instance_normalization=True, act_name='fm2')
        hid, fm1 = self.downsample(hid, 512, instance_normalization=True, act_name='fm1')
        hid = ZeroPadding2D(padding=((0, 1),(0, 1)))(hid)  # For input size is 224p
        
        hid = self.resblock(hid, 512)
        hid = self.resblock(hid, 512)
        
        hid = self.upsample(hid, 256, mean, var, instance_normalization=True)
        hid = self.upsample(hid, 256, mean, var, instance_normalization=True)
        hid = Cropping2D(cropping=((0,1), (0,1)))(hid)  # For input size is 224p
        hid = self.upsample(hid, 256, mean, var, instance_normalization=True)
        hid = self.upsample(hid, 256, mean, var, instance_normalization=True)
        hid = self.upsample(hid, 128, mean, var, instance_normalization=True)
        hid = SelfAttention(256)(hid)
        hid = self.upsample(hid, 64, mean, var, instance_normalization=True)
        hid = self.upsample(hid, 64,  mean, var, instance_normalization=True)
        hid = ConvSN2D(3, (1, 1), padding='same')(hid)
        fake_frame = Activation('tanh', name='fake_frame')(hid)

        generator = Model(
            inputs=[inputs, mean, var],
            outputs=[fake_frame, fm1, fm2, fm3, fm4, fm5, fm6, fm7])
        return generator
    
    def build_discriminator(self, num_classes):
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

        condition = Input(shape=(num_classes,), name='condition')
        W_i = DenseSN(512, use_bias=False, name='W_i')(condition)  # Projection Matrix, P
        w = Bias(name='w')(W_i)  # w = W_i + w_0
        

        innerproduct = Dot(axes=-1)([v, w])
        realicity = Bias(name='realicity')(innerproduct)
        
        discriminator = Model(
            inputs=[input_frame, input_landmark, condition],
            outputs=[realicity]
        )
        return discriminator
