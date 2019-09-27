import keras
from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.engine import Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras.optimizers import Adam
from keras.utils import np_utils, multi_gpu_model
import numpy as np
import tensorflow as tf

from data_loader import flow_from_dir
from models import GAN


def hinge_loss(y_true, y_pred):
    """ y_true = 1 (True) or -1 (Fake) """
    return tf.math.maximum(0., 1. - y_true * y_pred)

def _vgg19_perceptual_loss(y_true, y_pred):
    vgg19 = VGG19(input_shape=y_pred.get_shape().as_list()[1:], weights='imagenet', include_top=False)
    vgg19.trainable = False
    # Paper says Conv1, 6, 11, 20, 29 VGG19 layers but it isn't clear which layer is which layer
    layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4']
    prcp_layers = [vgg19.get_layer(layer_name).output for layer_name in layer_names]

    prcp_model = Model(vgg19.input, prcp_layers)
    prcp_trues = prcp_model(y_true)
    prcp_preds = prcp_model(y_pred)

    l1_losses = [K.mean(K.abs(prcp_true - prcp_pred), axis=[1,2,3])
                 for prcp_true, prcp_pred in zip(prcp_trues, prcp_preds)]
    loss = K.mean(tf.convert_to_tensor(l1_losses))
    return loss

def _vggface_perceptual_loss(y_true, y_pred):
    vggface = VGGFace(input_shape=y_pred.get_shape().as_list()[1:], weights='vggface', include_top=False)
    vggface.trainable = False

    layer_names = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3']
    prcp_layers = [vggface.get_layer(layer_name).output for layer_name in layer_names]
    prcp_model = Model(vggface.input, prcp_layers)

    prcp_trues = prcp_model(y_true)
    prcp_preds = prcp_model(y_pred)

    l1_losses = [K.mean(K.abs(prcp_true - prcp_pred), axis=[1,2,3])
                 for prcp_true, prcp_pred in zip(prcp_trues, prcp_preds)]
    loss = K.mean(tf.convert_to_tensor(l1_losses))
    return loss

def perceptual_loss(y_true, y_pred):
    vgg19_loss = _vgg19_perceptual_loss(y_true, y_pred)
    vggface_loss = _vggface_perceptual_loss(y_true, y_pred)
    sum_loss =  1e-2 * vgg19_loss + 2e-3 * vggface_loss
    return sum_loss

def meta_learn():
    k = 8
    frame_shape = h, w, c = (256, 256, 3)
    input_embedder_shape = (h, w, k * c)
    BATCH_SIZE = 48
    num_videos = 145469
    num_batches = num_videos // BATCH_SIZE
    epochs = 75
    datapath = './datasets/voxceleb2-9f/'

    gan = GAN(input_shape=frame_shape, num_videos=num_videos, k=k)
    combined, discriminator = gan.build_models()
    parallel_discriminator = multi_gpu_model(discriminator, gpus=2)
    parallel_discriminator.compile(loss=hinge_loss, optimizer=Adam(lr=2e-4, beta_1=0.0001))
    discriminator.trainable = False
    parallel_combined = multi_gpu_model(combined, gpus=2)
    parallel_combined.compile(
        loss=[
            perceptual_loss,
            'hinge',
            'mae',  # Embedding match loss
            'mae',  # Feature matching 1-7 below
            'mae',
            'mae',
            'mae',
            'mae',
            'mae',
            'mae'],
        loss_weights=[
            1e0,  # VGG19 and VGG Face loss is summed up in loss function
            1e0,  # hinge loss
            8e1,  # Embedding match loss
            1e1,  # Feature matching 1-7 below
            1e1,
            1e1,
            1e1,
            1e1,
            1e1,
            1e1],
        optimizer=Adam(lr=5e-5, beta_1=0.001),
    )
    discriminator.summary()
    combined.get_layer('generator').summary()
    combined.get_layer('embedder').summary()
    combined.summary()

    discriminator_fms = Model(discriminator.get_input_at(0),
                              [discriminator.get_layer('fm1').output,
                               discriminator.get_layer('fm2').output,
                               discriminator.get_layer('fm3').output,
                               discriminator.get_layer('fm4').output,
                               discriminator.get_layer('fm5').output,
                               discriminator.get_layer('fm6').output,
                               discriminator.get_layer('fm7').output]
    )
    get_discriminator_fms = discriminator_fms.predict
    discriminator_embedding = Model(discriminator.get_input_at(0), discriminator.get_layer('W_i').output)
    get_discriminator_embedding = discriminator_embedding.predict

    valid = np.ones((BATCH_SIZE, 1))
#    invalid = -1. *  np.ones((BATCH_SIZE, 1))
    invalid = np.zeros((BATCH_SIZE, 1))
    
    for epoch in range(epochs):
        print('Epoch: ', epoch)
        for batch_ix, (frames, landmarks, embedding_frames, embedding_landmarks, condition) in enumerate(flow_from_dir(datapath, num_videos, (h, w), BATCH_SIZE, k)):
            fake_frames, *_ = combined.predict([landmarks, embedding_frames, embedding_landmarks, condition])
            d_fm1, d_fm2, d_fm3, d_fm4, d_fm5, d_fm6, d_fm7 = get_discriminator_fms([frames, landmarks, condition])
            w = get_discriminator_embedding([frames, landmarks, condition])
            g_loss = parallel_combined.train_on_batch(
                [landmarks, embedding_frames, embedding_landmarks, condition],
                [frames, valid, w, d_fm1, d_fm2, d_fm3, d_fm4, d_fm5, d_fm6, d_fm7]
            )
            fake_frames, *_ = combined.predict([landmarks, embedding_frames, embedding_landmarks, condition])
            d_loss_real = parallel_discriminator.train_on_batch(
                [frames, landmarks, condition],
                [valid]
            )
            d_loss_fake = parallel_discriminator.train_on_batch(
                [fake_frames, landmarks, condition],
                [invalid]
            )
            print(g_loss, (d_loss_real + d_loss_fake) / 2)
        print()


    combined.save('trained_models/meta_combined.h5')
    combined.save_weights('trained_models/meta_combined_weights.h5')
    combined.get_layer('generator').save_weights('trained_models/meta_generator_in_combined.h5')
    combined.get_layer('embedder').save_weights('trained_models/meta_embedder_in_combined.h5')
    discriminator.save('trained_models/meta_discriminator.h5')
    discriminator.save_weights('trained_models/meta_discriminator_weights.h5')
    
if __name__ == '__main__':
    meta_learn()
