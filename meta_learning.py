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
#from xla_multi_gpu_utils import multi_gpu_model

import logging
import os

LOG_FILE = 'logs/meta_learning.log'
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)
fmt = "%(asctime)s - %(levelname)s: %(message)s"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=fmt)

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
    BATCH_SIZE = 12
    num_videos = 145008  # This is dividable by BATCH_SIZE. All data is 145520
    num_batches = num_videos // BATCH_SIZE
    epochs = 75
    datapath = './datasets/voxceleb2-9f/train/lndmks'

    gan = GAN(input_shape=frame_shape, num_videos=num_videos, k=k)
    with tf.device("/cpu:0"):
        combined, discriminator = gan.build_models(meta=True)
    discriminator.trainable = True
    logging.info('==== discriminator ===')
    discriminator.summary(print_fn=logging.info)
    logging.info('=== generator ===')
    combined.get_layer('generator').summary(print_fn=logging.info)
    logging.info('=== embedder ===')
    combined.get_layer('embedder').summary(print_fn=logging.info)

    # 4 GPUs
    parallel_discriminator = multi_gpu_model(discriminator, gpus=4)
    parallel_discriminator.compile(loss=hinge_loss, optimizer=Adam(lr=2e-4, beta_1=0.0001))
    discriminator.trainable = False
    logging.info('=== combined ===')
    combined.summary(print_fn=logging.info)
    parallel_combined = multi_gpu_model(combined, gpus=4)
    parallel_combined.compile(
        loss=[
            perceptual_loss,
            hinge_loss,
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

    discriminator_fms = Model(discriminator.get_input_at(0),
                              [discriminator.get_layer('W_i').output,
                               discriminator.get_layer('fm1').output,
                               discriminator.get_layer('fm2').output,
                               discriminator.get_layer('fm3').output,
                               discriminator.get_layer('fm4').output,
                               discriminator.get_layer('fm5').output,
                               discriminator.get_layer('fm6').output,
                               discriminator.get_layer('fm7').output]
    )
    get_discriminator_fms = discriminator_fms.predict
    
    for epoch in range(epochs):
        logging.info(('Epoch: ', epoch))
        for batch_ix, (frames, landmarks, embedding_frames, embedding_landmarks, condition) in enumerate(flow_from_dir(datapath, num_videos, (h, w), BATCH_SIZE, k)):
            valid = np.ones((frames.shape[0], 1))
            invalid = - valid
            
            fake_frames, *_ = combined.predict_on_batch([landmarks, embedding_frames, embedding_landmarks, condition])
            w, d_fm1, d_fm2, d_fm3, d_fm4, d_fm5, d_fm6, d_fm7 = get_discriminator_fms([frames, landmarks, condition])
            g_loss = parallel_combined.train_on_batch(
                [landmarks, embedding_frames, embedding_landmarks, condition],
                [frames, valid, w, d_fm1, d_fm2, d_fm3, d_fm4, d_fm5, d_fm6, d_fm7]
            )
            d_loss_real = parallel_discriminator.train_on_batch(
                [frames, landmarks, condition],
                [valid]
            )
            d_loss_fake = parallel_discriminator.train_on_batch(
                [fake_frames, landmarks, condition],
                [invalid]
            )
            logging.info((epoch, batch_ix, g_loss, (d_loss_real, d_loss_fake)))

            if batch_ix % 1000 == 0:
                # Save whole model
                combined.save('trained_models/{}_meta_combined.h5'.format(epoch))
                discriminator.save('trained_models/{}_meta_discriminator.h5'.format(epoch))

                # Save weights only
                combined.save_weights('trained_models/{}_meta_combined_weights.h5'.format(epoch))
                combined.get_layer('generator').save_weights('trained_models/{}_meta_generator_in_combined.h5'.format(epoch))
                combined.get_layer('embedder').save_weights('trained_models/{}_meta_embedder_in_combined.h5'.format(epoch))
                discriminator.save_weights('trained_models/{}_meta_discriminator_weights.h5'.format(epoch))
        print()
    
if __name__ == '__main__':
    meta_learn()
