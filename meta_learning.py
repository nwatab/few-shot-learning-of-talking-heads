import keras.backend as K
from keras.utils import np_utils, multi_gpu_model
import numpy as np
import tensorflow as tf

from data_loader import flow_from_dir
from models import GAN
#from xla_multi_gpu_utils import multi_gpu_model

import logging
import os


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
        combined_to_train, combined, discriminator_to_train, discriminator = gan.compile_models(meta=True, gpus=0)
        embedder = gan.embedder
        generator = gan.generator
        intermediate_vgg19 = gan.intermediate_vgg19
        intermediate_vggface = gan.intermediate_vggface
        intermediate_discriminator = gan.intermediate_discriminator
        embedding_discriminator = gan.embedding_discriminator

    logger.info('==== discriminator ===')
    discriminator.summary(print_fn=logger.info)
    logger.info('=== generator ===')
    combined.get_layer('generator').summary(print_fn=logger.info)
    logger.info('=== embedder ===')
    combined.get_layer('embedder').summary(print_fn=logger.info)
    combined.summary(print_fn=logger.info)

    for epoch in range(epochs):
        logger.info(('Epoch: ', epoch))
        for batch_ix, (frames, landmarks, style, condition) in enumerate(flow_from_dir(datapath, num_videos, (h, w), BATCH_SIZE, k)):
            if batch_ix == num_batches:
                break
            valid = np.ones((frames.shape[0], 1))
            invalid = - valid

            intermediate_vgg19_reals = intermediate_vgg19.predict_on_batch(frames)
            intermediate_vggface_reals = intermediate_vggface.predict_on_batch(frames)
            intermediate_discriminator_reals = intermediate_discriminator.predict_on_batch([frames, landmarks])

            style_list = [style[:, i, :, :, :] for i in range(k)]
            embeddings_list = [embedder.predict_on_batch(style) for style in style_list]
            average_embedding = np.mean(np.array(embeddings_list), axis=0)
#            e_hat = embedder.predict_on_batch([embedding_frames, embedding_landmarks])
            w_i = embedding_discriminator.predict_on_batch(condition)
            fake_frames = generator.predict_on_batch([landmarks, average_embedding])
            g_loss = combined_to_train.train_on_batch(
                [landmarks] + style_list + [condition],
                intermediate_vgg19_reals + intermediate_vggface_reals + [valid] + intermediate_discriminator_reals + [w_i] * k
            )

            d_loss_real = discriminator_to_train.train_on_batch(
                [frames, landmarks, condition],
                [valid]
            )

            d_loss_fake = discriminator_to_train.train_on_batch(
                [fake_frames, landmarks, condition],
                [invalid]
            )
            logger.info((epoch, batch_ix, g_loss, (d_loss_real, d_loss_fake)))

            if batch_ix % 100 == 0 and batch_ix > 0:
                # Save whole model
                # combined.save('trained_models/{}_meta_combined.h5'.format(epoch))
                # discriminator.save('trained_models/{}_meta_discriminator.h5'.format(epoch))

                # Save weights only
                combined.save_weights('trained_models/{}_meta_combined_weights.h5'.format(epoch))
                combined.get_layer('generator').save_weights('trained_models/{}_meta_generator_in_combined.h5'.format(epoch))
                combined.get_layer('embedder').save_weights('trained_models/{}_meta_embedder_in_combined.h5'.format(epoch))
                discriminator.save_weights('trained_models/{}_meta_discriminator_weights.h5'.format(epoch))
                logger.info('Checkpoint saved at Epoch: {}; batch_ix: {}'.format(epoch, batch_ix))
        print()
    
if __name__ == '__main__':
    LOG_FILE = 'logs/meta_learning.log'
    fmt = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=fmt)
    logger = logging.getLogger(__name__)
    meta_learn()
