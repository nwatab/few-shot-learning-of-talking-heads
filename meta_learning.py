import keras.backend as K
from keras.utils import np_utils, multi_gpu_model
import numpy as np
import tensorflow as tf

from data_loader import flow_from_dir
from models import GAN
#from xla_multi_gpu_utils import multi_gpu_model

import logging
import os


def meta_learn():
    k = 8
    frame_shape = h, w, c = (256, 256, 3)
    input_embedder_shape = (h, w, k * c)
    BATCH_SIZE = 12
    num_videos = 145008  # This is dividable by BATCH_SIZE. All data is 145520
    num_batches = num_videos // BATCH_SIZE
    epochs = 75
    datapath = '../few-shot-learning-of-talking-heads/datasets/voxceleb2-9f/train/lndmks'

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
        for batch_ix, (frames, landmarks, styles, condition) in enumerate(flow_from_dir(datapath, num_videos, (h, w), BATCH_SIZE, k)):
            if batch_ix == num_batches:
                break
            valid = np.ones((frames.shape[0], 1))
            invalid = - valid

            intermediate_vgg19_reals = intermediate_vgg19.predict_on_batch(frames)
            intermediate_vggface_reals = intermediate_vggface.predict_on_batch(frames)
            intermediate_discriminator_reals = intermediate_discriminator.predict_on_batch([frames, landmarks])

            style_list = [styles[:, i, :, :, :] for i in range(k)]

            w_i = embedding_discriminator.predict_on_batch(condition)

            g_loss = combined_to_train.train_on_batch(
                [landmarks] + style_list + [condition],
                intermediate_vgg19_reals + intermediate_vggface_reals + [valid] + intermediate_discriminator_reals + [w_i] * k
            )

            d_loss_real = discriminator_to_train.train_on_batch(
                [frames, landmarks, condition],
                [valid]
            )

            embeddings_list = [embedder.predict_on_batch(style) for style in style_list]
            average_embedding = np.mean(np.array(embeddings_list), axis=0)
            fake_frames = generator.predict_on_batch([landmarks, average_embedding])
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
#                combined.save_weights('trained_models/{}_meta_combined_weights.h5'.format(epoch))
                combined.get_layer('generator').save_weights('trained_models/{}_meta_generator_in_combined.h5'.format(epoch))
                combined.get_layer('embedder').save_weights('trained_models/{}_meta_embedder_in_combined.h5'.format(epoch))
                discriminator.save_weights('trained_models/{}_meta_discriminator_weights.h5'.format(epoch))
                logger.info('Checkpoint saved at Epoch: {}; batch_ix: {}'.format(epoch, batch_ix))
        print()
    
if __name__ == '__main__':
    LOG_FILE = 'logs/meta_learning2.log'
    fmt = "%(asctime)s:%(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=fmt)
    logger = logging.getLogger(__name__)
    meta_learn()
