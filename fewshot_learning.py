import logging
import os

from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import numpy as np
import tensorflow as tf

from models import GAN
from data_loader import flow_from_dir


def fewshot_learn():
    metalearning_epoch=0
    BATCH_SIZE = 1
    k = 1
    frame_shape = h, w, c = (256, 256, 3)
    input_embedder_shape = (h, w, k * c)
    BATCH_SIZE = 1
    num_videos = 1
    num_batches = 1
    epochs = 40
    dataname = 'monalisa'
    datapath = '../few-shot-learning-of-talking-heads/datasets/fewshot/' + dataname + '/lndmks'

    gan = GAN(input_shape=frame_shape, num_videos=num_videos, k=k)
    with tf.device("/cpu:0"):
        combined_to_train, combined, discriminator_to_train, discriminator = gan.compile_models(meta=False, gpus=0)
        embedder = gan.embedder
        generator = gan.generator
        intermediate_vgg19 = gan.intermediate_vgg19
        intermediate_vggface = gan.intermediate_vggface
        intermediate_discriminator = gan.intermediate_discriminator

    discriminator.load_weights('trained_models/{}_meta_discriminator_weights.h5'.format(metalearning_epoch), by_name=True, skip_mismatch=True)
    combined.get_layer('embedder').load_weights('trained_models/{}_meta_embedder_in_combined.h5'.format(metalearning_epoch))
    combined.get_layer('generator').load_weights('trained_models/{}_meta_generator_in_combined.h5'.format(metalearning_epoch))
    
    for epoch in range(epochs):
        for batch_ix, (frames, landmarks, styles) in enumerate(flow_from_dir(datapath, num_videos, (h, w), BATCH_SIZE, k, meta=False)):
            if batch_ix == num_batches:
                break
            valid = np.ones((frames.shape[0], 1))
            invalid = - valid

            intermediate_vgg19_outputs = intermediate_vgg19.predict_on_batch(frames)
            intermediate_vggface_outputs =intermediate_vggface.predict_on_batch(frames)
            intermediate_discriminator_outputs = intermediate_discriminator.predict_on_batch([frames, landmarks])

            style_list = [styles[:, i, :, :, :] for i in range(k)]
            embeddings_list = [embedder.predict_on_batch(style) for style in style_list]
            average_embedding = np.mean(np.array(embeddings_list), axis=0)
            fake_frames = generator.predict_on_batch([landmarks, average_embedding])
            
            g_loss = combined_to_train.train_on_batch(
                [landmarks] + style_list,
                intermediate_vgg19_outputs + intermediate_vggface_outputs + [valid] + intermediate_discriminator_outputs
            )

            embeddings_list = [embedder.predict_on_batch(style) for style in style_list]
            average_embedding = np.mean(np.array(embeddings_list), axis=0)

            d_loss_real = discriminator_to_train.train_on_batch(
                [frames, landmarks, average_embedding],
                [valid]
            )
            
            fake_frames = generator.predict_on_batch([landmarks, average_embedding])

            d_loss_fake = discriminator_to_train.train_on_batch(
                [fake_frames, landmarks, average_embedding],
                [invalid]
            )
            logger.info((epoch, batch_ix, g_loss, (d_loss_real, d_loss_fake)))

    # Save whole model
#    combined.save('trained_models/{}_fewshot_combined.h5'.format(dataname))
#    discriminator.save('trained_models/{}_fewshot_discriminator.h5'.format(dataname))

    # Save weights only
#    combined.save_weights('trained_models/{}_fewshot_combined_weights.h5'.format(dataname))
    combined.get_layer('generator').save_weights('trained_models/{}_fewshot_generator_in_combined.h5'.format(dataname))
#    combined.get_layer('embedder').save_weights('trained_models/{}_fewshot_embedder_in_combined.h5'.format(dataname))
    discriminator.save_weights('trained_models/{}_fewshot_discriminator_weights.h5'.format(dataname))
        

if __name__=='__main__':
    LOG_FILE = 'logs/fewshot_learning.log'
    fmt = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=fmt)
    logger = logging.getLogger(__name__)
    fewshot_learn()
