from keras.models import Model, load_model, model_from_json
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import numpy as np
import tensorflow as tf

from models import GAN
from utils import AdaIN, ConvSN2D, SelfAttention, DenseSN, Bias, GlobalSumPooling2D, Bias
from data_loader import flow_from_dir
from meta_learning import perceptual_loss, hinge_loss

import logging
import os


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
    datapath = './datasets/fewshot/' + dataname + '/lndmks'
    gan = GAN(input_shape=frame_shape, num_videos=num_videos, k=k)
    with tf.device("/cpu:0"):
        combined, discriminator = gan.build_models(meta=False)

    # Load meta-learned weights
    # discriminator in combined also initialized with trained weights
    discriminator.load_weights('trained_models/{}_meta_discriminator_weights.h5'.format(metalearning_epoch), by_name=True, skip_mismatch=True)
    combined.get_layer('embedder').load_weights('trained_models/{}_meta_embedder_in_combined.h5'.format(metalearning_epoch))
    combined.get_layer('generator').load_weights('trained_models/{}_meta_generator_in_combined.h5'.format(metalearning_epoch))

    discriminator_fms_model = Model(discriminator.get_input_at(0),
                              [discriminator.get_layer('fm1').output,
                               discriminator.get_layer('fm2').output,
                               discriminator.get_layer('fm3').output,
                               discriminator.get_layer('fm4').output,
                               discriminator.get_layer('fm5').output,
                               discriminator.get_layer('fm6').output,
                               discriminator.get_layer('fm7').output]
    )
    get_discriminator_fms = discriminator_fms_model.predict
    embedder_embedding_model = Model(combined.get_layer('embedder').get_input_at(0), combined.get_layer('embedder').get_layer('embedder_embedding').output)
    get_embedder_embedding = embedder_embedding_model.predict
    
    discriminator.compile(loss=hinge_loss, optimizer=Adam(lr=2e-4, beta_1=0.0001))
    discriminator.summary(print_fn=logger.info)
    discriminator.trainable = False
    combined.compile(
        loss=[
            perceptual_loss,
            hinge_loss,
            'mae',  # Embedding match loss
            'mae',  # Feature matching 1-7 below
            'mae',
            'mae',
            'mae',
            'mae',
            'mae'],
        loss_weights=[
            1e0,  # VGG19 and VGG Face loss is summed up in loss function
            1e0,  # hinge loss
            1e1,  # Feature matching 1-7 below
            1e1,
            1e1,
            1e1,
            1e1,
            1e1,
            1e1],
        optimizer=Adam(lr=5e-5, beta_1=0.001),
    )
    combined.summary(print_fn=logger.info)
    combined.get_layer('embedder').summary(print_fn=logger.info)
    combined.get_layer('generator').summary(print_fn=logger.info)
    
    for epoch in range(epochs):
        for batch_ix, (frames, landmarks, embedding_frames, embedding_landmarks) in enumerate(flow_from_dir(datapath, num_videos, (h, w), BATCH_SIZE, k, meta=False)):
            if batch_ix == num_batches:
                break
            valid = np.ones((frames.shape[0], 1))
            invalid = - valid
            fake_frames, *_ = combined.predict([landmarks, embedding_frames, embedding_landmarks])
            embedder_embedding =  get_embedder_embedding([embedding_frames, embedding_landmarks])
            d_fm1, d_fm2, d_fm3, d_fm4, d_fm5, d_fm6, d_fm7 = get_discriminator_fms([frames, landmarks, embedder_embedding])
            
            g_loss = combined.train_on_batch(
                [landmarks, embedding_frames, embedding_landmarks],
                [frames, valid, d_fm1, d_fm2, d_fm3, d_fm4, d_fm5, d_fm6, d_fm7]
            )
            d_loss_real = discriminator.train_on_batch(
                [frames, landmarks, embedder_embedding],
                [valid]
            )
            d_loss_fake = discriminator.train_on_batch(
                [fake_frames, landmarks, embedder_embedding],
                [invalid]
            )
        logger.info((epoch, g_loss, (d_loss_real, d_loss_fake)))

    # Save whole model
    combined.save('trained_models/{}_fewshot_combined.h5'.format(dataname))
    discriminator.save('trained_models/{}_fewshot_discriminator.h5'.format(dataname))

    # Save weights only
    combined.save_weights('trained_models/{}_fewshot_combined_weights.h5'.format(dataname))
    combined.get_layer('generator').save_weights('trained_models/{}_fewshot_generator_in_combined.h5'.format(dataname))
    combined.get_layer('embedder').save_weights('trained_models/{}_fewshot_embedder_in_combined.h5'.format(dataname))
    discriminator.save_weights('trained_models/{}_fewshot_discriminator_weights.h5'.format(dataname))
        

if __name__=='__main__':
    LOG_FILE = 'logs/fewshot_learning.log'
    fmt = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=fmt)
    logger = logging.getLogger(__name__)
    fewshot_learn()
