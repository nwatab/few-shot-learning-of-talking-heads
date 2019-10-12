from keras.models import Model, load_model, model_from_json
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import numpy as np

from models import GAN
from utils import AdaIN, ConvSN2D, SelfAttention, DenseSN, Bias, GlobalSumPooling2D, Bias
from data_loader import flow_from_dir
from meta_learning import perceptual_loss


def fewshot_learn():
    datapath = './datasets/fewshot/'
    BATCH_SIZE = 1
    k = 8
    frame_shape = h, w, c = (256, 256, 3)
    input_embedder_shape = (h, w, k * c)
    BATCH_SIZE = 1
    num_videos = 1
    num_batches = 1
    epochs = 40
    datapath = './datasets/fewshot/monalisa/'
    gan = GAN(input_shape=frame_shape, num_videos=1, k=8)
    with tf.device("/cpu:0"):
        combined, discriminator = gan.build_models(meta=False)
    discriminator.load_weights('trained_models/meta_discriminator_weights.h5', by_name=True, skip_mismatch=True)
    combined.get_layer('embedder').load_weights('trained_models/meta_embedder_in_combined.h5')
    combined.get_layer('generator').load_weights('trained_models/meta_generator_in_combined.h5')
    discriminator.summary()
    combined.get_layer('embedder').summary()
    combined.get_layer('generator').summary()
    combined.summary()

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
    parallel_discriminator = multi_gpu_model(discriminator, gpus=4)
    parallel_combined = multi_gpu_model(combined, gpus=4)
    parallel_discriminator = discriminator
    parallel_discriminator.compile(loss='hinge', optimizer=Adam(lr=2e-4, beta_1=0.0001))
    discriminator.trainable = False
    parallel_combined = combined
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

    valid = np.ones((BATCH_SIZE, 1))
    invalid = np.zeros((BATCH_SIZE, 1))
    
    for epoch in range(epochs):
        print('Epoch: ', epoch)
        for batch_ix, (frames, landmarks, embedding_frames, embedding_landmarks) in enumerate(flow_from_dir(datapath, num_videos, (h, w), BATCH_SIZE, k, meta=False)):
            fake_frames, *_ = combined.predict([landmarks, embedding_frames, embedding_landmarks])
            embedder_embedding =  get_embedder_embedding([embedding_frames, embedding_landmarks])
            d_fm1, d_fm2, d_fm3, d_fm4, d_fm5, d_fm6, d_fm7 = get_discriminator_fms([frames, landmarks, embedder_embedding])
            
            g_loss = parallel_combined.train_on_batch(
                [landmarks, embedding_frames, embedding_landmarks],
                [frames, valid, d_fm1, d_fm2, d_fm3, d_fm4, d_fm5, d_fm6, d_fm7]
            )
            d_loss_real = parallel_discriminator.train_on_batch(
                [frames, landmarks, embedder_embedding],
                [valid]
            )
            d_loss_fake = parallel_discriminator.train_on_batch(
                [fake_frames, landmarks, embedder_embedding],
                [invalid]
            )
            print(g_loss, (d_loss_real + d_loss_fake) / 2)

    # Save whole model
    combined.save('trained_models/monalisa_combined.h5')
    discriminator.save('trained_models/monalisa_discriminator.h5')

    # Save weights only
    combined.save_weights('trained_models/monalisa_combined_weights.h5')
    combined.get_layer('generator').save_weights('trained_models/monalisa_generator_in_combined.h5')
    combined.get_layer('embedder').save_weights('trained_models/monalisa_embedder_in_combined.h5')
    discriminator.save_weights('trained_models/monalisa_discriminator_weights.h5')

        

if __name__=='__main__':
    fewshot_learn()
