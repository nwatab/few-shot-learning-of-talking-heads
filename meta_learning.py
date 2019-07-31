from models import GAN
from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.engine import Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np
import tensorflow as tf

from data_loader import flow_from_dir


def hinge_loss(y_true, y_pred):
    """ y_true = 1 (True) or -1 (Fake) """
    return K.maximum(0., 1. - y_true * y_pred)

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


def get_batch(batch_ix, batch_size=48, k=8, num_classes=140000):
    """
    params
    batch_ix: j-th video. This is needed for condition
    nb_classes: the total number of videos
    k: get k frames from video. This is for embedder input.

    return 
    embedding_frames: (BATCH_SIZE, W, H, 3 * k) for embedder input
    embedding_landmarks: (BATCH_SIZE, W, H, 3 * k) for embedder input
    frames: (BATCH_SIZE, W, H, 3) for generator output and discriminator input
    landmarks: (BATCH_SIZE, W, H, 3) for generator input and discriminator input
    condition: (BATCG_SIZE, NUMBER_OF_CLASSES) for discriminator input

    """
    embedding_frames = np.random.rand(batch_size, 224, 224, 24)
    embedding_landmarks = embedding_frames ** 2
    frames = np.random.rand(batch_size, 224, 224, 3)
    landmarks = frames ** 2
    condition = np.repeat(np_utils.to_categorical(batch_ix, num_classes)[np.newaxis,...], batch_size, axis=0)  # not batch_ix
    
    return embedding_frames, embedding_landmarks, frames, landmarks, condition
    
def meta_learn():
    k = 8
    num_classes = 140000//1000
    frame_shape = h, w, c = (224, 224, 3)
    input_embedder_shape = (h, w, k * c)
    BATCH_SIZE = 48
    num_videos = 145469 // 1000
    num_batches = num_videos // BATCH_SIZE
    epochs = 75
    datapath = './dataset/voxceleb2-9f/'

    gan = GAN(input_shape=frame_shape)
    embedder = gan.build_embedder(k)
    generator = gan.build_generator()
    discriminator = gan.build_discriminator(num_classes)
    discriminator.compile(loss=hinge_loss, optimizer=Adam(lr=2e-4, beta_1=0.0001))

    # Generator + Embedder + Discriminator
    input_embedder_frames = Input(shape=input_embedder_shape)
    input_embedder_landmarks = Input(shape=input_embedder_shape)
    input_landmark = Input(shape=frame_shape)
    condition = Input(shape=(num_classes,))
    
    average_embedding, mean, stdev = embedder([input_embedder_frames, input_embedder_landmarks])
    fake_frame, g_fm1,  g_fm2, g_fm3, g_fm4, g_fm5, g_fm6, g_fm7 = generator([input_landmark, mean, stdev])
    discriminator.trainable = False
    get_discriminator_fms = K.function([*discriminator.input,
                                        K.learning_phase()],
                                       [discriminator.get_layer('fm1').output,
                                        discriminator.get_layer('fm2').output,
                                        discriminator.get_layer('fm3').output,
                                        discriminator.get_layer('fm4').output,
                                        discriminator.get_layer('fm5').output,
                                        discriminator.get_layer('fm6').output,
                                        discriminator.get_layer('fm7').output
                                        ]
                                       )
    get_discriminator_embedding = K.function([*discriminator.input, K.learning_phase()],
                                             [discriminator.get_layer('w').output])

    realicity = discriminator([fake_frame, input_landmark, condition])

    combined = Model(
        inputs=[input_landmark, input_embedder_frames, input_embedder_landmarks, condition],
        outputs=[fake_frame, realicity, average_embedding, g_fm1, g_fm2, g_fm3, g_fm4, g_fm5, g_fm6, g_fm7]
        )

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
            'mae',
            'mae'],
        loss_weights=[
            1e0,  # VGG19 and VGG Face loss is summed up in loss function
            1e0,  # hinge loss
            8e1,  # embedding match loss
            1e1,  # Feature matching 1-7 below
            1e1,
            1e1,
            1e1,
            1e1,
            1e1,
            1e1],
        optimizer=Adam(lr=5e-5, beta_1=0.001),
    )

    # function to get feature matching layers in discriminator
    valid = np.ones((BATCH_SIZE, 1))
    invalid = -1. *  np.ones((BATCH_SIZE, 1))
    
    for epoch in range(epochs):
        for enumerate batch_ix, (frames, landmarks, embedding_frames, embedding_landmarks, condition) in flow_from_dir(datapath, num_videos, BATCH_SIZE, k):
            d_fm1, d_fm2, d_fm3, d_fm4, d_fm5, d_fm6, d_fm7 = get_discriminator_fms([frames, landmarks, condition, 1])
            w = get_discriminator_embedding([frames, landmarks, condition, 1])[0]
            train_loss = combined.train_on_batch(
                [landmarks, embedding_frames, embedding_landmarks, condition],
                [frames, valid, w, d_fm1, d_fm2, d_fm3, d_fm4, d_fm5, d_fm6, d_fm7]
            )

            fake_frames, _, _, _, _, _, _, _, _ = combined.predict(landmarks)
            discriminator.train_on_batch(
                [frames, landmarks, condition],
                [valid]
            )
            discriminator.train_on_batch(
                [fake_frames, landmarks, condition],
                [invalid]
            )

        if epoch == 0:
            break
    
if __name__ == '__main__':
    meta_learn()
