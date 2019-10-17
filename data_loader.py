import os
import imageio
import skimage
from keras import utils
import numpy as np


def flow_from_dir(path, num_video, output_shape=None, batch_size=48, k=8, meta=True):
    """
    params:
    output_shape: (H, W)
    return frame, landmark, frames, landmarks, condition
    frame: [BATCH_SIZE, H, W, C]
    landmark: [BATCH_SIZE, H, W, C]
    frames_embedding: [BATCH_SIZE, H, W, 8 * C]
    lndmks_embedding: [BATCH_SIZE, H, W, 8 * C]

    """
    if not os.path.exists(path):
        raise Exception(path, 'does not exist.')
        import sys
        sys.exist()
    while True:
        j = 0
        frame = []
        lndmk = []
        frames_embedding = []
        lndmks_embedding = []
        condition = []
        for cur, dirs, files in os.walk(path):
            if not files:
                continue
            if j == num_video:
                print('iteration ends')
                j = 0
                break
            path_to_lndmk_dir = cur
            path_to_frame_dir = cur.replace('lndmks', 'frames')
            frame_path = os.path.join(path_to_frame_dir, files[0])
            lndmk_path = os.path.join(path_to_lndmk_dir, files[0])
            frame_array = imageio.imread(frame_path)
            lndmk_array = imageio.imread(lndmk_path)
            if output_shape:
                frame_array = skimage.transform.resize(frame_array, output_shape)
                lndmk_array = skimage.transform.resize(lndmk_array, output_shape)
            frame.append(frame_array)
            lndmk.append(lndmk_array)
            frames_embedding_paths = [os.path.join(path_to_frame_dir, f) for f in files[1:]]
            lndmks_embedding_paths = [os.path.join(path_to_lndmk_dir, f) for f in files[1:]]
            if len(lndmks_embedding_paths) < 4:
                continue
            frames_embedding_list = [imageio.imread(path) for path in frames_embedding_paths]
            lndmks_embedding_list = [imageio.imread(path) for path in lndmks_embedding_paths]
            if output_shape:
                frames_embedding_list = [skimage.transform.resize(img, output_shape) for img in frames_embedding_list]
                lndmks_embedding_list = [skimage.transform.resize(img, output_shape) for img in lndmks_embedding_list]
            frames_embedding_arr = np.concatenate(frames_embedding_list, axis=-1)
            lndmks_embedding_arr = np.concatenate(lndmks_embedding_list, axis=-1)
            # augmente to 8 frames
            if frames_embedding_arr.shape[-1] < 8 * 3:
                frames_embedding_arr = np.concatenate((frames_embedding_arr, frames_embedding_arr[:, ::-1, :]), axis=-1)[:, :, :k * 3]
                lndmks_embedding_arr = np.concatenate((lndmks_embedding_arr, lndmks_embedding_arr[:, ::-1, :]), axis=-1)[:, :, :k * 3]
                
            frames_embedding.append(frames_embedding_arr)
            lndmks_embedding.append(lndmks_embedding_arr)
            if meta:
                condition.append(j)
                j += 1
            if len(frame) == batch_size:
                frame_temp = np.array(frame) / 127.5 - 1
                lndmk_temp = np.array(lndmk) / 127.5 - 1
                frames_embedding_temp = np.array(frames_embedding) / 127.5 - 1
                lndmks_embedding_temp = np.array(lndmks_embedding) / 127.5 - 1
                if meta:
                    condition_temp = utils.to_categorical(condition, num_classes=num_video)
                    condition = []
                frame = []
                lndmk = []
                frames_embedding = []
                lndmks_embedding = []
                if meta:
                    yield frame_temp, lndmk_temp, frames_embedding_temp, lndmks_embedding_temp, condition_temp
                else:
                    yield frame_temp, lndmk_temp, frames_embedding_temp, lndmks_embedding_temp
                
if __name__ == '__main__':
    path = './datasets/voxceleb2-9f/train/lndmks/'
    for f, l, fs, ls, c in flow_from_dir(path, num_video=145000):
        pass

        
    
