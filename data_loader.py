import os
import imageio
import skimage
from keras import utils
import numpy as np


def flow_from_dir(landmark_path, num_video, output_shape=None, batch_size=48, k=8, meta=True):
    """
    params:
    output_shape: (H, W)
    return frame, landmark, frames, landmarks, condition
    frame: [BATCH_SIZE, H, W, C]
    landmark: [BATCH_SIZE, H, W, C]
    frames_embedding: [BATCH_SIZE, H, W, 8 * C]
    lndmks_embedding: [BATCH_SIZE, H, W, 8 * C]

    """
    if not os.path.exists(landmark_path):
        raise Exception(landmark_path, 'does not exist.')
        import sys
        sys.exist()
    while True:
        j = 0
        frame = []
        lndmk = []
        styles = []
        condition = []
        for cur, dirs, files in os.walk(landmark_path):
            if not files:
                continue
            if j == num_video:
                print('End of iteration')
                j = 0
                break
            if '.DS_Store' in files:
                files.remove('.DS_Store')
            if len(files[1:]) < k // 2:
                # Lacking embedding input. if k // 2 <=  len(files[1:]) < k, then augment later
                continue
            path_to_lndmk_dir = cur
            path_to_frame_dir = cur.replace('lndmks', 'frames')
            frame_path = os.path.join(path_to_frame_dir, files[0])
            lndmk_path = os.path.join(path_to_lndmk_dir, files[0])

            frames_embedding_paths = [os.path.join(path_to_frame_dir, f) for f in files[1:]]
            lndmks_embedding_paths = [os.path.join(path_to_lndmk_dir, f) for f in files[1:]]

            frame_array = imageio.imread(frame_path)
            lndmk_array = imageio.imread(lndmk_path)
            if output_shape:
                frame_array = skimage.transform.resize(frame_array, output_shape, preserve_range=True)
                lndmk_array = skimage.transform.resize(lndmk_array, output_shape, preserve_range=True)
            frame.append(frame_array)
            lndmk.append(lndmk_array)
            frames_embedding_list = [imageio.imread(path) for path in frames_embedding_paths]
            lndmks_embedding_list = [imageio.imread(path) for path in lndmks_embedding_paths]
            if output_shape:
                frames_embedding_list = [skimage.transform.resize(img, output_shape, preserve_range=True) for img in frames_embedding_list]
                lndmks_embedding_list = [skimage.transform.resize(img, output_shape, preserve_range=True) for img in lndmks_embedding_list]
            frames_embedding_arr = np.array(frames_embedding_list)
            lndmks_embedding_arr = np.array(lndmks_embedding_list)
            if frames_embedding_arr.shape[0] < k :
                # augmente to k frames if embedding input is not enough
                frames_embedding_arr = np.concatenate((frames_embedding_arr, frames_embedding_arr[:, :, ::-1, :]), axis=0)
                lndmks_embedding_arr = np.concatenate((lndmks_embedding_arr, lndmks_embedding_arr[:, :, ::-1, :]), axis=0)
            frames_embedding_arr = frames_embedding_arr[:k, :, :, :]
            lndmks_embedding_arr = lndmks_embedding_arr[:k, :, :, :]
            style = np.concatenate((lndmks_embedding_arr, frames_embedding_arr),axis=-1)  # (k, H, W, 6)
            styles.append(style)
            if meta:
                condition.append(j)
                j += 1
            if len(frame) == batch_size:
                frame_temp = np.array(frame) / 127.5 - 1
                lndmk_temp = np.array(lndmk) / 127.5 - 1
                styles_temp = np.array(styles) / 127.5 - 1  # (BATCH_SIZE, k, H, W, 6)
                if meta:
#                    condition_temp = np.eye(num_video)[condition]  Causes memory error
                    condition_temp = utils.to_categorical(condition, num_classes=num_video)
                    condition = []
                frame = []
                lndmk = []
                styles = []
                if meta:
                    yield frame_temp, lndmk_temp, styles_temp, condition_temp
                else:
                    yield frame_temp, lndmk_temp, styles_temp
                
if __name__ == '__main__':
    path = './datasets/voxceleb2-9f/train/lndmks/'
#    path = './datasets/fewshot/monalisa/lndmks'
    for batch_ix, (f, l, s, c) in enumerate(flow_from_dir(path, num_video=145000, batch_size=12, k=8, meta=True)):
        print(batch_ix, f.shape, s.shape)
        print(f.min(), f.max(), l.min(), l.max(), s.min(), s.max())
