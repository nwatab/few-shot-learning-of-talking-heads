import os
import imageio
from keras import utils
import numpy as np


def flow_from_dir(path, num_video, batch_size=48, k=8):
    """
    frame: [BATCH_SIZE, H, W, C]
    landmark: [BATCH_SIZE, H, W, C]
    frames_embedding: [BATCH_SIZE, H, W, 8 * C]
    lndmks_embedding: [BATCH_SIZE, H, W, 8 * C]
    return frame, landmark, frames, landmarks, condition
    """
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
            if not len(files) == 9:
                continue

            path_to_frame_dir = cur
            path_to_lndmk_dir = cur.replace('frames', 'lndmks')
            frame_path = os.path.join(path_to_frame_dir, files[0])
            lndmk_path = os.path.join(path_to_lndmk_dir, files[0])
            frame.append(imageio.imread(frame_path))
            lndmk.append(imageio.imread(lndmk_path))
            frames_embedding_paths = [os.path.join(path_to_frame_dir, f) for f in files[1:]]
            lndmks_embedding_paths = [os.path.join(path_to_lndmk_dir, f) for f in files[1:]]
            frames_embedding_list = [imageio.imread(path) for path in frames_embedding_paths]
            lndmks_embedding_list = [imageio.imread(path) for path in lndmks_embedding_paths]
            frames_embedding_arr = np.concatenate(frames_embedding_list, axis=-1)
            lndmks_embedding_arr = np.concatenate(lndmks_embedding_list, axis=-1)
            frames_embedding.append(frames_embedding_arr)
            lndmks_embedding.append(lndmks_embedding_arr)
            condition.append(j)
            j += 1

            if len(frame) == batch_size:
                frame_temp = np.array(frame)
                lndmk_temp = np.array(lndmk)
                frames_embedding_temp = np.array(frames_embedding)
                lndmks_embedding_temp = np.array(lndmks_embedding)
                condition_temp = utils.to_categorical(condition, num_classes=num_video)
                frame = []
                lndmk = []
                frames_embedding = []
                lndmks_embedding = []
                condition = []
                yield frame_temp, lndmk_temp, frames_embedding_temp, lndmks_embedding_temp, condition_temp

                
if __name__ == '__main__':
    for f, l, fs, ls, c in flow_from_dir('../face-alignment/data/dataset/train/frames', num_video=1000):
        yield f, l, fs, ls, c
        
    
