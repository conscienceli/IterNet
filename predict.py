############Test


import os
import tensorflow as tf
from keras.backend import tensorflow_backend

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

from utils import define_model, prepare_dataset, crop_prediction
from utils.evaluate import evaluate
from keras.layers import ReLU
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
import cv2
from sklearn.metrics import roc_auc_score
import math
import pickle


def predict(ACTIVATION='ReLU', dropout=0.1, batch_size=32, repeat=4, minimum_kernel=32,
            epochs=200, iteration=3, crop_size=128, stride_size=3, DATASET='DRIVE'):
    prepare_dataset.prepareDataset(DATASET)
    test_data = [prepare_dataset.getTestData(0, DATASET),
                 prepare_dataset.getTestData(1, DATASET),
                 prepare_dataset.getTestData(2, DATASET)]

    IMAGE_SIZE = None
    if DATASET == 'DRIVE':
        IMAGE_SIZE = (565, 584)
    elif DATASET == 'CHASEDB1':
        IMAGE_SIZE = (999, 960)
    elif DATASET == 'STARE':
        IMAGE_SIZE = (700, 605)

    gt_list_out = {}
    pred_list_out = {}
    for out_id in range(iteration + 1):
        try:
            os.makedirs(f"./output/{DATASET}/crop_size_{crop_size}/out{out_id + 1}/", exist_ok=True)
            gt_list_out.update({f"out{out_id + 1}": []})
            pred_list_out.update({f"out{out_id + 1}": []})
        except:
            pass

    activation = globals()[ACTIVATION]
    model = define_model.get_unet(minimum_kernel=minimum_kernel, do=dropout, activation=activation, iteration=iteration)
    model_name = f"Final_Emer_Iteration_{iteration}_cropsize_{crop_size}_epochs_{epochs}"
    print("Model : %s" % model_name)
    load_path = f"trained_model/{DATASET}/{model_name}.hdf5"
    model.load_weights(load_path, by_name=False)

    imgs = test_data[0]
    segs = test_data[1]
    masks = test_data[2]

    for i in tqdm(range(len(imgs))):

        img = imgs[i]
        seg = segs[i]
        if masks:
            mask = masks[i]

        patches_pred, new_height, new_width, adjustImg = crop_prediction.get_test_patches(img, crop_size, stride_size)
        preds = model.predict(patches_pred)

        out_id = 0
        for pred in preds:
            pred_patches = crop_prediction.pred_to_patches(pred, crop_size, stride_size)
            pred_imgs = crop_prediction.recompone_overlap(pred_patches, crop_size, stride_size, new_height, new_width)
            pred_imgs = pred_imgs[:, 0:prepare_dataset.DESIRED_DATA_SHAPE[0], 0:prepare_dataset.DESIRED_DATA_SHAPE[0],
                        :]
            probResult = pred_imgs[0, :, :, 0]
            pred_ = probResult
            with open(f"./output/{DATASET}/crop_size_{crop_size}/out{out_id + 1}/{i + 1:02}.pickle", 'wb') as handle:
                pickle.dump(pred_, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pred_ = resize(pred_, IMAGE_SIZE[::-1])
            if masks:
                mask_ = mask
                mask_ = resize(mask_, IMAGE_SIZE[::-1])
            seg_ = seg
            seg_ = resize(seg_, IMAGE_SIZE[::-1])
            gt_ = (seg_ > 0.5).astype(int)
            gt_flat = []
            pred_flat = []
            for p in range(pred_.shape[0]):
                for q in range(pred_.shape[1]):
                    if not masks or mask_[p, q] > 0.5:  # Inside the mask pixels only
                        gt_flat.append(gt_[p, q])
                        pred_flat.append(pred_[p, q])

            gt_list_out[f"out{out_id + 1}"] += gt_flat
            pred_list_out[f"out{out_id + 1}"] += pred_flat

            pred_ = 255. * (pred_ - np.min(pred_)) / (np.max(pred_) - np.min(pred_))
            cv2.imwrite(f"./output/{DATASET}/crop_size_{crop_size}/out{out_id + 1}/{i + 1:02}.png", pred_)
            out_id += 1

    for out_id in range(iteration + 1)[-1:]:
        print('\n\n', f"out{out_id + 1}")
        evaluate(gt_list_out[f"out{out_id + 1}"], pred_list_out[f"out{out_id + 1}"], DATASET)


if __name__ == "__main__":
    predict(batch_size=32, epochs=200, iteration=3, stride_size=3, DATASET='DRIVE')
