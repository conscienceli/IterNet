import h5py
import numpy as np
import os.path
from PIL import Image
from glob import glob
from skimage.transform import resize

raw_training_x_path_DRIVE = './data/DRIVE/training/images/*.tif'
raw_training_y_path_DRIVE = './data/DRIVE/training/1st_manual/*.gif'
raw_test_x_path_DRIVE = './data/DRIVE/test/images/*.tif'
raw_test_y_path_DRIVE = './data/DRIVE/test/1st_manual/*.gif'
raw_test_mask_path_DRIVE = './data/DRIVE/test/mask/*.gif'

raw_training_x_path_CHASEDB1 = './data/CHASEDB1/training/images/*.jpg'
raw_training_y_path_CHASEDB1 = './data/CHASEDB1/training/1st_manual/*1stHO.png'
raw_test_x_path_CHASEDB1 = './data/CHASEDB1/test/images/*.jpg'
raw_test_y_path_CHASEDB1 = './data/CHASEDB1/test/1st_manual/*1stHO.png'
raw_test_mask_path_CHASEDB1 = './data/CHASEDB1/test/mask/*mask.png'

raw_training_x_path_STARE = './data/STARE/training/stare-images/*.ppm'
raw_training_y_path_STARE = './data/STARE/training/labels-ah/*.ppm'
raw_test_x_path_STARE = './data/STARE/test/stare-images/*.ppm'
raw_test_y_path_STARE = './data/STARE/test/labels-ah/*.ppm'
raw_test_mask_path_STARE = './data/STARE/test/mask/*mask.png'

raw_data_path = None
raw_data_path_DRIVE = [raw_training_x_path_DRIVE, raw_training_y_path_DRIVE, raw_test_x_path_DRIVE,
                       raw_test_y_path_DRIVE, raw_test_mask_path_DRIVE]
raw_data_path_CHASEDB1 = [raw_training_x_path_CHASEDB1, raw_training_y_path_CHASEDB1, raw_test_x_path_CHASEDB1,
                          raw_test_y_path_CHASEDB1, raw_test_mask_path_CHASEDB1]
raw_data_path_STARE = [raw_training_x_path_STARE, raw_training_y_path_STARE, raw_test_x_path_STARE,
                       raw_test_y_path_STARE, raw_test_mask_path_STARE]

HDF5_data_path = './data/HDF5/'

DESIRED_DATA_SHAPE_DRIVE = (576, 576)
DESIRED_DATA_SHAPE_CHASEDB1 = (960, 960)
DESIRED_DATA_SHAPE_STARE = (592, 592)
DESIRED_DATA_SHAPE = None


def isHDF5exists(raw_data_path, HDF5_data_path):
    for raw in raw_data_path:
        if not raw:
            continue

        raw_splited = raw.split('/')
        HDF5 = ''.join([HDF5_data_path, '/'.join(raw_splited[2:-1]), '/*.hdf5'])

        if len(glob(HDF5)) == 0:
            return False

    return True


def read_input(path):
    if path.find('mask') > 0 and (path.find('CHASEDB1') > 0 or path.find('STARE') > 0):
        fn = lambda x: 1.0 if x > 0.5 else 0
        x = np.array(Image.open(path).convert('L').point(fn, mode='1')) / 1.
    elif path.find('2nd') > 0 and path.find('DRIVE') > 0:
        x = np.array(Image.open(path)) / 1.
    elif path.find('_manual') > 0 and path.find('CHASEDB1') > 0:
        x = np.array(Image.open(path)) / 1.
    else:
        x = np.array(Image.open(path)) / 255.
    if x.shape[-1] == 3:
        return x
    else:
        return x[..., np.newaxis]


def preprocessData(data_path, dataset):
    global DESIRED_DATA_SHAPE

    data_path = list(sorted(glob(data_path)))

    if data_path[0].find('mask') > 0:
        return np.array([read_input(image_path) for image_path in data_path])
    else:
        return np.array([resize(read_input(image_path), DESIRED_DATA_SHAPE) for image_path in data_path])


def createHDF5(data, HDF5_data_path):
    try:
        os.makedirs(HDF5_data_path, exist_ok=True)
    except:
        pass
    f = h5py.File(HDF5_data_path + 'data.hdf5', 'w')
    f.create_dataset('data', data=data)
    return


def prepareDataset(dataset):
    global raw_data_path, HDF5_data_path, raw_data_path_DRIVE, raw_data_path_CHASEDB1, raw_data_path_STARE

    global DESIRED_DATA_SHAPE
    if dataset == 'DRIVE':
        DESIRED_DATA_SHAPE = DESIRED_DATA_SHAPE_DRIVE
    elif dataset == 'CHASEDB1':
        DESIRED_DATA_SHAPE = DESIRED_DATA_SHAPE_CHASEDB1
    elif dataset == 'STARE':
        DESIRED_DATA_SHAPE = DESIRED_DATA_SHAPE_STARE

    if dataset == 'DRIVE':
        raw_data_path = raw_data_path_DRIVE
    elif dataset == 'CHASEDB1':
        raw_data_path = raw_data_path_CHASEDB1
    elif dataset == 'STARE':
        raw_data_path = raw_data_path_STARE

    if isHDF5exists(raw_data_path, HDF5_data_path):
        return

    for raw in raw_data_path:
        if not raw:
            continue

        raw_splited = raw.split('/')
        HDF5 = ''.join([HDF5_data_path, '/'.join(raw_splited[2:-1]), '/'])

        preprocessed = preprocessData(raw, dataset)
        createHDF5(preprocessed, HDF5)


def getTrainingData(XorY, dataset):
    global HDF5_data_path, raw_data_path_DRIVE, raw_data_path_CHASEDB1, raw_data_path_STARE

    if dataset == 'DRIVE':
        raw_training_x_path, raw_training_y_path = raw_data_path_DRIVE[:2]
    elif dataset == 'CHASEDB1':
        raw_training_x_path, raw_training_y_path = raw_data_path_CHASEDB1[:2]
    elif dataset == 'STARE':
        raw_training_x_path, raw_training_y_path = raw_data_path_STARE[:2]

    if XorY == 0:
        raw_splited = raw_training_x_path.split('/')
    else:
        raw_splited = raw_training_y_path.split('/')

    data_path = ''.join([HDF5_data_path, dataset, '/', '/'.join(raw_splited[3:-1]), '/data.hdf5'])
    f = h5py.File(data_path, 'r')
    data = f['data']

    return data


def getTestData(XorYorMask, dataset):
    global HDF5_data_path, raw_data_path_DRIVE, raw_data_path_CHASEDB1, raw_data_path_STARE

    if dataset == 'DRIVE':
        raw_test_x_path, raw_test_y_path, raw_test_mask_path = raw_data_path_DRIVE[2:]
    elif dataset == 'CHASEDB1':
        raw_test_x_path, raw_test_y_path, raw_test_mask_path = raw_data_path_CHASEDB1[2:]
    elif dataset == 'STARE':
        raw_test_x_path, raw_test_y_path, raw_test_mask_path = raw_data_path_STARE[2:]

    if XorYorMask == 0:
        raw_splited = raw_test_x_path.split('/')
    elif XorYorMask == 1:
        raw_splited = raw_test_y_path.split('/')
    else:
        if not raw_test_mask_path:
            return None
        raw_splited = raw_test_mask_path.split('/')

    data_path = ''.join([HDF5_data_path, dataset, '/', '/'.join(raw_splited[3:-1]), '/data.hdf5'])
    f = h5py.File(data_path, 'r')
    data = f['data']

    return data
