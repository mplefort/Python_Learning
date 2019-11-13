from os import path

# path to dataset
BASE_PATH = r'D:\matth\Documents\projects\python\datasets\fer2013'

#
INPUT_PATH = path.sep.join([BASE_PATH, r"fer2013\fer2013.csv"])

NUM_CLASSES = 6

# HDF5 file paths
TRAIN_HDF5 = path.sep.join([BASE_PATH, r"hdf5\train.hdf5"])
VAL_HDF5 = path.sep.join([BASE_PATH, r"hdf5\val.hdf5"])
TEST_HDF5 = path.sep.join([BASE_PATH, r"hdf5\test.hdf5"])

BATCH_SIZE = 128

OUTPUT_PATH = path.sep.join([BASE_PATH, "output"])




