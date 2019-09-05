# Image dir
IMAGES_PATH = r"D:\matth\Documents\projects\python\datasets\kaggle_dogs_vs_cats\train"

# 2 classes and 25,000 total images (12500 cats, 12500 dogs)  80% train, 10% val, 10% test
NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250*NUM_CLASSES   # 10%
NUM_TEST_IMAGES = 1250*NUM_CLASSES  # 10%

# output paths
TRAIN_HDF5 = r"D:\matth\Documents\projects\python\datasets\kaggle_dogs_vs_cats\hdf5\train.hdf5"
VAL_HDF5   = r"D:\matth\Documents\projects\python\datasets\kaggle_dogs_vs_cats\hdf5\val.hdf5"
TEST_HDF5  = r"D:\matth\Documents\projects\python\datasets\kaggle_dogs_vs_cats\hdf5\test.hdf5"

MODEL_PATH = r"D:\matth\Documents\projects\python\datasets\kaggle_dogs_vs_cats\models\alexnat_dogs_vs_cats.model"
DATA_SET_MEAN = r"D:\matth\Documents\projects\python\datasets\kaggle_dogs_vs_cats\models\dogs_vs_cats_rgbmean.json"
OUTPUT_PATH = r"D:\matth\Documents\projects\python\datasets\kaggle_dogs_vs_cats\models"