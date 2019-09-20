from os import path


# define paths to the training and validation
TRAIN_IMAGES = r'D:\matth\Documents\projects\python\datasets\tiny_imagenet\dataset\tiny-imagenet-200\train'
VAL_IMAGES = r'D:\matth\Documents\projects\python\datasets\tiny_imagenet\dataset\tiny-imagenet-200\val\images'

# annotations for val images
VAL_MAPPINGS = r'D:\matth\Documents\projects\python\datasets\tiny_imagenet\dataset\tiny-imagenet-200\val\val_annotations.txt'

# Wordnet ids, convert annotations to readable names
WORDNET_IDS = r'D:\matth\Documents\projects\python\datasets\tiny_imagenet\dataset\tiny-imagenet-200\wnids.txt'
WORD_LABELS = r'D:\matth\Documents\projects\python\datasets\tiny_imagenet\dataset\tiny-imagenet-200\words.txt'

# use training data for testing set since we do not have access to test folders label
NUM_CLASSES = 200
NUM_TEST_IMAGES = 50 * NUM_CLASSES

# Define the HDF5 paths for train, val, test sets
TRAIN_HDF5 = r'D:\matth\Documents\projects\python\datasets\tiny_imagenet\dataset\train.hdf5'
VAL_HDF5 = r'D:\matth\Documents\projects\python\datasets\tiny_imagenet\dataset\val.hdf5'
TEST_HDF5 = r'D:\matth\Documents\projects\python\datasets\tiny_imagenet\dataset\test.hdf5'

# dataset RGB mean saved to disk
DATASET_MEAN = r'D:\matth\Documents\projects\python\datasets\tiny_imagenet\dataset\tiny-image-net-200-mean.json'

# output paths for model, figures, and json files, classification reports
OUTPUT_PATH = r'D:\matth\Documents\projects\python\datasets\tiny_imagenet\outputs'
MODEL_PATH = path.sep.join([OUTPUT_PATH, "checkpoints/epoch_70.hdf5"])
FIG_PATH = path.sep.join([OUTPUT_PATH, "tiny_imagenet_minigooglenet.png"])
JSON_PAT = path.sep.join([OUTPUT_PATH, "tiny_imagenet_minigooglenet.json"])

