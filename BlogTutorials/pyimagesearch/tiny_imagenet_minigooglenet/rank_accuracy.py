from BlogTutorials.pyimagesearch.tiny_imagenet_minigooglenet.config import tiny_imagenet_config as config
from BlogTutorials.pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from BlogTutorials.pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from BlogTutorials.pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
from BlogTutorials.pyimagesearch.io_module.hdf5datasetgenerator import HD5FDatasetGenerator
from BlogTutorials.pyimagesearch.utils.ranked import rank5_accuracy
from keras.models import load_model

means = json.loads(open(config.DATASET_MEAN).read())

# preprocessors
sp = SimplePreprocessor(64,64)  # resize image to size
mp = MeanPreprocessor(means["R"], means["B"], means["G"])
iap = ImageToArrayPreprocessor()


testGen = HD5FDatasetGenerator(config.TEST_HDF5, 64, preprocessor=[sp, mp, iap],
                               classes=config.NUM_CLASSES)

print("laoding model...")
model = load_model(config.MODEL_PATH)

# predictions form testing data
predictions = model.predict_generator(testGen.generator(),
                                      steps=testGen.numImages // 64,
                                      max_queue_size=64 * 2)

(rank1, rank5) = rank5_accuracy(predictions, testGen.db["labels"][i:])
print("rank1: {:.2f}%".format(rank1 * 100) )
print("rank5: {:.2f}%".format(rank5 * 100) )

testGen.close()