from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from BlogTutorials.pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from BlogTutorials.pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

# Step 1: load data from disk
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset", required=True, help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=3, help="# of nearest neighbros for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for k-NN distances (-1 uses all avalable cores)")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePathes = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor, load the dataset from disk, and reshape the data matrix
sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePathes, verbose=500)
data = data.reshape((data.shape[0], 3072))

# show some info on memory consumption of hte images
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

#  Step 2: Build test train split
# encode the labels as ints
le = LabelEncoder()
labels = le.fit_transform(labels)

# split data
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))
