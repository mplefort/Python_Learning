from BlogTutorials.pyimagesearch.nn.neuralnetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

print(" loading datasets")

digits = datasets.load_digits()
data = digits.data.astype("float")

data = (data - data.min()) / (data.max() - data.min())
print("Samples: {}, dim: {}".format(data.shape[0], data.shape[1]))

(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("training NN")
nn = NeuralNetwork( [trainX.shape[1], 32, 16, 16, 10] )
print(nn)
nn.fit(trainX, trainY, epochs = 1000)
predictions = nn.predict(testX)
predictions_max = predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))

