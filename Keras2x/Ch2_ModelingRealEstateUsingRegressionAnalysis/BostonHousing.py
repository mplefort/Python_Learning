import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


data = pd.read_csv('./data/train.csv')
data = data.drop('ID',axis=1)
scaler = MinMaxScaler()
scaler.fit(data)
DataScaled = scaler.fit_transform(data)


BHNames= ['crim','zn','indus','chas','nox','rm',
'age','dis','rad','tax','ptratio','black','lstat','medv']

DataScaled = pd.DataFrame(DataScaled, columns=BHNames)

# boxplot = DataScaled.boxplot(column=BHNames)
# plt.show()

# Check corelation of data to dependent variable to throw out uncorellated data
# correlation methods are: pearson, kendall, spearman.
CorData = DataScaled.corr(method='pearson')
# plt.matshow(CorData)
# plt.xticks(range(len(CorData.columns)), CorData.columns)
# plt.yticks(range(len(CorData.columns)), CorData.columns)
# plt.colorbar()
# plt.show()
# plot shows on bottom row corr. of independents on medv. values closes to 1 or -1 denote a positive or negative correlation.

'''
Splitting Data
 - Train:Test Split of 70:30 typical
'''
from sklearn.model_selection import train_test_split

X = DataScaled.drop('medv', axis=1)
print(X.describe())
Y = DataScaled['medv']
print(Y.describe())


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=5)
        # random_state - random number seed generator to ensure repeatable results when debugging

'''
Neural network Keras model
 Use Keras Sequential model
    1. import Sequential from keras.models
    2. stack layers with .add()
    3. config learning process with .compile()
    4. Train model using .fit()
'''

from keras.models import Sequential  # linear block of network layers
from keras.layers import Dense       # feedforward fullly connected layer
from keras import metrics            # performance measurement

model = Sequential()
model.add(Dense(20, input_dim=13, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))
    # 13 input
    # 20 dims of outputs/nodes,
    # relu - rectified linear y = 0 if x < 0, y=x if x >= 0
    # single output neuron with linear output layer

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    # adam - optimizer first order gradient based of stochasitc objective functions
    # mean_squared_error - MSE average square of errors
    # accuracy - metric for evaluating dring training

model.fit(X_train, Y_train, epochs=1000, verbose=1)
    # epochs - number of epochs to train hte model, an iteration over the entrie x,y data provided
    # verbose (0,1, or 2) - 0 silent, 1 progress bar, 2 one line per opock

model.summary()

# compare predictions of model to actual
Y_predKM = model.predict(X_test)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Keras Model')
print(score[0]) # given as MSE from loss definition

'''
Multiple Linear Regression Model
 - dependent variable related to 2 or more independent variables.
 - Use sklearn.linear_model LinearRegression class
'''

from sklearn.linear_model import LinearRegression
import numpy as np

LModel = LinearRegression()
LModel.fit(X_train, Y_train)
Y_predLM = LModel.predict(X_test)

a = np.linspace(0,1,20)
b = a

plt.figure(3)
plt.subplot(121)
plt.scatter(Y_test, Y_predKM)
plt.plot(a,b)
plt.xlabel("acutal values")
plt.ylabel("Predicted values")
plt.title('Keras Neural Network Model')

plt.subplot(122)
plt.scatter(Y_test, Y_predLM)
plt.plot(a,b)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("SKLearn Linear Regression Model")
plt.show()

# check MSE error for metrics
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(Y_test, Y_predLM)
print('Linear Regression Model')
print(mse)

'''
See that the regression ANN outperforms the linear regression model when comparing the MSE for the two models on
the test data sets. Quick example to setup a dense fully connected ANN.
'''


