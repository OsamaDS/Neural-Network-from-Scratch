import numpy as np

from DenseLayers import Dense_Layer
from ActivationLayer import Activation_Layer
from ActivationFunction import tanh, tanh_prime
from Loss import mse, mse_prime
from NeuralNetwork import Neural_Network

X_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

model = Neural_Network()
model.add(Dense_Layer(2,3))
model.add(Activation_Layer(tanh, tanh_prime))
model.add(Dense_Layer(3,1))
model.add(Activation_Layer(tanh, tanh_prime))

model.compile(mse, mse_prime)

model.fit(X_train,y_train, epochs=200, learning_rate=0.1)

y_pred = model.predict(X_train)
print(y_pred)