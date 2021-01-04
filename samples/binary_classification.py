import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from autograd.tensor import Tensor
from autograd.module import Module
from autograd.functions import Sigmoid, Tanh, Relu
from autograd.parameter import Parameter
from autograd.optimizer import SGD
from autograd.helpers import Dependency
from autograd.loss import CrossEntropyBinary

np.random.seed(2)

def load_dataset_1():
    np.random.seed(1)
    m = 300
    N = int(m/2)
    D = 2
    X = np.zeros((m,D))
    Y = np.zeros((m,1))
    a = 8
    
    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
    
    return X, Y

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[x_grid.reshape(-1), y_grid.reshape(-1)])
    Z = Z.reshape(x_grid.shape)
    # Plot the contour and training examples
    plt.figure(figsize=(12,6))
    plt.contourf(x_grid, y_grid, Z, cmap=plt.cm.Spectral)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Spectral)
    plt.colorbar()

class NeuralNet(Module):
    def __init__(self, input_size, hidden_size,  num_classes):
        self.w1 = Parameter(input_size, hidden_size)
        self.b1 = Parameter(hidden_size)

        self.w2 = Parameter(hidden_size, num_classes)
        self.b2 = Parameter(num_classes)
    
    def forward(self, input_data):
        x1 = input_data @ self.w1 + self.b1
        a1 = Tanh()(x1)
        x2 = a1 @ self.w2 + self.b2
        a2 = Sigmoid()(x2)    
        return a2

    def predict(self, input_data):
        return np.round(self.forward(input_data).data)

X,Y = load_dataset_1()
X = Tensor(X)
Y = Tensor(Y)
print(f"X_shape: {X.shape}, Y_shape: {Y.shape}")

input_size = X.shape[1]
hidden_size = 4
num_classes = 1

model = NeuralNet(input_size, hidden_size, num_classes)

optimizer = SGD(lr=1)
num_epochs = 10001

for epoch in range(num_epochs):
    # Forward pass
    outputs = model.forward(X)
    loss = CrossEntropyBinary()(Y, outputs)
    
    # Backward and optimize
    model.zero_grad()
    loss.backward()
    optimizer.step(model)
    
    if (epoch+1) % 500 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, num_epochs, loss.data))

predictions = model.predict(X)
accuracy = accuracy_score(Y.data.flatten(), predictions.flatten())
print("Trainning Accuracy: %f"%(accuracy)+"%")

plot_decision_boundary(lambda x: model.predict(Tensor(x)), X.data, Y.data)
plt.title("Decision Boundary for hidden layer size 4")
plt.show()

