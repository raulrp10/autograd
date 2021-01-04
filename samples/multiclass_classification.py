import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from autograd.tensor import Tensor
from autograd.module import Module
from autograd.functions import Sigmoid, Tanh, Relu, Softmax
from autograd.parameter import Parameter
from autograd.optimizer import SGD
from autograd.helpers import Dependency
from autograd.loss import CrossEntropyBinary, CrossEntropy

np.random.seed(2)

def load_dataset_1():
    np.random.seed(1)    
    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    Y = np.zeros((N*K,1), dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
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

def convert_to_vector(X, categories):
    ''' Funcion que transforma un array, en un one-hot vector '''
    m = X.size
    result = np.zeros( shape = (m, categories))
    for i in range(m):
        result[i, X[i,0]] = 1
    return result

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
        a2 = Softmax()(x2)    
        return a2

    def predict(self, input_data):
        Y2_hat = self.forward(input_data).data
        predictions = np.argmax(Y2_hat, axis = 1)
        predictions = predictions.reshape(predictions.size, 1)
        return predictions

X,y_true = load_dataset_1()
X = Tensor(X)
Y = Tensor(convert_to_vector(y_true, 3))

input_size = X.shape[1]
hidden_size = 4
num_classes = 3

model = NeuralNet(input_size, hidden_size, num_classes)

optimizer = SGD(lr=0.1)
num_epochs = 10001

for epoch in range(num_epochs):
    # Forward pass
    outputs = model.forward(X)
    loss = CrossEntropy()(Y, outputs)
    
    # Backward and optimize
    model.zero_grad()
    loss.backward()
    optimizer.step(model)
    
    if (epoch+1) % 500 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, num_epochs, loss.data))

predictions = model.predict(X)
accuracy = accuracy_score(y_true.flatten(), predictions.flatten())
print("Trainning Accuracy: %f"%(accuracy)+"%")


plot_decision_boundary(lambda x: model.predict(Tensor(x)), X.data, Y.data)
plt.title("Decision Boundary for hidden layer size 4")
plt.show()

