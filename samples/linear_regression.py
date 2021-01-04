import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

import numpy as np
from autograd.tensor import Tensor
from autograd.helpers import show_data, PrintableData

np.random.seed(1)

def load_data() -> ('Tensor', 'Tensor'):
    x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
    y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)

    x_data = Tensor(x_data.reshape(-1, 1), requires_grad=True)
    y_label = Tensor(y_label.reshape(-1, 1), requires_grad=True)
    
    return x_data, y_label

def init_weights(w_val: float, b_val: float) -> ('Tensor', 'Tensor'):
    w = Tensor(w_val, requires_grad=True)
    b = Tensor(b_val, requires_grad=True)
    return w, b

def forward(x, w, b):
    return x * w + b

def backward(predicted, y):
    errors = predicted - y
    loss = (errors * errors).sum()
    loss.backward()
    return loss

def train(x, y, learning_rate = 0.001, epochs = 100) -> tuple():
    w, b = init_weights(0.39, 0.2)

    learning_rate = Tensor(learning_rate)
    for epoch in range(epochs):
        w.zero_grad()
        b.zero_grad()

        predicted = forward(x, w, b)
        loss = backward(predicted, y)

        epoch_loss = loss.data

        w = w - w.grad * learning_rate
        b = b - b.grad * learning_rate

        print(f'Epoch: {epoch}, Loss: {epoch_loss}')
    
    return w, b

def load_test_data():
    x_test = np.linspace(-1,11,10).reshape(1, -1).T
    return Tensor(x_test)

if __name__ == "__main__": 
    x, y = load_data()
    w, b = train(x, y)
    x_test = load_test_data()
    y_pred_plot = forward(x_test, w, b)
    line = PrintableData(x_test.data, y_pred_plot.data, 'r')
    data_points = PrintableData(x_test.data, y.data, '*')
    show_data([line, data_points])
