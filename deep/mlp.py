import numpy as np
import matplotlib.pyplot as plt

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def ReLU(X):
    return np.maximum(0, X)

def init_network():
    network = {}
    network['l1_w'] = [[0.1, 0.3, 0.7], [0.5, 0.8, 0.4]]
    network['l1_b'] = [1, 1, 1]

    network['l2_w'] = [[0.3, 0.1], [0.5, 0.6], [0.9, 0.3]]
    network['l2_b'] = [1, 1]

    network['l3_w'] = [0.4, 0.1]
    network['l3_b'] = [1]

    return network

def forward(inputs) :

    net = init_network()

    # 1 X 2 by 2 X 3 => 1 X 3
    layer1_weights = net['l1_w']
    layer1_bias = net['l1_b']

    z1 = np.dot(inputs, layer1_weights) + layer1_bias
    o1 = ReLU(z1)

    # 1 X 3 by 3 X 2 => 1 X 2
    layer2_weights = net['l2_w']
    layer2_bias = net['l2_b']

    z2 = np.dot(o1, layer2_weights) + layer2_bias
    o2 = ReLU(z2)

    # 1 X 2 by 2 X 1 = > 1 X 1
    layer3_weights = net['l3_w']
    layer3_bias = net['l3_b']

    z3 = np.dot(o2, layer3_weights) + layer3_bias
    rst = sigmoid(z3)

    return rst

# x = np.arange(-5, 5, 0.1)
# y1 = sigmoid(x)
# y2 = ReLU(x)

rst = forward([0.1, 0.3])
print(rst)


def save(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)
def load(name):
    try:
        with open(name, 'rb') as f:
            return pickle.load(f)
    except:
        return None

import sys, os
sys.path.append(os.pardir)
# from tensorflow import keras
import pickle

# fashion_mnist = keras.datasets.fashion_mnist
# (trd, trt), (tsd, tst) = fashion_mnist.load_data()  # 학습용과 테스트용으로 분리되어서 제공
# save(trd, "trd.pkl")
# save(tsd, "tsd.pkl")
# save(trt, "trt.pkl")
# save(tst, "tst.pkl")

trd = load("trd.pkl")
tsd = load("tsd.pkl")
trt = load("trt.pkl")
tst = load("tst.pkl")

print(trd.shape)
print(tsd.shape)
print(trt.shape)
print(tst.shape)

img = trd[11]
plt.imshow(img)
plt.show()
