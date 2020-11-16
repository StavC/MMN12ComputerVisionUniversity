import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import gzip


def Question1():

    y_train_path='train-labels-idx1-ubyte.gz'
    x_train_path='train-images-idx3-ubyte.gz'

    x_test_path='t10k-images-idx3-ubyte.gz'
    y_test_path='t10k-labels-idx1-ubyte.gz'

    with gzip.open(y_train_path, 'rb') as lpath:
        y_train = np.frombuffer(lpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(x_train_path, 'rb') as ipath:
        x_train = np.frombuffer(ipath.read(), dtype=np.uint8,
                               offset=16).reshape(len(y_train), 784)

    with gzip.open(y_test_path, 'rb') as lpath:
        y_test = np.frombuffer(lpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(x_test_path, 'rb') as ipath:
        x_test = np.frombuffer(ipath.read(), dtype=np.uint8,
                               offset=16).reshape(len(y_test), 784)


    # show sample #100 from the train set
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                   'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    i = 100
    im = x_train[i].reshape([28, 28])
    label = class_names[y_train[i]]
    plt.figure()
    plt.imshow(im, cmap='gray', interpolation='nearest')
    plt.title(label)
    plt.show()


if __name__ == '__main__':
    # just comment and uncomment which Question that you want to check :)
    Question1()
    #Question2()
    #Question3()
