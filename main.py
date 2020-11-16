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


    plt.figure(figsize=(20, 5))
    g=sns.countplot(y_train) #checking out how many pictures there are for each class- 6000 per class
    plt.title('Pictures per Label', fontsize=20)
    plt.xlabel('Labels', fontsize=20)
    for p in g.patches:
        g.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='gray', xytext=(0, 20),
                    textcoords='offset points')
    _ = g.set_ylim(0, 7000)  # To make space for the annotations

    plt.show()

    #displaying the first 12 pictures from the Train Set
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                   'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    fig = plt.figure(figsize=(10, 10))
    for j in range(12):
        j=j+1
        im = x_train[j].reshape([28, 28])
        label = class_names[y_train[j]]
        plt1=fig.add_subplot(3,4,j)
        plt.title(label)
        plt.imshow(im, cmap='gray', interpolation='nearest')

    plt.show()


if __name__ == '__main__':
    # just comment and uncomment which Question that you want to check :)
    Question1()
    #Question2()
    #Question3()
