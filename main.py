import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

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




    myPCA(x_train,y_train,x_test,y_test)
    #myKNN(x_train,y_train,x_test,y_test)


def myPCA(x_train,y_train,x_test,y_test):
    pca = PCA( random_state=20)
    pca.fit(x_train)
    #print(pca.singular_values_)
    #print(pca.n_components)
    #print(pca.explained_variance_ratio_)
    #print(pca.components_[1].shape)

    fig = plt.figure(figsize=(10, 10))
    pic = np.zeros([28, 28])
    for j in range(6):
        p = pca.components_[j].reshape(28, 28)
        plt1 = fig.add_subplot(3, 2, j + 1)
        pic = pic + p
        plt.imshow(p, cmap='gray', interpolation='nearest')
    plt.show()

    plt.imshow(pic, cmap='gray') # wanted to see the combined pic (not related to the must do work)
    plt.show()

    plt.imshow(pca.mean_.reshape([28,28]),cmap='gray') # the mean of
    plt.show()


    sns.set()

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

    pca=PCA(n_components=2)
    projected = pca.fit_transform(x_train)
    print(x_train.shape)
    print(projected.shape)

    plt.scatter(projected[:, 0], projected[:, 1],
                c=y_train, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('gist_rainbow', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()

    #starting F

    compoList=[2,10,20]

    for comp in compoList:
        pcaForKNN=PCA(n_components=comp)
        pcaForKNN.fit(x_train,y_train)
        x_train_ready=pcaForKNN.transform(x_train)
        x_test_ready=pcaForKNN.transform(x_test)

        KNNScore=KNNHelper(x_train_ready,y_train,x_test_ready,y_test)
        plt.figure()
        plt.title(f'componenets {comp}')
        plt.plot(range(1,11),KNNScore,marker='x')
    plt.show()

def KNNHelper(x_train,y_train,x_test,y_test):
    scores=[]
    for k in range(1,11):
        KNN = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        KNN.fit(x_train, y_train)
        print(f"finished fiting :{k} ")
        scores.append(KNN.score(x_test, y_test))
        print('finished predicting')
    return scores



def myKNN(x_train,y_train,x_test,y_test):
    scores = []
    from sklearn.utils import shuffle
    # x, y = shuffle(x_train, y_train, random_state=2)
    x = x_train
    y = y_train
    for k in range(1, 11):
        KNN = KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
        KNN.fit(x, y)
        print(f"finished fiting :{k} ")
        scores.append(KNN.score(x_test, y_test))
        print('finished predicting')

    plt.plot(scores, label='score')
    plt.xticks(np.arange(len(scores)), np.arange(1, len(scores) + 1))
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # just comment and uncomment which Question that you want to check :)
    Question1()
    #Question2()
    #Question3()
