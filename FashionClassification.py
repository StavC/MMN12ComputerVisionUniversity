import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

import gzip





def Question1(x_train,y_train,x_test,y_test):
    scores = []
    x = x_train
    y = y_train
    for k in range(1, 11):
        KNN = KNeighborsClassifier(n_neighbors=k, n_jobs=-1) #remove the param n_jobs if you dont want to overload your computer
        KNN.fit(x, y)
        #print(f"finished fiting :{k} ") # its possible to uncomment these prints to see the progress of the algorithem
        scores.append(KNN.score(x_test, y_test))
        #print(f'finished predicting: {k}')

    plt.plot(scores, label='score')
    plt.xticks(np.arange(len(scores)), np.arange(1, len(scores) + 1))
    plt.legend()
    plt.show()



def KNNHelper(x_train,y_train,x_test,y_test): # Helpfull function for Question 2.F
    scores=[]
    for k in range(1,11):
        KNN = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        KNN.fit(x_train, y_train)
        #print(f"finished fiting :{k} ")
        scores.append(KNN.score(x_test, y_test))
        #print('finished predicting')
    return scores

def Question2(x_train,y_train,x_test,y_test):
    #Starting Question 2.B
    pca = PCA(random_state=20)
    pca.fit(x_train)
    # print(pca.singular_values_)
    # print(pca.n_components)
    # print(pca.explained_variance_ratio_)

    fig = plt.figure(figsize=(10, 10))
    pic = np.zeros([28, 28])
    for j in range(6):
        p = pca.components_[j].reshape(28, 28)
        plt1 = fig.add_subplot(3, 2, j + 1)
        pic = pic + p
        plt.imshow(p, cmap='gray', interpolation='nearest')
    plt.show()

    plt.imshow(pic, cmap='gray')  # wanted to see the combined pic (not related to the must do work)
    plt.show()

    plt.imshow(pca.mean_.reshape([28, 28]), cmap='gray')  # the mean
    plt.show()
    #Finished 2.B

    #Starting 2.C+D
    sns.set()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    #checking out how many componenets are required to get to 80% and 95% exactly on point!
    sum=0
    i=0
    check80=True
    while True:
        sum= sum +pca.explained_variance_ratio_[i]
        if sum >=0.80 and check80==True:
            print(f'after {i} components we got to 80% ')
            check80=False
        if sum >= 0.95:
            print(f'after {i} components we got to 95% ')
            break
        i = i + 1
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    # Finished 2.C+D

    #Starting 2.E
    pca = PCA(n_components=2)
    projected = pca.fit_transform(x_train)
    #print(x_train.shape)
    #print(projected.shape)

    plt.scatter(projected[:, 0], projected[:, 1],
                c=y_train, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('gist_rainbow', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()

    # starting 2.F

    compoList = [2, 10, 20]

    for comp in compoList:
        pcaForKNN = PCA(n_components=comp)
        pcaForKNN.fit(x_train)
        x_train_ready = pcaForKNN.transform(x_train)
        x_test_ready = pcaForKNN.transform(x_test)

        KNNScore = KNNHelper(x_train_ready, y_train, x_test_ready, y_test)
        plt.figure()
        plt.title(f'componenets {comp}')
        plt.plot(range(1, 11), KNNScore, marker='x')

    plt.show()
    #Finished 2.F

    sns.reset_orig()
    # Starting G
    comp = [2, 5, 10, 50, 100, 150, 200]
    fig = plt.figure(figsize=(28, 10))
    plt1 = fig.add_subplot(1, 8, 1)
    plt.imshow(x_test[0].reshape([28, 28]), cmap='gray')
    plt.title('Original Picture')
    for i in range(len(comp)):
        pca = PCA(n_components=comp[i])
        pca.fit(x_train, y_train)
        test_tran = pca.transform(x_test[0].reshape(1, -1))
        inverse_trans = pca.inverse_transform(test_tran).reshape([28, 28])
        plt1 = fig.add_subplot(1, 8, i + 2)
        plt.imshow(inverse_trans, cmap='gray', interpolation='nearest')

        plt.title(f'comps: {comp[i]}')
    plt.show()
    # Finished G


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def Question3(x_train,y_train,x_test,y_test):
    #Staring 3.C+D
    clf=LinearDiscriminantAnalysis()
    clf.fit(x_train,y_train)
    sns.set()
    plt.plot(np.cumsum(clf.explained_variance_ratio_))
    # checking out how many componenets are required to get to 80% and 95% exactly on point!
    sum = 0
    i = 0
    check80 = True
    while True:
        sum = sum + clf.explained_variance_ratio_[i]
        if sum >= 0.80 and check80 == True:
            print(f'after {i} components we got to 80% ')
            check80 = False
        if sum >= 0.95:
            print(f'after {i} components we got to 95% ')
            break
        i = i + 1
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    #Finished 3.C+D

    #Starting 3.E

    clf = LinearDiscriminantAnalysis(n_components=2)
    projected = clf.fit_transform(x_train,y_train)
    #print(x_train.shape)
    #print(projected.shape)

    plt.scatter(projected[:, 0], projected[:, 1],
                c=y_train, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('gist_rainbow', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()

    #Finished 3.E

    # starting 3.F

    compoList = [2, 5, 8] #should be 2,10,20 but the maximum is 8

    for comp in compoList:
        ldaForKNN = LinearDiscriminantAnalysis(n_components=comp)
        ldaForKNN.fit(x_train,y_train)
        x_train_ready = ldaForKNN.transform(x_train)
        x_test_ready = ldaForKNN.transform(x_test)

        KNNScore = KNNHelper(x_train_ready, y_train, x_test_ready, y_test)
        plt.figure()
        plt.title(f'componenets {comp}')
        plt.plot(range(1, 8), KNNScore, marker='x')

    plt.show()
    # Finished 3.F







def main():
    # Opening the DataSet
    y_train_path = 'train-labels-idx1-ubyte.gz'
    x_train_path = 'train-images-idx3-ubyte.gz'

    x_test_path = 't10k-images-idx3-ubyte.gz'
    y_test_path = 't10k-labels-idx1-ubyte.gz'

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
    g = sns.countplot(y_train)  # checking out how many pictures there are for each class- 6000 per class
    plt.title('Pictures per Label', fontsize=20)
    plt.xlabel('Labels', fontsize=20)
    for p in g.patches:
        g.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', fontsize=11, color='gray', xytext=(0, 20),
                   textcoords='offset points')
    _ = g.set_ylim(0, 7000)  # To make space for the annotations

    plt.show()

    # displaying the first 12 pictures from the Train Set
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                   'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    fig = plt.figure(figsize=(10, 10))
    for j in range(12):
        j = j + 1
        im = x_train[j].reshape([28, 28])
        label = class_names[y_train[j]]
        plt1 = fig.add_subplot(3, 4, j)
        plt.title(label)
        plt.imshow(im, cmap='gray', interpolation='nearest')

    plt.show()
    # just comment and uncomment which Question that you want to check :)
    #Question1(x_train,y_train,x_test,y_test)
    #Question2(x_train,y_train,x_test,y_test)
    #Question3(x_train,y_train,x_test,y_test)

if __name__ == '__main__':
    main()
