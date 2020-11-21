import os
#import cv2 as cv
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pysift
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn import svm
from sklearn import metrics
from itertools import cycle
import tensorflow as tf


def main2():

    path = 'spatial_envelope_256x256_static_8outdoorcategories'
    coast=[]
    forest=[]
    highway=[]
    insidecity=[]
    mountain=[]
    opencountry=[]
    street=[]
    tallbuilding=[]
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if 'coast' in file:
                coast.append(os.path.join(r, file))
            if 'forest' in file:
                forest.append(os.path.join(r, file))
            if 'highway' in file:
                highway.append(os.path.join(r, file))
            if 'insidecity' in file:
                insidecity.append(os.path.join(r, file))
            if 'mountain' in file:
                mountain.append(os.path.join(r, file))
            if 'opencountry' in file:
                opencountry.append(os.path.join(r, file))
            if 'street' in file:
                street.append(os.path.join(r, file))
            if 'tallbuilding' in file:
                tallbuilding.append(os.path.join(r, file))
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]
    minPictures=(min(len(coast),len(forest),len(highway),len(insidecity),len(mountain),len(opencountry),len(street),len(tallbuilding)))
    print(f'the least amount of pictures from all of the categories is : {minPictures}')
    ratio_list=[0.8]
    for ratios in ratio_list:
        ratio=int(minPictures*ratios) # im balancing the dataset so each class will have equal amount of pictures to avoid overfit!
        #for i in range (ratio):
        for i in range (minPictures):
            if i< ratio:

                train_x.append(coast[i])
                train_y.append('coast')
                train_x.append(forest[i])
                train_y.append('forest')
                train_x.append(highway[i])
                train_y.append('highway')
                train_x.append(insidecity[i])
                train_y.append('insidecity')
                train_x.append(mountain[i])
                train_y.append('mountain')
                train_x.append(opencountry[i])
                train_y.append('opencountry')
                train_x.append(street[i])
                train_y.append('street')
                train_x.append(tallbuilding[i])
                train_y.append('tallbuilding')
            else:
                test_x.append(coast[i])
                test_y.append('coast')
                test_x.append(forest[i])
                test_y.append('forest')
                test_x.append(highway[i])
                test_y.append('highway')
                test_x.append(insidecity[i])
                test_y.append('insidecity')
                test_x.append(mountain[i])
                test_y.append('mountain')
                test_x.append(opencountry[i])
                test_y.append('opencountry')
                test_x.append(street[i])
                test_y.append('street')
                test_x.append(tallbuilding[i])
                test_y.append('tallbuilding')


        print(f'train_x and train_y sizes are:  {len(train_x)},{len(train_y)}') #making sure the math done right and we got 260*8*0.8=1664 pictures in the training set
        print(f'test_x and test_y sizes are:  {len(test_x)},{len(test_y)}') #making sure the math done right and we got 260*8*0.2=416 pictures in the testing set
    #now the Dataset is ready, we lost some data along the way but each class have equal amounts of pictures in training,testing set

    #Starting Question 1
    POI=[]
    mbk=MiniBatchKMeans(200)
    c_array=[1,10,50,200,500,1000]

    sift = cv.xfeatures2d.SIFT_create()


    for train_x_img in train_x:
        img=cv.imread(train_x_img)
        imgGray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        _, des = sift.detectAndCompute(imgGray,None)
        POI.append(des)
        print(des.shape)
        mbk.partial_fit(des)

    #building histogram
    hist=[]
    for des in POI:
        pred=mbk.predict(des)
        his,_=np.histogram(pred,bins=200)
        hist.append(his)

    #SVM

    for c in c_array: #trying out diffrent values for C

        linear_svm=svm.LinearSVC(C=c,max_iter=5000,dual=False,random_state=20)
        linear_svm.fit(hist,train_y)

       #starting the testing phase
        POI_test=[]
        for picture in test_x:
            img = cv.imread(picture)
            imgGray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            _,des=sift.detectAndCompute(imgGray,None)
            POI_test.append(des)
            mbk.partial_fit(des)

        hist_test=[]
        for des in POI_test:
            pred=mbk.predict(des)
            his,_=np.histogram(pred,bins=200)
            hist_test.append(his)

        predictions=linear_svm.predict(hist_test)
        y_score=linear_svm.decision_function(hist_test)

        print(f" Class report for classifier {linear_svm},\n{metrics.classification_report(test_y,predictions)}")
        '''
        fpr,tpr,_=metrics.roc_curve(test_y,y_score=y_score)
        auc=metrics.auc(fpr,tpr)
        plt.figure()
        plt.plot(fpr,tpr,label=f'auc :{auc}')
        plt.show()
        '''

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input
def BOWDeepLearning():

    path = 'spatial_envelope_256x256_static_8outdoorcategories'
    coast=[]
    forest=[]
    highway=[]
    insidecity=[]
    mountain=[]
    opencountry=[]
    street=[]
    tallbuilding=[]
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if 'coast' in file:
                coast.append(os.path.join(r, file))
            if 'forest' in file:
                forest.append(os.path.join(r, file))
            if 'highway' in file:
                highway.append(os.path.join(r, file))
            if 'insidecity' in file:
                insidecity.append(os.path.join(r, file))
            if 'mountain' in file:
                mountain.append(os.path.join(r, file))
            if 'opencountry' in file:
                opencountry.append(os.path.join(r, file))
            if 'street' in file:
                street.append(os.path.join(r, file))
            if 'tallbuilding' in file:
                tallbuilding.append(os.path.join(r, file))
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]
    minPictures=(min(len(coast),len(forest),len(highway),len(insidecity),len(mountain),len(opencountry),len(street),len(tallbuilding)))
    print(f'the least amount of pictures from all of the categories is : {minPictures}')
    ratio_list=[0.8]
    for ratios in ratio_list:
        ratio=int(minPictures*ratios) # im balancing the dataset so each class will have equal amount of pictures to avoid overfit!
        #for i in range (ratio):
        for i in range (minPictures):
            if i< ratio:

                train_x.append(coast[i])
                train_y.append('coast')
                train_x.append(forest[i])
                train_y.append('forest')
                train_x.append(highway[i])
                train_y.append('highway')
                train_x.append(insidecity[i])
                train_y.append('insidecity')
                train_x.append(mountain[i])
                train_y.append('mountain')
                train_x.append(opencountry[i])
                train_y.append('opencountry')
                train_x.append(street[i])
                train_y.append('street')
                train_x.append(tallbuilding[i])
                train_y.append('tallbuilding')
            else:
                test_x.append(coast[i])
                test_y.append('coast')
                test_x.append(forest[i])
                test_y.append('forest')
                test_x.append(highway[i])
                test_y.append('highway')
                test_x.append(insidecity[i])
                test_y.append('insidecity')
                test_x.append(mountain[i])
                test_y.append('mountain')
                test_x.append(opencountry[i])
                test_y.append('opencountry')
                test_x.append(street[i])
                test_y.append('street')
                test_x.append(tallbuilding[i])
                test_y.append('tallbuilding')


        print(f'train_x and train_y sizes are:  {len(train_x)},{len(train_y)}') #making sure the math done right and we got 260*8*0.8=1664 pictures in the training set
        print(f'test_x and test_y sizes are:  {len(test_x)},{len(test_y)}') #making sure the math done right and we got 260*8*0.2=416 pictures in the testing set
    #now the Dataset is ready, we lost some data along the way but each class have equal amounts of pictures in training,testing set


    input_tensor = Input(shape=(None, None, 3))
    model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
    print(model.summary())

    im = plt.imread(train_x[0])  # read an image file
    im = np.expand_dims(im, axis=0)
    im = preprocess_input(im)
    pred = model.predict(im)

    # Starting Question 1
    POI = []
    mbk = MiniBatchKMeans(200)
    c_array = [1, 10, 50]


    for train_x_img in train_x:
        im = cv.imread(train_x_img)
        im = np.expand_dims(im, axis=0)
        im = preprocess_input(im)
        des=model.predict(im)
        #des=des.reshape([-1,1])
        print(des.shape)
        des=des.reshape([1024,32])
        print(des.shape)
        POI.append(des)
        mbk.partial_fit(des)

    # building histogram
    hist = []
    for des in POI:
        pred = mbk.predict(des)
        his, _ = np.histogram(pred, bins=200)
        hist.append(his)

    # SVM


    for c in c_array:  # trying out diffrent values for C

        linear_svm = svm.LinearSVC(C=c, max_iter=5000, dual=False, random_state=20)
        linear_svm.fit(hist, train_y)

        # starting the testing phase
        POI_test = []
        for picture in test_x:
            im = cv.imread(picture)
            im = np.expand_dims(im, axis=0)
            im = preprocess_input(im)
            des=model.predict(im)
            #des = des.reshape([-1, 1])
            des = des.reshape([1024,32])
            POI_test.append(des)
            mbk.partial_fit(des)

        hist_test = []
        for des in POI_test:
            pred = mbk.predict(des)
            his, _ = np.histogram(pred, bins=200)#200 normally
            hist_test.append(his)

        predictions = linear_svm.predict(hist_test)
        y_score = linear_svm.decision_function(hist_test)

        print(f" Class report for classifier {linear_svm},\n{metrics.classification_report(test_y, predictions)}")


if __name__ == '__main__':
    #main2()
    BOWDeepLearning()

#(512, 64)
#0.71

#1024,32  200 normally bins and such MBK
#0.73

#2048,16
#0.46

#1024,32  100 normally bins and such MBK
#0.74



#1024,32  300 normally bins and such MBK
#
