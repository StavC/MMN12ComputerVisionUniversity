import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn import svm
from sklearn import metrics
import seaborn as sn
import pandas as pd
import scikitplot as skplt
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input


def BOWSift(train_x,train_y,test_x,test_y):


    #Starting Question 1
    POI=[]
    k=100 #feel free to change this hyperparmeter to test different sizes
    mbk=MiniBatchKMeans(k)
    bins=100 #feel free to change this hyperparmeter to test different sizes
    c_array=[1,10,50,200]

    sift = cv.xfeatures2d.SIFT_create()

    # using SIFT
    for train_x_img in train_x:
        img=cv.imread(train_x_img)
        imgGray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        _, des = sift.detectAndCompute(imgGray,None)
        POI.append(des)
        mbk.partial_fit(des)

    #building histogram
    hist=[]
    for des in POI:
        pred=mbk.predict(des)
        his,_=np.histogram(pred,bins=bins)
        hist.append(his)

    #SVM

    for c in c_array: #trying out diffrent values for C
        linear_svm=svm.SVC(C=c,max_iter=5000,random_state=20,probability=True,kernel='linear') #Model that supports ROC Plot
        #linear_svm=svm.LinearSVC(C=c,max_iter=5000,dual=False,random_state=20) #Model without ROC Plot


        linear_svm.fit(hist,train_y)

       #starting the testing phase
        POI_test=[]
        for picture in test_x:
            img = cv.imread(picture)
            imgGray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            _,des=sift.detectAndCompute(imgGray,None)
            POI_test.append(des)

        hist_test=[]
        for des in POI_test:
            pred=mbk.predict(des)
            his,_=np.histogram(pred,bins=bins)
            hist_test.append(his)

        predictions=linear_svm.predict(hist_test)
        predictionsProb=linear_svm.predict_proba(hist_test)

        skplt.metrics.plot_roc(test_y, predictionsProb)
        plt.show()

        print(f" Class report for classifier {linear_svm},\n{metrics.classification_report(test_y,predictions)}")
        report=metrics.classification_report(test_y, predictions,output_dict=True)
        conf = metrics.confusion_matrix(test_y, predictions)
        #print(conf)


        df_cm = pd.DataFrame(conf, ['coast', 'forest', 'highway', 'insidecity', 'mountain', 'opencountry', 'street',
                                    'tallbuilding'],
                             ['coast', 'forest', 'highway', 'insidecity', 'mountain', 'opencountry', 'street',
                              'tallbuilding'])
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()




def BOWDeepLearning(train_x,train_y,test_x,test_y):


    input_tensor = Input(shape=(None, None, 3))
    model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
    print(model.summary())
    '''
    im = plt.imread(train_x[0])  # read an image file
    im = np.expand_dims(im, axis=0)
    im = preprocess_input(im)
    pred = model.predict(im)
    '''

    # Starting Question 1
    POI = []
    k=100 #feel free to change this hyperparmeter to test different sizes
    mbk = MiniBatchKMeans(k)
    c_array = [1, 10, 50]
    bins=100 #feel free to change this hyperparmeter to test different sizes


    for train_x_img in train_x:
        im = cv.imread(train_x_img)
        im = np.expand_dims(im, axis=0)
        im = preprocess_input(im)
        des=model.predict(im)
        des=des.reshape([1024,32]) # #feel free to change this hyperparmeter to test different shapes
        POI.append(des)
        mbk.partial_fit(des)

    # building histogram
    hist = []
    for des in POI:
        pred = mbk.predict(des)
        his, _ = np.histogram(pred, bins=bins)
        hist.append(his)

    # SVM


    for c in c_array:  # trying out diffrent values for C

        linear_svm=svm.SVC(C=c,max_iter=5000,random_state=20,probability=True,kernel='linear') #Model that supports ROC
        #linear_svm = svm.LinearSVC(C=c, max_iter=5000, dual=False, random_state=20)# Model that not support ROC

        linear_svm.fit(hist, train_y)

        # starting the testing phase
        POI_test = []
        for picture in test_x:
            im = cv.imread(picture)
            im = np.expand_dims(im, axis=0)
            im = preprocess_input(im)
            des=model.predict(im)
            des = des.reshape([1024,32]) # #feel free to change this hyperparmeter to test different shapes
            POI_test.append(des)

        hist_test = []
        for des in POI_test:
            pred = mbk.predict(des)
            his, _ = np.histogram(pred, bins=bins)
            hist_test.append(his)

        predictions = linear_svm.predict(hist_test)
        predictionsProb = linear_svm.predict_proba(hist_test)

        skplt.metrics.plot_roc(test_y, predictionsProb)
        plt.show()

        print(f" Class report for classifier {linear_svm},\n{metrics.classification_report(test_y, predictions)}")
        conf = metrics.confusion_matrix(test_y, predictions)
        #print(conf)

        df_cm = pd.DataFrame(conf, ['coast', 'forest', 'highway', 'insidecity', 'mountain', 'opencountry', 'street',
                                    'tallbuilding'],
                             ['coast', 'forest', 'highway', 'insidecity', 'mountain', 'opencountry', 'street',
                              'tallbuilding'])
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


if __name__ == '__main__':


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
    ratios=0.8 # the ratio of Train\Test
    ratio=int(minPictures*ratios)  # im balancing the dataset so each class will have equal amount of pictures to avoid overfit!
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
    BOWSift(train_x,train_y,test_x,test_y)
    BOWDeepLearning(train_x,train_y,test_x,test_y)


