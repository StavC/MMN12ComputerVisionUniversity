import os
#import cv2 as cv
import cv2 as cv
import numpy as np
#import matplotlib.pyplot as plt
import pysift
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
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
    #now the Dataset is ready, we lost some data along the way but each class have equal amouns of pictures in training,testing set

    #Starting Question 1
    POI=[]
    #dense = cv.FastFeatureDetector_create()
    mbk=MiniBatchKMeans(150)
    kmeans=KMeans(150)
    i=0

    sift = cv.xfeatures2d.SIFT_create()

    for train_x_img in train_x:
        print(train_x_img)
        img=cv.imread(train_x_img)
        imgGray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(imgGray,None)
        POI.append(des)
        mbk.partial_fit(des)


    print(mbk.cluster_centers_)
    hist=[]
    for des in POI:
        im_mean=mbk.predict(des)
        his,bins=np.histogram(im_mean,bins=149)
        hist.append(his)


    print(POI)
    print(len(POI))




if __name__ == '__main__':
    main2()
