import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


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







if __name__ == '__main__':
    main2()
