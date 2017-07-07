import cv2
import numpy
import os
import imageio
import h5py
import imageProcess
from scipy import ndimage

import matplotlib.pyplot as plt

##############################################################
# GET SOME KEY INFROMATION FROM HDF5 FILE POINTER
##############################################################
def getVitals(fp):
    [row,col,numFrames] = fp.attrs['row'],fp.attrs['col'],fp.attrs['numFrames']
    frameList = range(1,numFrames+1)
    #frameList = fp.attrs['frameList']
    return row,col,numFrames,frameList
##############################################################


##############################################################
# GECTH() LIKE FUNCTION
##############################################################
def wait():
    raw_input()
    return 0
##############################################################


##############################################################
# FIND SEED POINTS FOR ALL REGIONS OF THE SEGMENTED IMAGE
##############################################################
def findLocalMinima(bImg,gImg,size=1):
    row,col = gImg.shape
    labelImg = numpy.zeros([row,col], dtype='uint16')
    seedList,connectedSeedLabel = [],[]
    R,C = numpy.where(bImg==True)
    for r,c in zip(R,C):
        if (r>=size and c>=size and r<row-size and c<=col-size):
            r0,c0,value0 = r,c,gImg[r,c]
            while (1):
                flag = 0
                bImgT = bImg[r0-size:r0+size+1,c0-size:c0+size+1]
                gImgT = gImg[r0-size:r0+size+1,c0-size:c0+size+1]
                gImgT = gImgT*bImgT
                gImgT_max = numpy.max(gImgT)
                if (value0 < gImgT_max):
                    r1,c1 = numpy.where(gImgT==gImgT_max)
                    r1,c1 = r1[0],c1[0]
                    r0,c0,value0 = r0-size+r1,c0-size+c1,gImgT_max
                    flag = 1
                if (flag == 0):
                    if ([r0,c0] not in seedList):
                        seedList.append([r0,c0])
                    break
    bImgSeed = numpy.zeros([row,col], dtype='bool')
    for r,c in seedList:
        bImgSeed[r,c] = True
    labelSeed,numSeed=ndimage.label(bImgSeed,structure=[[1,1,1],[1,1,1],[1,1,1]])
    for r,c in seedList:
        connectedSeedLabel.append(labelSeed[r,c])
    for r,c in zip(R,C):
        if (r>=size and c>=size and r<row-size and c<=col-size):
            r0,c0,value0 = r,c,gImg[r,c]
            while (1):
                flag = 0
                bImgT = bImg[r0-size:r0+size+1,c0-size:c0+size+1]
                gImgT = gImg[r0-size:r0+size+1,c0-size:c0+size+1]
                gImgT = gImgT*bImgT
                gImgT_max = numpy.max(gImgT)
                if (value0 < gImgT_max):
                    r1,c1 = numpy.where(gImgT==gImgT_max)
                    r1,c1 = r1[0],c1[0]
                    r0,c0,value0 = r0-size+r1,c0-size+c1,gImgT_max
                    flag = 1
                if (flag == 0):
                    labelImg[r,c] = connectedSeedLabel[seedList.index([r0,c0])]
                    break
    return labelImg,seedList
##############################################################
