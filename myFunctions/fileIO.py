import cv2
import numpy
import os
import imageio
import h5py
import imageProcess
#from libtiff import TIFF
from tqdm import tqdm
from os.path import join

##############################################################
# CREATE DIRECTORY
##############################################################
def mkdir(dirName):
    if (os.path.exists(dirName) == False):
        os.makedirs(dirName)
##############################################################


##############################################################
# CREATE DIRECTORY STRUCTURE USED FOR STORING SEGMENTATION IMAGES 
##############################################################
def mkdirs(dirName):
    mkdir(dirName)
    mkdir(dirName+'/dataProcessing')
    mkdir(dirName+'/dataProcessing/gImgRawStack')
    mkdir(dirName+'/dataProcessing/processedStack')
    mkdir(dirName+'/segmentation')
    mkdir(dirName+'/segmentation/result')
    mkdir(dirName+'/segmentation/tracking')
    return 0
##############################################################


##############################################################
# CREATE AND INITIALIZE THE H5 DATASET
##############################################################
def createH5(fileName):
    print "Creating the H5 dataset"
    fp = h5py.File(fileName, "w")
    
    #fp.create_group("/dataProcessing")
    fp.create_group("/dataProcessing/dm4RawStack")
    fp.create_group("/dataProcessing/gImgRawStack")
    fp.create_group("/dataProcessing/processedStack")
    
    #fp.create_group("/segmentation")
    fp.create_group("/segmentation/bImgStack")
    fp.create_group("/segmentation/labelStack")
    
    return fp
##############################################################


##############################################################
# READ A GREYSCALE/RGB AVI FILE
##############################################################
def readTIFF(fileName):
    tif = TIFF.open(fileName, mode='r')
    img = tif.read_image()
    return img
##############################################################


##############################################################
# READ A GREYSCALE/RGB AVI FILE
##############################################################
def readAVI(fileName):
    print "Reading avi file and creating volume stack"
    reader = imageio.get_reader(fileName)
    col, row = reader.get_meta_data()['size']
    numFrames = reader.get_meta_data()['nframes']
    
    gImgRawStack = numpy.zeros([row,col,numFrames],dtype='uint8')
    for i, img in enumerate(reader):
        gImgRawStack[:,:,i] = img[:,:,0]
    reader.close()
    return gImgRawStack,row,col,numFrames
##############################################################


##############################################################
# READ A GREYSCALE PNG IMAGE SEQUENCE
##############################################################
def readImageSequence(folder,frameList,zfillVal,extension='png'):
    print "Reading image sequence and creating volume stack"
    gImg = cv2.imread(folder+'/'+str(frameList[0]).zfill(zfillVal)+'.'+extension,0)
    row,col = gImg.shape
    numFrames = len(frameList)
    
    gImgRawStack = numpy.zeros([row,col,numFrames],dtype='uint8')
    for frame,i in tqdm(zip(frameList,range(numFrames))):
        gImg = cv2.imread(folder+'/'+str(frame).zfill(zfillVal)+'.'+extension,0)
        gImgRawStack[:,:,i] = gImg
    return gImgRawStack,row,col,numFrames
##############################################################


##############################################################
# SAVE A GREYSCALE PNG IMAGE SEQUENCE
##############################################################
def saveImageSequence(gImgStack,outputDir,normalize=False):
    print "Saving volume stack as image sequence"
    [row,col,numFrames] = gImgStack.shape
    for frame in tqdm(range(1,numFrames+1)):
        if (normalize==True):
            gImg = imageProcess.normalize(gImgStack[:,:,frame-1])
        else:
            gImg = gImgStack[:,:,frame-1]
        cv2.imwrite(outputDir+'/'+str(frame).zfill(6)+'.png',gImg)
    return 0
##############################################################


##############################################################
# DELETE A FILE
##############################################################
def delete(fileName):
    if (os.path.exists(fileName)):
        os.remove(fileName)
##############################################################


##############################################################
# WRITE A DATASET TO HDF5 FILE
##############################################################
def writeH5Dataset(fp,datasetName,dataset):
    flag = 0
    try:
        fp[datasetName]
    except:
        flag = 1
    if (flag==0):
        del fp[datasetName]
        fp.create_dataset(datasetName, data=dataset, compression='gzip', compression_opts=9)
    else:
        fp.create_dataset(datasetName, data=dataset, compression='gzip', compression_opts=9)
    return 0
##############################################################


##############################################################
# GET A LIST OF FILES IN A DIRECTORY (WORKS RECURSIVELY)
##############################################################
def getFilesInDirectory(dirName,format=''):
    fileList=[]
    for root, dirs, files in os.walk(dirName):
        for name in files:
            if name.endswith((format)):
                fileList.append(join(root,name))
    fileList = numpy.sort(fileList,kind='mergesort')
    return fileList
##############################################################
