import os, sys
import h5py
import cv2
import numpy
import gc
import mpi4py
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from skimage.draw import circle_perimeter,ellipse_perimeter
from skimage.morphology import disk
from skimage.filters.rank import median
from tqdm import tqdm
from scipy import ndimage
from mpi4py import MPI
from time import time

sys.path.append(os.path.abspath('../myFunctions'))
import fileIO
import imageProcess
import myCythonFunc
import dataViewer
import misc
import tracking
import fitFunc

#######################################################################
# USER INPUTS
#######################################################################
inputFile = ''
outputFile = r'F:\Data\Rotation\20170612125247-NRLD\20170612125247.h5'
inputDir = r'Y:\seewee\Shufen\14-09-22.893_Export\20170612125247'
outputDir = r'F:\Data\Rotation\20170612125247-NRLD\output'
pixInNM = 0.5772
fps = 300
microscope = 'JOEL2200' #'JOEL2010','T12','JOEL2200'
camera = 'DE16' #'Orius','One-view','DE16'
owner = 'See Wee'
zfillVal = 6
fontScale = 2
structure = [[1,1,1],[1,1,1],[1,1,1]]
scale=1
#######################################################################


#######################################################################
# INITIALIZATION FOR THE MPI ENVIRONMENT
#######################################################################
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
#######################################################################

if (rank==0):
    tic = time()
#######################################################################
# DATA PROCESSSING
# 1. READ THE INPUT FILES AND STORE THEM FRAME-WISE IN H5 FILE
# 2. PERFORM BACKGROUND SUBTRACTION (IF REQUIRED)
#######################################################################
#########
# PART 1
#########
#if (rank==0):
    #fp = fileIO.createH5(outputFile)
    
    #print "Reading raw images, writing to hdf5 file, and saving images"
    #fileIO.mkdirs(outputDir)
    #frameList = range(0,10059)
    #numFrames = len(frameList)#frameList[-1]
    #for frame in tqdm(frameList):
        #gImg = cv2.imread(inputDir+'/14-09-22.893_'+str(frame).zfill(0)+'.tif',0)
        #gImg = gImg[200:200+1800,70:70+1800]
        #fileIO.writeH5Dataset(fp,'/dataProcessing/gImgRawStack/'+str(frame+1).zfill(zfillVal),gImg)
        #cv2.imwrite(outputDir+'/dataProcessing/gImgRawStack/'+str(frame+1).zfill(zfillVal)+'.png',gImg)
        
    #[row,col] = gImg.shape
    #fp.attrs['inputFile'] = inputFile
    #fp.attrs['outputFile'] = outputFile
    #fp.attrs['inputDir'] = inputDir
    #fp.attrs['outputDir'] = outputDir
    #fp.attrs['pixInNM'] = pixInNM
    #fp.attrs['pixInAngstrom'] = pixInNM*10
    #fp.attrs['fps'] = fps
    #fp.attrs['microscope'] = microscope
    #fp.attrs['camera'] = camera
    #fp.attrs['owner'] = owner
    #fp.attrs['row'] = row
    #fp.attrs['col'] = col
    #fp.attrs['numFrames'] = numFrames
    ##fp.attrs['frameList'] = range(1,numFrames+1)
    #fp.attrs['zfillVal'] = zfillVal
    
    #fp.flush(), fp.close()
    #gc.collect()
#comm.Barrier()

########
# PART 2
#########
#if (rank==0):
    #print "Inverting the image and performing background subtraction"
#invertFlag=True
#bgSubFlag=False; bgSubSigmaTHT=2; radiusTHT=15
#blurFlag=True

#if (rank==0):
    #fp = h5py.File(outputFile, 'r+')
#else:
    #fp = h5py.File(outputFile, 'r')
#[row,col,numFrames,frameList] = misc.getVitals(fp)
#procFrameList = numpy.array_split(frameList,size)

#for frame in tqdm(procFrameList[rank]):
    #gImgProc = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
    #if (blurFlag==True):
        #gImgProc = median(gImgProc,disk(9))
    #if (invertFlag==True):
        #gImgProc = imageProcess.invertImage(gImgProc)
    #if (bgSubFlag==True):
        #gImgProc = imageProcess.subtractBackground(gImgProc, sigma=bgSubSigmaTHT, radius=radiusTHT)
    #gImgProc = imageProcess.normalize(gImgProc)
    #cv2.imwrite(outputDir+'/dataProcessing/processedStack/'+str(frame).zfill(zfillVal)+'.png',gImgProc)

#comm.Barrier()
    
#if (rank==0):
    #print "Writing processed image to h5 dataset"
    #for frame in tqdm(frameList):
        #gImgProc = cv2.imread(outputDir+'/dataProcessing/processedStack/'+str(frame).zfill(zfillVal)+'.png',0)
        #fileIO.writeH5Dataset(fp,'/dataProcessing/processedStack/'+str(frame).zfill(zfillVal),gImgProc)
        
#fp.flush(), fp.close()
#comm.Barrier()
#####################################################################


#######################################################################
# IMAGE SEGMENTATION
#######################################################################
#if (rank==0):
    #print "Performing segmentation for all the frames"
    
#fp = h5py.File(outputFile, 'r')
#[row,col,numFrames,frameList] = misc.getVitals(fp)
#procFrameList = numpy.array_split(frameList,size)
#cropSize = 80
#areaRange = numpy.array([500,6000], dtype='float64')
#outFile = open(outputDir+'/segmentation_'+str(rank)+'.dat','wb')

#for frame in tqdm(procFrameList[rank]):
    #gImgRaw = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
    #gImgNorm = imageProcess.normalize(gImgRaw,min=0,max=230)
    #gImgProc = fp['/dataProcessing/processedStack/'+str(frame).zfill(zfillVal)].value
    #bImgBdry = gImgRaw.copy(); bImgBdry[:] = 0
    
    #for threshold in numpy.arange(0.20,0.10,-0.01):
        #blobs_log = blob_log(gImgProc, min_sigma=20, max_sigma=20, num_sigma=1, threshold=threshold)
        #flag=False
        #if (blobs_log.size>0):
            #blobs_log[:,2] = blobs_log[:,2]*numpy.sqrt(2)
            #counter = 0; particleDetails = []
            #for r,c,rad in blobs_log:
                #rr,cc = circle_perimeter(int(r),int(c),int(rad))
                #if ((rr<0).any()==True or (cc<0).any()==True):
                    #pass
                #elif ((rr>row-1).any()==True or (cc>col-1).any()==True):
                    #pass
                #elif (r-cropSize<0 or r+cropSize>=row):
                    #pass
                #elif (c-cropSize<0 or c+cropSize>=col):
                    #pass
                #else:
                    #counter+=1
                    #particleDetails.append([counter,r,c,rad])
                    
            #if (len(particleDetails)>0 and len(particleDetails)<3):
                #for counter,r,c,rad in particleDetails:
                    #rCenter,cCenter,theta = r,c,1
                    #gImgCrop = imageProcess.normalize(gImgProc[int(rCenter)-cropSize:int(rCenter)+cropSize+1,int(cCenter)-cropSize:int(cCenter)+cropSize+1])
                    
                    #bImgCrop1,flag1,err1 = fitFunc.fitting(gImgCrop,theta)
                    #bImgCrop2,flag2,err2 = fitFunc.fitting(gImgCrop,-theta)
                    
                    #if (flag1==True and flag2==True):
                        ##err1 = numpy.sum(numpy.abs(gImgCrop-imageProcess.normalize(bImgCrop1)))
                        ##err2 = numpy.sum(numpy.abs(gImgCrop-imageProcess.normalize(bImgCrop2)))
                        #if (err1<err2):
                            #bImgCrop,flag = bImgCrop1,flag1
                        #else:
                            #bImgCrop,flag = bImgCrop2,flag2
                    #elif (flag1==True and flag2==False):
                        #bImgCrop,flag = bImgCrop1,flag1
                    #elif (flag1==False and flag2==True):
                        #bImgCrop,flag = bImgCrop2,flag2
                    #elif (flag1==False and flag2==False):
                        #bImgCrop = bImgCrop1; bImgCrop[:]=False
                        
                    #if (bImgCrop.max()==False):
                        #flag=False
                        
                    #if (flag == True):
                        #bImgCropBdry = imageProcess.normalize(imageProcess.boundary(bImgCrop))
                        #bImgBdry[int(rCenter)-cropSize:int(rCenter)+cropSize+1,int(cCenter)-cropSize:int(cCenter)+cropSize+1] = numpy.maximum(bImgBdry[int(rCenter)-cropSize:int(rCenter)+cropSize+1,int(cCenter)-cropSize:int(cCenter)+cropSize+1],bImgCropBdry)
                        
            #if (flag==True):
                #bImg = imageProcess.fillHoles(bImgBdry)
                #bImg = myCythonFunc.areaThreshold(bImg.astype('uint8'), areaRange=areaRange)
                #bImgBdry = imageProcess.normalize(imageProcess.boundary(bImg))
                #label, numLabel, dictionary = imageProcess.regionProps(bImg, gImgRaw, structure=[[1,1,1],[1,1,1],[1,1,1]], centroid=True, area=True, effRadius=True)
                #if (numLabel>0):
                    #outFile.write("%d %f %f %f %f\n" %(frame, dictionary['centroid'][0][0], dictionary['centroid'][0][1], dictionary['effRadius'][0], dictionary['area'][0]*pixInNM*pixInNM))
                #break
                
    #finalImg = numpy.column_stack((numpy.maximum(gImgNorm,bImgBdry), gImgProc))
    #cv2.imwrite(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.png', finalImg)
#outFile.close()
#fp.flush(), fp.close()
#comm.Barrier()

#if (rank==0):
    #for r in range(size):
        #if (r==0):
            #measures = numpy.loadtxt(outputDir+'/segmentation_'+str(r)+'.dat')
        #else:
            #measures = numpy.row_stack((measures,numpy.loadtxt(outputDir+'/segmentation_'+str(r)+'.dat')))
        #fileIO.delete(outputDir+'/segmentation_'+str(r)+'.dat')
    #numpy.savetxt(outputDir+'/segmentation.dat', measures, fmt='%.6f')
#comm.Barrier()

#######################################################################


#######################################################################
# CREATE BINARY IMAGES INTO HDF5 FILE
#######################################################################
if (rank==0):
    print "Creating binary images from segmented images"
    
if (rank==0):
    fp = h5py.File(outputFile, 'r+')
else:
    fp = h5py.File(outputFile, 'r')
[row,col,numFrames,frameList] = misc.getVitals(fp)
procFrameList = numpy.array_split(frameList,size)

for frame in tqdm(procFrameList[rank]):
    bImg = cv2.imread(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.png',0)[0:row,0:col]
    bImg = bImg==255
    bImg = imageProcess.fillHoles(bImg)
    numpy.save(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.npy', bImg)
comm.barrier()

if (rank==0):
    print "Saving the binary stack to h5 dataset"
    for frame in tqdm(frameList):
        bImg = numpy.load(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.npy')
        fileIO.writeH5Dataset(fp,'/segmentation/bImgStack/'+str(frame).zfill(zfillVal),bImg)
        fileIO.delete(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.npy')
        
fp.flush(), fp.close()
comm.Barrier()
#######################################################################


#######################################################################
# LABELLING PARTICLES
#######################################################################
centerDispRange = [200,200]
perAreaChangeRange = [500,600]
missFramesTh = 100
    
if (rank==0):
    print "Labelling segmented particles"
    fp = h5py.File(outputFile, 'r+')
    maxID, occurenceFrameList = tracking.labelParticlesFromText(fp, centerDispRange=centerDispRange, perAreaChangeRange=perAreaChangeRange, missFramesTh=missFramesTh,scale=scale)
    fp.attrs['particleList'] = range(1,maxID+1)
    numpy.savetxt(outputDir+'/frameOccurenceList.dat',numpy.column_stack((fp.attrs['particleList'],occurenceFrameList)),fmt='%d')
    fp.flush(), fp.close()
comm.Barrier()

if (rank==0):
    print "Generating images with labelled particles"
fp = h5py.File(outputFile, 'r')
tracking.generateLabelImages(fp,outputDir+'/segmentation/tracking',fontScale,size,rank)
fp.flush(), fp.close()
comm.Barrier()
#######################################################################


#######################################################################
# REMOVING UNWANTED PARTICLES
#######################################################################
#keepList = range(1,5)+range(6,57)
#removeList = []

#if (rank==0):
	#print "Removing unwanted particles"

#if (rank==0):
    #fp = h5py.File(outputFile, 'r+')
#else:
    #fp = h5py.File(outputFile, 'r')
#particleList = fp.attrs['particleList']
#if not removeList:
	#removeList = [s for s in particleList if s not in keepList]
#tracking.removeParticles(fp,removeList,comm,size,rank)
#fp.flush(), fp.close()    
#comm.Barrier()
#######################################################################


#######################################################################
# GLOBAL RELABELING OF PARTICLES
#######################################################################
#correctionList = [[7,8,71]]

#if (rank==0):
	#print "Global relabeling of  particles"
    
#if (rank==0):
    #fp = h5py.File(outputFile, 'r+')
#else:
    #fp = h5py.File(outputFile, 'r')
#tracking.globalRelabelParticles(fp,correctionList,comm,size,rank)
#fp.flush(), fp.close()    
#comm.Barrier()
#######################################################################


#######################################################################
# FRAME-WISE CORRECTION OF PARTICLE LABELS
#######################################################################
#frameWiseCorrectionList = [\
#[range(2,5),[18,77]],\
#[range(1,10),[1,0]]\
#]

#if (rank==0):
    #print "Frame-wise relabeling of  particles"
    
#if (rank==0):
    #fp = h5py.File(outputFile, 'r+')
#else:
    #fp = h5py.File(outputFile, 'r')
#tracking.framewiseRelabelParticles(fp,frameWiseCorrectionList,comm,size,rank)
#fp.flush(), fp.close()    
#comm.Barrier()
#######################################################################


#######################################################################
# RELABEL PARTICLES IN THE ORDER OF OCCURENCE
#######################################################################
#if (rank==0):
    #fp = h5py.File(outputFile, 'r+')
#else:
    #fp = h5py.File(outputFile, 'r')
#tracking.relabelParticles(fp,comm,size,rank)
#fp.flush(), fp.close()    
#comm.Barrier()
#######################################################################


#######################################################################
# GENERATING IMAGES WITH LABELLED PARTICLES
#######################################################################
#if (rank==0):
    #print "Generating images with labelled particles"
#fp = h5py.File(outputFile, 'r')
#tracking.generateImages(fp,outputDir+'/segmentation/tracking',fontScale,size,rank)
#fp.flush(), fp.close()
#comm.Barrier()
#######################################################################


#######################################################################
# FINDING OUT THE MEASURES FOR TRACKED PARTICLES
#######################################################################
#if (rank==0):
	#print "Finding measures for tracked particles"

#fp = h5py.File(outputFile, 'r')
#[row,col,numFrames,frameList] = misc.getVitals(fp)
#particleList = fp.attrs['particleList']
#zfillVal = fp.attrs['zfillVal']
#procFrameList = numpy.array_split(frameList,size)
#fps = fp.attrs['fps']
#pixInNM = fp.attrs['pixInNM']

#outFile = open(str(rank)+'.dat','wb')

#area=True
#perimeter=True
#circularity=True
#pixelList=False
#bdryPixelList=False
#centroid=True
#intensityList=False
#sumIntensity=False
#effRadius=True
#radius=False
#circumRadius=False
#inRadius=False
#radiusOFgyration=False
#orientation = True

#for frame in procFrameList[rank]:
    #labelImg = fp['/segmentation/labelStack/'+str(frame).zfill(zfillVal)].value
    #gImgRaw = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
    #outFile.write("%f " %(1.0*frame/fps))
    #for particle in particleList:
        #bImg = labelImg==particle
        #if (bImg.max() == True):
            #label, numLabel, dictionary = imageProcess.regionProps(bImg, gImgRaw, structure=structure, centroid=True, area=True, perimeter=True, orientation=True, effRadius=True)
            #outFile.write("%f %f %f %f %f %f " %(dictionary['centroid'][0][1]*pixInNM, (row-dictionary['centroid'][0][0])*pixInNM, dictionary['area'][0]*pixInNM*pixInNM, dictionary['perimeter'][0]*pixInNM, dictionary['orientation'][0], dictionary['effRadius'][0]*pixInNM))
        #else:
            #outFile.write("nan nan nan nan nan nan ")
    #outFile.write("\n")
#outFile.close()
#fp.flush(), fp.close()
#comm.Barrier()

#if (rank==0):
    #for r in range(size):
        #if (r==0):
            #measures = numpy.loadtxt(str(r)+'.dat')
        #else:
            #measures = numpy.row_stack((measures,numpy.loadtxt(str(r)+'.dat')))
        #fileIO.delete(str(r)+'.dat')
    #measures = measures[numpy.argsort(measures[:,0])]
    #numpy.savetxt(outputDir+'/segmentation/imgDataNM.dat', measures, fmt='%.6f')
#######################################################################

if (rank==0):
    toc = time()
    print toc-tic, (toc-tic)/numFrames
