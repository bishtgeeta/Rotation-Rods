import cv2
import numpy
import h5py
import sys
import imageProcess
import fileIO
import misc
import matplotlib.pyplot as plt
from skimage import measure
from skimage.draw import circle,circle_perimeter
from tqdm import tqdm
from scipy import ndimage

#######################################################################
# LABELING PARTICLES
#######################################################################
def labelParticlesFromBImg(fp, centerDispRange=[5,5], perAreaChangeRange=[20,20], missFramesTh=10, structure=[[1,1,1],[1,1,1],[1,1,1]]):
    
    [row,col,numFrames] = fp.attrs['row'],fp.attrs['col'],fp.attrs['numFrames']
    frameList = fp.attrs['frameList']
    zfillVal = fp.attrs['zfillVal']
    
    labelStack = numpy.zeros([row,col,numFrames], dtype='uint32')
    for frame in frameList:
        str1 = str(frame)+'/'+str(frameList[-1]); str2 = '\r'+' '*len(str1)+'\r'
        sys.stdout.write(str1)
        bImg = fp['/segmentation/bImgStack/'+str(frame).zfill(zfillVal)].value
        gImg = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value

        if (frame==frameList[0]):
            labelImg_0, numLabel_0, dictionary_0 = imageProcess.regionProps(bImg, gImg, structure=structure, centroid=True, area=True)
            maxID = numLabel_0
            occurenceFrameList = [frame]*maxID
            dictionary_0['frame'] = []
            for i in range(len(dictionary_0['id'])):
                dictionary_0['frame'].append(frame)
            labelStack[:,:,frame-1] = labelImg_0
        else:
            labelImg_1, numLabel_1, dictionary_1 = imageProcess.regionProps(bImg, gImg, structure=structure, centroid=True, area=True)
            if (numLabel_1>0):
                areaMin = min(dictionary_1['area']); areaMax = max(dictionary_1['area'])
            for i in range(len(dictionary_1['id'])):
                flag = 0
                bImg_1_LabelN = labelImg_1==dictionary_1['id'][i]
                center_1 = dictionary_1['centroid'][i]
                area_1 = dictionary_1['area'][i]
                frame_1 = frame
                if (areaMax-areaMin>0):
                    factor = 1.0*(area_1-areaMin)/(areaMax-areaMin)
                    perAreaChangeTh = perAreaChangeRange[1] - factor*(perAreaChangeRange[1]-perAreaChangeRange[0])
                    centerDispTh = centerDispRange[1] - factor*(centerDispRange[1]-centerDispRange[0])
                else:
                    perAreaChangeTh = perAreaChangeRange[1]
                    centerDispTh = centerDispRange[1]
                closeness,J = 1e10,0
                for j in range(len(dictionary_0['id'])-1,-1,-1):
                    center_0 = dictionary_0['centroid'][j]
                    area_0 = dictionary_0['area'][j]
                    frame_0 = dictionary_0['frame'][j]
                    centerDisp = numpy.sqrt((center_1[0]-center_0[0])**2 + (center_1[1]-center_0[1])**2)
                    perAreaChange = 100.0*numpy.abs(area_1-area_0)/numpy.maximum(area_1,area_0)
                    missFrames = frame_1-frame_0
                    if (centerDisp <= centerDispTh):
                        if (perAreaChange <= perAreaChangeTh):
                            if (missFrames <= missFramesTh):
                                if (centerDisp < closeness):
                                    closeness = centerDisp
                                    J = j
                                    flag = 1
                                    
                if (flag == 1):
                    labelStack[:,:,frame-1] += (bImg_1_LabelN*dictionary_0['id'][J]).astype('uint32')
                    dictionary_0['centroid'][J] = center_1
                    dictionary_0['area'][J] = area_1
                    dictionary_0['frame'][J] = frame
                if (flag == 0):
                    maxID += 1
                    occurenceFrameList.append(frame)
                    labelN_1 = bImg_1_LabelN*maxID
                    labelStack[:,:,frame-1] += labelN_1.astype('uint32')
                    dictionary_0['id'].append(maxID)
                    dictionary_0['centroid'].append(center_1)
                    dictionary_0['area'].append(area_1)
                    dictionary_0['frame'].append(frame)
        sys.stdout.flush(); sys.stdout.write(str2)
    sys.stdout.flush()

    #if (labelStack.max() < 256):
        #labelStack = labelStack.astype('uint8')
    #elif (labelStack.max()<65536):
        #labelStack = labelStack.astype('uint16')
        
    print "Checking for multiple particles in a single frame"
    for frame in frameList:
        str1 = str(frame)+'/'+str(frameList[-1]); str2 = '\r'+' '*len(str1)+'\r'
        sys.stdout.write(str1)
        labelImg = labelStack[:,:,frame-1]
        numLabel = imageProcess.regionProps(labelImg.astype('bool'), gImg, structure=structure)[1]
        if (numLabel != numpy.size(numpy.unique(labelImg)[1:])):
            for N in numpy.unique(labelImg)[1:]:
                labelImgN = labelImg==N
                numLabel = imageProcess.regionProps(labelImgN, gImg, structure=structure)[1]
                if (numLabel>1):
                    labelImg[labelImg==N] = 0
                    labelStack[:,:,frame-1] = labelImg
        sys.stdout.flush(); sys.stdout.write(str2)
    sys.stdout.flush()
    
    print "Saving the labelled particles in h5 file"
    for frame in frameList:
        fileIO.writeH5Dataset(fp,'/segmentation/labelStack/'+str(frame).zfill(zfillVal),labelStack[:,:,frame-1])
    del labelStack
    return maxID, occurenceFrameList
#######################################################################


#######################################################################
# LABELING PARTICLES
#######################################################################
def labelParticlesFromText(fp, centerDispRange=[5,5], perAreaChangeRange=[20,20], missFramesTh=10, scale=1):
    [row,col,numFrames] = fp.attrs['row'],fp.attrs['col'],fp.attrs['numFrames']
    frameList = range(1,numFrames+1)
    zfillVal = fp.attrs['zfillVal']
    segmentation = numpy.loadtxt(fp.attrs['outputDir']+'/segmentation.dat')
    dictionary_0,dictionary_1,particleLabel = {},{},[]
    #labelStack = numpy.zeros([row*scale,col*scale,numFrames], dtype='uint32')
    
    for frame in tqdm(frameList):
        data = segmentation[segmentation[:,0]==frame]
        if (frame==frameList[0]):
            maxID = int(data[-1,1])
            occurenceFrameList = [frame]*maxID
            dictionary_0['id'] = []
            dictionary_0['frame'] = []
            dictionary_0['centroid'] = []
            dictionary_0['area'] = []
            dictionary_0['radius'] = []
            for particle,r,c,rad,area in data[:,1:]:
                dictionary_0['id'].append(particle)
                dictionary_0['frame'].append(frame)
                dictionary_0['centroid'].append([r,c])
                dictionary_0['radius'].append(rad)
                dictionary_0['area'].append(area)
                particleLabel.append(particle)
        else:
            if (data.size>0):
                areaMin,areaMax = data[:,5].min(),data[:,5].max()
            for particle,r,c,rad,area in data[:,1:]:
                flag=0
                center_1 = [r,c]
                area_1 = area
                radius_1 = rad
                frame_1 = frame
                if (areaMax-areaMin > 0):
                    factor = 1.0*(area_1-areaMin)/(areaMax-areaMin)
                    perAreaChangeTh = perAreaChangeRange[1] - factor*(perAreaChangeRange[1]-perAreaChangeRange[0])
                    centerDispTh = centerDispRange[1] - factor*(centerDispRange[1]-centerDispRange[0])
                else:
                    perAreaChangeTh = perAreaChangeRange[1]
                    centerDispTh = centerDispRange[1]
                closeness,J = 1e10,0
                for j in range(len(dictionary_0['id'])-1,-1,-1):
                    frame_0 = dictionary_0['frame'][j]
                    center_0 = dictionary_0['centroid'][j]
                    radius_0 = dictionary_0['radius'][j]
                    area_0 = dictionary_0['area'][j]
                    centerDisp = numpy.sqrt((center_1[0]-center_0[0])**2 + (center_1[1]-center_0[1])**2)
                    perAreaChange = 100.0*numpy.abs(area_1-area_0)/numpy.maximum(area_1,area_0)
                    missFrames = frame_1-frame_0
                    if (centerDisp <= centerDispTh):
                        if (perAreaChange <= perAreaChangeTh):
                            if (missFrames <= missFramesTh):
                                if (centerDisp < closeness):
                                    closeness = centerDisp
                                    J = j
                                    flag = 1
                                    
                if (flag==1):
                    dictionary_0['frame'][J] = frame
                    dictionary_0['centroid'][J] = center_1
                    dictionary_0['radius'][J] = radius_1
                    dictionary_0['area'][J] = area_1
                    particleLabel.append(dictionary_0['id'][J])
                if (flag==0):
                    maxID+=1
                    occurenceFrameList.append(frame)
                    dictionary_0['id'].append(maxID)
                    dictionary_0['frame'].append(frame)
                    dictionary_0['centroid'].append(center_1)
                    dictionary_0['radius'].append(radius_1)
                    dictionary_0['area'].append(area_1)
                    particleLabel.append(maxID)
    labels = numpy.column_stack((segmentation,particleLabel))
    numRows,numCols = labels.shape
    flagArr = numpy.zeros(numRows,dtype='bool')
    
    print "Checking for multiple particles in a single frame"
    for frame,particle,r,c,rad,area,label,nRow in tqdm(numpy.column_stack((labels,range(numRows)))):
        for n in range(int(nRow)+1,numRows):
            if (frame != labels[n,0]):
                break
            elif (label == labels[n,6]):
                flagArr[[n,int(nRow)]] = True
    outFile = open(fp.attrs['outputDir']+'/tracking.dat','wb')
    
    print "Writing the labeled imaged to text file"
    for frame,particle,r,c,rad,area,label,nRow in tqdm(numpy.column_stack((labels,range(numRows)))):
        if (flagArr[int(nRow)] == True):
            labels[int(nRow),6] = 0
            outFile.write("%d %d %d %d %f %f 0\n" %(frame,particle,r,c,rad,area))
        else:
            #rr,cc = circle(int(r*scale),int(c*scale),int(rad*scale))
            #if ((rr<0).any()==True or (cc<0).any()==True):
                #pass
            #elif ((rr>row*scale-1).any()==True or (cc>col*scale-1).any()==True):
                #pass
            #else:
                #labelStack[rr,cc,int(frame)-1] = label
            outFile.write("%d %d %d %d %f %f %d\n" %(frame,particle,r,c,rad,area,label))
    outFile.close()
    
    #for frame in tqdm(frameList):
        #data = 
        #fileIO.writeH5Dataset(fp,'/segmentation/labelStack/'+str(frame).zfill(zfillVal),labelStack[:,:,frame-1])
    #del labelStack
    #return maxID, occurenceFrameList
    
    print "Saving the labelled particles in h5 file"
    DATA = numpy.loadtxt(fp.attrs['outputDir']+'/tracking.dat')
    labelStack = numpy.zeros([row*scale,col*scale], dtype='uint32')
    for frame in tqdm(frameList):
        data = DATA[DATA[:,0]==frame]
        bImg = fp['/segmentation/bImgStack/'+str(frame).zfill(zfillVal)].value
        [labelImg, numLabel] = ndimage.label(bImg, structure=[[1,1,1],[1,1,1],[1,1,1]])
        for f,particle,r,c,rad,area,label in data:
            bImgLabelN = labelImg==particle
            labelStack = labelStack+bImgLabelN*label
        fileIO.writeH5Dataset(fp,'/segmentation/labelStack/'+str(frame).zfill(zfillVal),labelStack)
        labelStack[:] = 0
    return maxID, occurenceFrameList
#######################################################################


#######################################################################
# REMOVE UNWANTED PARTICLES AFTER TRACKING
#######################################################################
def removeParticles(fp,removeList,comm,size,rank):
    [row,col,numFrames,frameList] = misc.getVitals(fp)
    particleList = fp.attrs['particleList']
    zfillVal = fp.attrs['zfillVal']
    procFrameList = numpy.array_split(frameList,size)
    for frame in procFrameList[rank]:
        labelImg = fp['/segmentation/labelStack/'+str(frame).zfill(zfillVal)].value
        for r in removeList:
            labelImg[labelImg==r] = 0
        numpy.save(str(frame).zfill(zfillVal)+'.npy', labelImg)
    comm.Barrier()
        
    if (rank==0):
        for frame in frameList:
            labelImg = numpy.load(str(frame).zfill(zfillVal)+'.npy')
            fileIO.writeH5Dataset(fp,'/segmentation/labelStack/'+str(frame).zfill(zfillVal), labelImg)
            fileIO.delete(str(frame).zfill(zfillVal)+'.npy')
        for r in removeList:
            fp.attrs['particleList'] = numpy.delete(fp.attrs['particleList'], numpy.where(fp.attrs['particleList']==r))
    comm.Barrier()
    return 0
#######################################################################


#######################################################################
# REMOVE UNWANTED PARTICLES AFTER TRACKING
#######################################################################
def removeParticlesText(fp,removeList):
    [row,col,numFrames,frameList] = misc.getVitals(fp)
    trackingData = numpy.loadtxt(fp.attrs['outputDir']+'/tracking.dat')
    outFile = open(fp.attrs['outputDir']+'/tracking.dat','wb')
    for frame,particle,r,c,rad,area,label in trackingData:
        if (int(label) in removeList):
            pass
        elif (int(label)==0):
            pass
        else:
            outFile.write("%d %d %d %d %f %f %d\n" %(frame,particle,r,c,rad,area,label))
    outFile.close()
    trackingData = numpy.loadtxt(fp.attrs['outputDir']+'/tracking.dat')
    particleList = numpy.unique(trackingData[:,-1])
    fp.attrs['particleList'] = particleList.astype('int')
#######################################################################


#######################################################################
# RELABEL THE PARTICLES WITH WRONG LABELS
#######################################################################
def globalRelabelParticles(fp,correctionList,comm,size,rank):
    [row,col,numFrames,frameList] = misc.getVitals(fp)
    particleList = fp.attrs['particleList']
    zfillVal = fp.attrs['zfillVal']
    procFrameList = numpy.array_split(frameList,size)
    for frame in procFrameList[rank]:
        labelImg = fp['/segmentation/labelStack/'+str(frame).zfill(zfillVal)].value
        for i in range(len(correctionList)):
			for j in range(len(correctionList[i])-1):
				labelImg[labelImg==correctionList[i][j]] = correctionList[i][-1]
        numpy.save(str(frame).zfill(zfillVal)+'.npy', labelImg)
    comm.Barrier()
    
    if (rank==0):
        for frame in frameList:
            labelImg = numpy.load(str(frame).zfill(zfillVal)+'.npy')
            fileIO.writeH5Dataset(fp,'/segmentation/labelStack/'+str(frame).zfill(zfillVal), labelImg)
            fileIO.delete(str(frame).zfill(zfillVal)+'.npy')
            particleInFrame = numpy.unique(labelImg)[1:]
            if (frame==frameList[0]):
                particleList = particleInFrame.copy()
            else:
                particleList = numpy.unique(numpy.append(particleList,particleInFrame))
        fp.attrs['particleList'] = particleList
    comm.Barrier()
    return 0
#######################################################################


#######################################################################
# RELABEL THE PARTICLES WITH WRONG LABELS
#######################################################################
def globalRelabelParticlesText(fp,correctionList):
    [row,col,numFrames,frameList] = misc.getVitals(fp)
    trackingData = numpy.loadtxt(fp.attrs['outputDir']+'/tracking.dat')
    outFile = open(fp.attrs['outputDir']+'/tracking.dat','wb')
    for frame,particle,r,c,rad,area,label in tqdm(trackingData):
        flag = 0
        if (label==0):
            flag=1
        for i in range(len(correctionList)):
            for j in range(len(correctionList[i])-1):
                if (label==correctionList[i][j]):
                    outFile.write("%d %d %d %d %f %f %d\n" %(frame,particle,r,c,rad,area,correctionList[i][-1]))
                    flag = 1
        if (flag==0):
            outFile.write("%d %d %d %d %f %f %d\n" %(frame,particle,r,c,rad,area,label))
    outFile.close()
    trackingData = numpy.loadtxt(fp.attrs['outputDir']+'/tracking.dat')
    particleList = numpy.unique(trackingData[:,-1])
    fp.attrs['particleList'] = particleList.astype('int')
#######################################################################    


#######################################################################
# FRAME-WISE RELABELING OF PARTICLES
#######################################################################
def framewiseRelabelParticles(fp,frameWiseCorrectionList,comm,size,rank):
    [row,col,numFrames,frameList] = misc.getVitals(fp)
    particleList = fp.attrs['particleList']
    zfillVal = fp.attrs['zfillVal']
    procFrameList = numpy.array_split(frameList,size)
    
    for frame in procFrameList[rank]:
        labelImg = fp['/segmentation/labelStack/'+str(frame).zfill(zfillVal)].value
        for frameWiseCorrection in frameWiseCorrectionList:
            subFrameList, subCorrectionList = frameWiseCorrection[0], frameWiseCorrection[1]
            for subFrame in subFrameList:
                if (frame==subFrame):
                    newLabel = subCorrectionList[-1]
                    for oldLabel in subCorrectionList[:-1]:
                        labelImg[labelImg==oldLabel] = newLabel
        numpy.save(str(frame).zfill(zfillVal)+'.npy', labelImg)
    comm.Barrier()
    
    if (rank==0):
        for frame in frameList:
            labelImg = numpy.load(str(frame).zfill(zfillVal)+'.npy')
            fileIO.writeH5Dataset(fp,'/segmentation/labelStack/'+str(frame).zfill(zfillVal), labelImg)
            fileIO.delete(str(frame).zfill(zfillVal)+'.npy')
            particleInFrame = numpy.unique(labelImg)[1:]
            if (frame==frameList[0]):
                particleList = particleInFrame.copy()
            else:
                particleList = numpy.unique(numpy.append(particleList,particleInFrame))
        fp.attrs['particleList'] = particleList
    comm.Barrier()
    return 0
#######################################################################


#######################################################################
# FRAME-WISE RELABELING OF PARTICLES TEXT
#######################################################################
def framewiseRelabelParticlesText(fp,frameWiseCorrectionList):
    trackingData = numpy.loadtxt(fp.attrs['outputDir']+'/tracking.dat')
    outFile = open(fp.attrs['outputDir']+'/tracking.dat','wb')
    
    for frame,particle,r,c,rad,area,label in trackingData:
        flag = 0
        if (label==0):
            flag=1
        for i in range(len(frameWiseCorrectionList)):
            if (int(frame) in frameWiseCorrectionList[i][0]):
                for j in range(len(frameWiseCorrectionList[i][1])-1):
                    if (int(label)==frameWiseCorrectionList[i][1][j]):
                        outFile.write("%d %d %d %d %f %f %d\n" %(frame,particle,r,c,rad,area,frameWiseCorrectionList[i][1][-1]))
                        flag = 1
        if (flag==0):
            outFile.write("%d %d %d %d %f %f %d\n" %(frame,particle,r,c,rad,area,label))
    outFile.close()
    trackingData = numpy.loadtxt(fp.attrs['outputDir']+'/tracking.dat')
    particleList = numpy.unique(trackingData[:,-1])
    fp.attrs['particleList'] = particleList.astype('int')
#######################################################################
    

#######################################################################
# RELABELING OF PARTICLES IN ORDER OF OCCURENCE
#######################################################################
def relabelParticles(fp,comm,size,rank):
    [row,col,numFrames,frameList] = misc.getVitals(fp)
    particleList = fp.attrs['particleList']
    zfillVal = fp.attrs['zfillVal']
    procFrameList = numpy.array_split(frameList,size)
    
    maxLabel = numpy.max(particleList)+1
    counter = 1
    
    newLabels = {}
    for particle in particleList:
        newLabels[particle]=[]
        
    for frame in frameList:
        particlesInFrame = numpy.unique(fp['/segmentation/labelStack/'+str(frame).zfill(zfillVal)].value)[1:]
        for p in particlesInFrame:
            if not newLabels[p]:
                newLabels[p] = [maxLabel, counter]
                maxLabel+=1
                counter+=1
                
    for frame in procFrameList[rank]:
        labelImg = fp['/segmentation/labelStack/'+str(frame).zfill(zfillVal)].value
        for key in newLabels.keys():
            labelImg[labelImg==key] = newLabels[key][0]
        for key in newLabels.keys():
            labelImg[labelImg==newLabels[key][0]] = newLabels[key][1]
        numpy.save(str(frame).zfill(zfillVal)+'.npy', labelImg)
    comm.Barrier()
    
    if (rank==0):
        for frame in frameList:
            labelImg = numpy.load(str(frame).zfill(zfillVal)+'.npy')
            if (counter < 256):
                labelImg = labelImg.astype('uint8')
            elif (counter < 65536):
                labelImg = labelImg.astype('uint16')
            else:
                labelImg = labelImg.astype('uint32')
            fileIO.writeH5Dataset(fp,'/segmentation/labelStack/'+str(frame).zfill(zfillVal), labelImg)
            fileIO.delete(str(frame).zfill(zfillVal)+'.npy')
            particleInFrame = numpy.unique(labelImg)[1:]
            if (frame==frameList[0]):
                particleList = particleInFrame.copy()
            else:
                particleList = numpy.unique(numpy.append(particleList,particleInFrame))
        fp.attrs['particleList'] = particleList
    comm.Barrier()
    return 0
#######################################################################


#######################################################################
# RELABELING OF PARTICLES IN ORDER OF OCCURENCE TEXT FILE
#######################################################################
def relabelParticlesText(fp):
    [row,col,numFrames,frameList] = misc.getVitals(fp)
    trackingData = numpy.loadtxt(fp.attrs['outputDir']+'/tracking.dat')
    particleList = fp.attrs['particleList']
    
    maxLabel = numpy.max(particleList)+1
    counter = 1
    
    newLabels = {}
    for particle in particleList:
        newLabels[particle]=[]
        
    for frame in frameList:
        trackingFrame = trackingData[trackingData[:,0]==frame]
        particlesInFrame = numpy.unique(trackingFrame[:,-1])
        for p in particlesInFrame:
            if not newLabels[p]:
                newLabels[p] = [maxLabel, counter]
                maxLabel+=1
                counter+=1
                
    outFile = open(fp.attrs['outputDir']+'/tracking.dat','wb')
    for frame,particle,r,c,rad,area,label in trackingData:
        outFile.write("%d %d %d %d %f %f %d\n" %(frame,particle,r,c,rad,area,newLabels[label][1]))
    outFile.close()
    trackingData = numpy.loadtxt(fp.attrs['outputDir']+'/tracking.dat')
    particleList = numpy.unique(trackingData[:,-1])
    fp.attrs['particleList'] = particleList.astype('int')
    return 0
#######################################################################


#######################################################################
# GENERATE LABELLED IMAGES WITH LABEL TAGS ON BINARY IMAGE
#######################################################################
def generateLabelImages(fp,imgDir,fontScale=1,size=1,rank=0,structure=[[1,1,1],[1,1,1],[1,1,1]]):
    [row,col,numFrames,frameList] = misc.getVitals(fp)
    particleList = fp.attrs['particleList']
    zfillVal = fp.attrs['zfillVal']
    procFrameList = numpy.array_split(frameList,size)
    for frame in tqdm(procFrameList[rank]):
        labelImg = fp['/segmentation/labelStack/'+str(frame).zfill(zfillVal)].value
        gImg = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
        bImg = labelImg.astype('bool')
        bImgBdry = imageProcess.normalize(imageProcess.boundary(bImg))
        label, numLabel, dictionary = imageProcess.regionProps(bImg, gImg, structure=structure, centroid=True)
        bImg = imageProcess.normalize(bImg)
        for j in range(len(dictionary['id'])):
            bImgLabelN = label==dictionary['id'][j]
            ID = numpy.max(bImgLabelN*labelImg)
            bImg = imageProcess.textOnGrayImage(bImg, str(int(ID)), (int(dictionary['centroid'][j][0])+3,int(dictionary['centroid'][j][1])-3), fontScale=fontScale, color=127, thickness=1)
        finalImage = numpy.column_stack((bImg, numpy.maximum(bImgBdry,gImg)))
        cv2.imwrite(imgDir+'/'+str(frame).zfill(zfillVal)+'.png', finalImage)
    return 0
#######################################################################


#######################################################################
# GENERATE LABELLED IMAGES WITH LABEL TAGS ON BINARY IMAGE FROM TEXT FILE
#######################################################################
def generateLabelImagesText(fp,imgDir,fontScale=1,size=1,rank=0,scale=1):
    [row,col,numFrames,frameList] = misc.getVitals(fp)
    particleList = fp.attrs['particleList']
    zfillVal = fp.attrs['zfillVal']
    procFrameList = numpy.array_split(frameList,size)
    trackingData = numpy.loadtxt(fp.attrs['outputDir']+'/tracking.dat')
    
    for frame in tqdm(procFrameList[rank]):
        gImg = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
        gImgNormScaled = imageProcess.normalize(cv2.resize(gImg,(int(col*scale),int(row*scale)),interpolation=cv2.INTER_CUBIC),min=0,max=230)
        bImg = gImgNormScaled.copy(); bImg[:] = 0
        tracking = trackingData[trackingData[:,0]==frame]
        
        for f,particle,r,c,rad,area,label in tracking:
            if (label!=0):
                rr,cc = circle_perimeter(int(r*scale),int(c*scale),int(rad*scale))
                if ((rr<0).any()==True or (cc<0).any()==True):
                    pass
                elif ((rr>row*scale-1).any()==True or (cc>col*scale-1).any()==True):
                    pass
                else:
                    gImgNormScaled[rr,cc] = 255
                    rr,cc = circle(int(r*scale),int(c*scale),int(rad*scale))
                    bImg[rr,cc] = 255
        for f,particle,r,c,rad,area,label in tracking:
            if (label!=0):
                bImg = imageProcess.textOnGrayImage(bImg, str(int(label)), (int(r*scale),int(c*scale)), fontScale=fontScale, color=127, thickness=1)
        finalImage = numpy.column_stack((gImgNormScaled, bImg))
        cv2.imwrite(imgDir+'/'+str(frame).zfill(zfillVal)+'.png', finalImage)
    return 0
#######################################################################

#######################################################################
# GENERATE LABELLED IMAGES WITH LABEL TAGS ON COLORED IMAGE
#######################################################################
#def generateLabelImagesMatplotlib(fp,imgDir,fontScale=1,size=1,rank=0,structure=[[1,1,1],[1,1,1],[1,1,1]]):
    #[row,col,numFrames,frameList] = misc.getVitals(fp)
    #particleList = fp.attrs['particleList']
    #zfillVal = fp.attrs['zfillVal']
    #procFrameList = numpy.array_split(frameList,size)
    #fig = plt.figure(figsize=(2,1))
    #ax1 = fig.add_axes([0,0,0.5,1])
    #ax2 = fig.add_axes([0.5,0,0.5,1])
    #for frame in procFrameList[rank]:
        #labelImg = fp['/segmentation/labelStack/'+str(frame).zfill(zfillVal)].value
        #gImg = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
        #gImgBlur = fp['/dataProcessing/filterStack/'+str(frame).zfill(zfillVal)].value
        #id_center = []
        #for label in numpy.unique(labelImg)[1:]:
            #bImgLabelN = labelImg==label
            #temp1, temp2, dictionary = imageProcess.regionProps(bImgLabelN, bImgLabelN, structure=structure, centroid=True)
            #id_center.append([label,dictionary['centroid'][0][0],dictionary['centroid'][0][1]])
        #ax1.imshow(gImg)
        #ax1.imshow(labelImg,cmap='jet',alpha=0.5,vmax=100)
        #ax2.imshow(gImgBlur)
        #for label,r,c in id_center:
            #ax1.text(c,r,label, color='#FFFFFF', fontsize=1.5)
        #ax1.get_xaxis().set_visible(False)
        #ax1.get_yaxis().set_visible(False)
        #ax2.get_xaxis().set_visible(False)
        #ax2.get_yaxis().set_visible(False)
        #plt.savefig(imgDir+'/'+str(frame).zfill(zfillVal)+'.png')
        #ax1.cla(), ax2.cla()
    #plt.close()
    #return 0
#######################################################################


#######################################################################
# GENERATE RGB LABELLED IMAGES WITH LABEL TAGS ON BINARY IMAGE
#######################################################################
def generateLabelImagesRGB(fp,imgDir,fontScale=1,size=1,rank=0,structure=[[1,1,1],[1,1,1],[1,1,1]]):
    [row,col,numFrames,frameList] = misc.getVitals(fp)
    particleList = fp.attrs['particleList']
    zfillVal = fp.attrs['zfillVal']
    procFrameList = numpy.array_split(frameList,size)
    for frame in procFrameList[rank]:
        labelImg = fp['/segmentation/labelStack/'+str(frame).zfill(zfillVal)].value
        gImg = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
        bImg = labelImg.astype('bool')
        bImgBdry = imageProcess.normalize(imageProcess.boundary(bImg))
        label, numLabel, dictionary = imageProcess.regionProps(bImg, gImg, structure=structure, centroid=True)
        bImg = imageProcess.normalize(bImg)
        rgbImg = imageProcess.gray2rgb(bImg)
        for j in range(len(dictionary['id'])):
            bImgLabelN = label==dictionary['id'][j]
            ID = numpy.max(bImgLabelN*labelImg)
            rgbImg = imageProcess.textOnRGBImage(rgbImg, str(ID), (int(dictionary['centroid'][j][0])+3,int(dictionary['centroid'][j][1])-3), fontScale=fontScale, color=(255,0,0), thickness=1)
        finalImage = numpy.column_stack((rgbImg, imageProcess.gray2rgb(numpy.maximum(bImgBdry,gImg))))
        finalImage = imageProcess.RGBtoBGR(finalImage)
        cv2.imwrite(imgDir+'/'+str(frame).zfill(zfillVal)+'.png', finalImage)
    return 0
#######################################################################


########################################################################
# LABELING OF PARTICLES USING MPI
########################################################################
def labelParticlesMPI(fp, comm, size, rank, centerDispRange=[5,5], perAreaChangeRange=[10,20], missFramesTh=10, structure=[[1,1,1],[1,1,1],[1,1,1]]):
    [row,col,numFrames] = fp.attrs['row'],fp.attrs['col'],fp.attrs['numFrames']
    frameList = fp.attrs['frameList']
    zfillVal = fp.attrs['zfillVal']
    procFrameList = numpy.array_split(frameList,size)
    labelSubstacks(fp, procFrameList[rank], comm, size, rank, centerDispRange=[5,5], perAreaChangeRange=[10,20], missFramesTh=10, structure=[[1,1,1],[1,1,1],[1,1,1]])
    comm.Barrier()
    if (rank == 0):
        relabelMPIStacks(size)
        maxID, occurenceFrameList = connectIndividualLabelStacks(fp, frameList, size, centerDispRange, perAreaChangeRange, missFramesTh)
    return maxID, occurenceFrameList
########################################################################


########################################################################
# LABEL SUBSTACKS USING MPI
########################################################################
def labelSubstacks(fp, frameList, comm, size, rank, centerDispRange, perAreaChangeRange, missFramesTh, structure):
    if (rank==0):
        print "Labelling image stack"
    labelStack = numpy.zeros([row,col,len(frameList)], dtype='uint32')
    for frame in frameList:
        bImg = fp['/segmentation/bImgStack/'+str(frame).zfill(zfillVal)].value
        gImg = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
        if (frame==frameList[0]):
            labelImg_0, numLabel_0, dictionary_0 = imageProcess.regionProps(bImg, gImg, structure=structure, centroid=True, area=True)
            maxID = numLabel_0
            dictionary_0['frame'] = []
            dictionary_0['initialCentroid'],dictionary_0['initialArea'],dictionary_0['initialFrame'] = [],[],[]
            for i in range(len(dictionary_0['id'])):
                dictionary_0['frame'].append(frame)
                dictionary_0['initialCentroid'].append(dictionary_1['centroid'][i])
                dictionary_0['initialArea'].append(dictionary_1['area'][i])
                dictionary_0['initialFrame'].append(frame)
            labelStack[:,:,frame-frameList[0]] = labelImg_0
        else:
            labelImg_1, numLabel_1, dictionary_1 = imageProcess.regionProps(bImg, gImg, structure=structure, centroid=True, area=True)
            if (numLabel_1>0):
                areaMin = min(dictionary_1['area']); areaMax = max(dictionary_1['area'])
            for i in range(len(dictionary_1['id'])):
                flag = 0
                bImg_1_LabelN = labelImg_1==dictionary_1['id'][i]
                center_1 = dictionary_1['centroid'][i]
                area_1 = dictionary_1['area'][i]
                frame_1 = frame
                if (areaMax-areaMin>0):
                    factor = 1.0*(area_1-areaMin)/(areaMax-areaMin)
                    perAreaChangeTh = perAreaChangeRange[1] - factor*(perAreaChangeRange[1]-perAreaChangeRange[0])
                    centerDispTh = centerDispRange[1] - factor*(centerDispRange[1]-centerDispRange[0])
                else:
                    perAreaChangeTh = perAreaChangeRange[1]
                    centerDispTh = centerDispRange[1]
                closeness,J = 1e10,0
                for j in range(len(dictionary_0['id'])-1,-1,-1):
                    center_0 = dictionary_0['centroid'][j]
                    area_0 = dictionary_0['area'][j]
                    frame_0 = dictionary_0['frame'][j]
                    centerDisp = numpy.sqrt((center_1[0]-center_0[0])**2 + (center_1[1]-center_0[1])**2)
                    perAreaChange = 100.0*numpy.abs(area_1-area_0)/numpy.maximum(area_1,area_0)
                    missFrames = frame_1-frame_0
                    if (centerDisp <= centerDispTh):
                        if (perAreaChange <= perAreaChangeTh):
                            if (missFrames <= missFramesTh):
                                if (centerDisp < closeness):
                                    closeness = centerDisp
                                    J = j
                                    flag = 1
                                    
                if (flag == 1):
                    labelStack[:,:,frame-frameList[0]] += (bImg_1_LabelN*dictionary_0['id'][J]).astype('uint32')
                    dictionary_0['centroid'][J] = center_1
                    dictionary_0['area'][J] = area_1
                    dictionary_0['frame'][J] = frame
                if (flag == 0):
                    maxID += 1
                    labelN_1 = bImg_1_LabelN*maxID
                    labelStack[:,:,frame-frameList[0]] += labelN_1.astype('uint32')
                    dictionary_0['id'].append(maxID)
                    dictionary_0['centroid'].append(center_1)
                    dictionary_0['area'].append(area_1)
                    dictionary_0['frame'].append(frame)
                    dictionary_0['initialCentroid'].append(center_1)
                    dictionary_0['initialArea'].append(area_1)
                    dictionary_0['initialFrame'].append(frame)
                    
    if (rank==0):
        print "Checking for multiple particles in a single frame"
    for frame in frameList:
        labelImg = labelStack[:,:,frame-frameList[0]]
        numLabel = imageProcess.regionProps(labelImg.astype('bool'), gImg, structure=structure)[1]
        if (numLabel != numpy.size(numpy.unique(labelImg)[1:])):
            for N in numpy.unique(labelImg)[1:]:
                labelImgN = labelImg==N
                numLabel = imageProcess.regionProps(labelImgN, gImg, structure=structure)[1]
                if (numLabel>1):
                    labelImg[labelImg==N] = 0
                    labelStack[:,:,frame-frameList[0]] = labelImg
                    
    numpy.save('labelStack_'+str(rank)+'.npy', labelStack)
    pickle.dump(dictionary_0, open('dictionary_'+str(rank), 'wb'))
    return 0
########################################################################


########################################################################
# RELABEL INDIVIDUAL STACKS IN SEQUENTIAL ORDER
########################################################################
def relabelMPIStacks(size):
    print "Relabling particles across different substacks such that they are in sequential order"
    minID,counter = 1,1
    relabelList = []
    for rank in range(size):
        dictionary = pickle.load(open('dictionary_'+str(rank),'rb'))
        minID += len(dictionary['id'])
    maxID = minID
    for rank in range(size):
        relabelListSubstack = []
        labelStack = numpy.load('labelStack_'+str(rank)+'.npy')
        dictionary = pickle.load(open('dictionary_'+str(rank),'rb'))
        for label in dictionary_0['id']:
            labelStack[labelStack==label] = maxID
            relabelListSubstack.append([maxID,counter])
            maxID+=1
            counter+=1
        relabelList.append(relabelListSubstack)
        numpy.save('labelStack_'+str(rank)+'.npy', labelStack)
    for rank in range(size):
        labelStack = numpy.load('labelStack_'+str(rank)+'.npy')
        dictionary = pickle.load(open('dictionary_'+str(rank),'rb'))
        idList = []
        for newID in relabelListSubstack[rank]:
            id0,id1 = newID[0],newID[1]
            labelStack[labelStack==id0] = id1
            idList.append(id1)
        dictionary['id']=idList
        numpy.save('labelStack_'+str(rank)+'.npy', labelStack)
        pickle.dump(dictionary, open('dictionary_'+str(rank), 'wb'))
########################################################################


########################################################################
# CONNECT INDIVIDUAL LABEL STACKS WITH EACH OTHER
########################################################################
def connectIndividualLabelStacks(fp, frameList, size, centerDispRange, perAreaChangeRange, missFramesTh):
    print "Connecting each labelled substacks using the primary processor"
    dictionary_0 = pickle.load(open('dictionary_'+str(size-1),'rb'))
    newID = max(dictionary_0['id'])+1
    
    dictionary_0 = pickle.load(open('dictionary_'+str(0),'rb'))
    for rank in range(1,size):
        dictionary_1 = pickle.load(open('dictionary_'+str(rank),'rb'))
        labelStack = numpy.load('labelStack_'+str(rank)+'.npy')
        areaMin, areaMax = min(dictionary_1['area']), max(dictionary_1['area'])
            
        for i in range(len(dictionary_1['id'])):
            flag = 0
            label_1 = dictionary_1['id'][i]
            center_1_init = dictionary_1['initialCentroid'][i]
            area_1_init = dictionary_1['initialArea'][i]
            frame_1_init = dictionary_1['initialFrame'][i]
            center_1_fin = dictionary_1['centroid'][i]
            area_1_fin = dictionary_1['area'][i]
            frame_1_fin = dictionary_1['frame'][i]
            if (areaMax-areaMin>0):
                factor = 1.0*(area_1-areaMin)/(areaMax-areaMin)
                perAreaChangeTh = perAreaChangeRange[1] - factor*(perAreaChangeRange[1]-perAreaChangeRange[0])
                centerDispTh = centerDispRange[1] - factor*(centerDispRange[1]-centerDispRange[0])
            else:
                perAreaChangeTh = perAreaChangeRange[1]
                centerDispTh = centerDispRange[1]
            closeness,J = 1e10,0
            for j in range(len(dictionary_0['id'])):
                label_0 = dictionary_0['id'][j]
                center_0_fin = dictionary_0['centroid'][j]
                area_0_fin = dictionary_0['area'][j]
                frame_0_fin = dictionary_0['frame'][j]
                centerDisp = numpy.sqrt((center_1_init[0]-center_0_fin[0])**2 + (center_1_init[1]-center_0_fin[1])**2)
                perAreaChange = 100.0*numpy.abs(area_1_init-area_0_fin)/numpy.maximum(area_1_init,area_0_fin)
                missFrames = frame_1_init-frame_0_fin
                if (centerDisp <= centerDispTh):
                    if (perAreaChange <= perAreaChangeTh):
                        if (missFrames <= missFramesTh):
                            if (centerDisp < closeness):
                                closeness = centerDisp
                                label_0 = dictionary_0['id'][j]
                                flag = 1
                                
            if (flag == 1):
                dictionary_0['centroid'][i] = center_1_fin
                dictionary_0['area'][i] = area_1_fin
                dictionary_0['frame'][i] = frame1_fin
                labelStack[labelStack==label_1] = label_0
            else:
                dictionary_0['id'].append(label_1)
                dictionary_0['centroid'].append(center_1_fin)
                dictionary_0['area'].append(area_1_fin)
                dictionary_0['frame'].append(frame_fin)
                dictionary_0['initialCentroid'].append(center_1_init)
                dictionary_0['initialArea'].append(area_1_init)
                dictionary_0['initialFrame'].append(frame_1_init)
        numpy.save('labelStack_'+str(rank)+'.npy', labelStack)
        
    for rank in range(size):
        if (rank == 0):
            labelStack = numpy.load('labelStack_'+str(rank)+'.npy')
        else:
            labelStack = numpy.concatenate((labelStack, numpy.load('labelStack_'+str(rank)+'.npy')), axis=2)
            
    occurenceFrameList = []
    particleList = dictionary_0['id']
    newParticleList = range(1,len(dictionary_0['id'])+1)
    dictionary_0['id'] = []
    for particle, newParticle in zip(particleList,newParticleList):
        labelStack[labelStack==particle] = newParticle
        dictionary_0['id'].append(newParticle)
        occurenceFrameList.append([newParticle,dictionary_0['initialFrame'][newParticle-1]])
    for frame in frameList:
        fileIO.writeH5Dataset(fp,'/segmentation/labelStack/'+str(frame).zfill(zfillVal), labelStack[:,:,frame-1])
    return newParticleList[-1], occurenceFrameList
########################################################################
