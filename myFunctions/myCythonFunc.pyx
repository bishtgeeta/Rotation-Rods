import numpy
import time
cimport numpy
from scipy import ndimage
import skimage
from skimage import measure
import matplotlib.pyplot as plt
import cv2

DTYPE = numpy.double
ctypedef numpy.double_t DTYPE_t

cdef extern from "math.h":
	double sqrt(double x)
	double floor(double x)
	double atan(double x)
	double fabs(double x)
	double round(double x)

cdef extern from "complex.h":
	double cabs(complex x)

cimport cython
from cpython cimport bool
from cython.parallel import parallel, prange
@cython.boundscheck(False)

def angAvg2D_C(numpy.ndarray[numpy.double_t, ndim=2] fImg, numpy.ndarray[numpy.int32_t, ndim=2] qArr):
	assert fImg.dtype == numpy.double and qArr.dtype == numpy.int32

	cdef int qMax = qArr.max()+1
	cdef numpy.ndarray[numpy.double_t, ndim=1] avg = numpy.zeros([qMax], dtype=numpy.double)
	cdef numpy.ndarray[numpy.double_t, ndim=1] avg2 = numpy.zeros([qMax], dtype=numpy.double)
	cdef numpy.ndarray[numpy.double_t, ndim=1] measure = numpy.zeros([qMax], dtype=numpy.double)
	cdef numpy.ndarray[numpy.double_t, ndim=1] counter = numpy.zeros([qMax], dtype=numpy.double)

	cdef int row = qArr.shape[0]
	cdef int col = qArr.shape[1]
	cdef int i, j
	cdef double var

	ringVals = {i:[] for i in range(qMax)}

	for i in range(row):
		for j in range(col):
			avg[qArr[i,j]] += fImg[i,j]
			avg2[qArr[i,j]] += fImg[i,j]*fImg[i,j]
			counter[qArr[i,j]] += 1
			ringVals[qArr[i,j]].append(fImg[i,j])
	for i in range(qMax):
		if counter[i] > 0:
			avg[i] /= counter[i]
			avg2[i] /= counter[i]
	for i in range(qMax):
		if avg[i] > 0:
			var = avg2[i] - avg[i]*avg[i]
			measure[i] = sqrt(var) / avg[i]

	return (avg, measure, ringVals)


def filterFourier_C(numpy.ndarray[numpy.double_t, ndim=2] fImg, numpy.ndarray[numpy.int32_t, ndim=2] qArr, double cutoff=1.0):
	assert fImg.dtype == numpy.double and qArr.dtype == numpy.int32

	cdef int qMax = qArr.max()+1
	cdef int row = qArr.shape[0]
	cdef int col = qArr.shape[1]
	cdef numpy.ndarray[numpy.double_t, ndim=1] avg = numpy.zeros([qMax], dtype=numpy.double)
	cdef numpy.ndarray[numpy.double_t, ndim=1] avg2 = numpy.zeros([qMax], dtype=numpy.double)
	cdef numpy.ndarray[numpy.double_t, ndim=1] counter = numpy.zeros([qMax], dtype=numpy.double)
	cdef numpy.ndarray[numpy.uint8_t, ndim=2] bImg = numpy.zeros([row, col], dtype=numpy.uint8)

	cdef int i, j, k
	cdef double var, temp

	for i in range(row):
		for j in range(col):
			avg[qArr[i,j]] += fImg[i,j]
			avg2[qArr[i,j]] += fImg[i,j]*fImg[i,j]
			counter[qArr[i,j]] += 1
	for i in range(qMax):
		if counter[i] > 0:
			avg[i] /= counter[i]
			avg2[i] /= counter[i]
	for i in range(qMax):
		if avg[i] > 0:
			avg2[i] = sqrt(avg2[i] - avg[i]*avg[i])
	for i in range(row):
		for j in range(col):
			k = qArr[i,j]
			temp = fabs(fImg[i,j])
			if (fabs(temp-avg[k]) > (cutoff*avg2[k])) and (avg[k] > 0):
				bImg[i,j] = 1
	return bImg


def connAreaID_C(numpy.ndarray[numpy.uint8_t, ndim=2] fImg1, numpy.ndarray[numpy.double_t, ndim=2] fImg, numpy.ndarray[numpy.uint8_t, ndim=2] bImg, numpy.ndarray[numpy.int_t, ndim=2] qArr, qTcks, rTcks, areaRange=[10, 20], ringRange=[0, 1E10]):
	assert fImg1.dtype == numpy.uint8 and fImg.dtype == numpy.double and bImg.dtype == numpy.uint8 and qArr.dtype == numpy.int

	cdef int r = bImg.shape[0]
	cdef int c = bImg.shape[1]
	cdef int ID = 0
	cdef numpy.ndarray[numpy.uint8_t, ndim=2] visitID = numpy.zeros([r,c], dtype=numpy.uint8)
	cdef numpy.ndarray[numpy.uint_t, ndim=2] IDImg = numpy.zeros([r,c], dtype=numpy.uint)
	cdef numpy.ndarray[numpy.int32_t, ndim=1] areaRangeC = numpy.zeros([2], dtype=numpy.int32)
	cdef numpy.ndarray[numpy.double_t, ndim=1] centroid = numpy.zeros([2], dtype=numpy.double)
	cdef int i, j, X, Y, lenRingRange
	cdef double theta
	cdef double tic

	areaRangeC[0] = areaRange[0]
	areaRangeC[1] = areaRange[1]

	cdef numpy.ndarray[numpy.double_t, ndim=1] ringRangeC = numpy.zeros(2, dtype=numpy.double)

	ringRangeC[0] = ringRange[0]
	ringRangeC[1] = ringRange[1]

	IDpixelListTheta = {}
	IDpixelListTheta['ID'] = []
	IDpixelListTheta['pixelList'] = []
	IDpixelListTheta['intensityList'] = []
	IDpixelListTheta['theta'] = []
	IDpixelListTheta['area'] = []
	IDpixelListTheta['sumIntensity'] = []
	IDpixelListTheta['qArrList'] = []
	IDpixelListTheta['rTcks'] = []
	IDpixelListTheta['radiusList'] = []
	IDpixelListTheta['centroidList'] = []

	for i in range(r):
		for j in range(c):
			pixelList = []
			intensityList = []
			qArrList = []
			if (visitID[i,j] == 0):
				visitID[i,j] = 1
				if (bImg[i,j] == 255):
					flag = 0
					if (qArr[i,j] >= ringRangeC[0] and qArr[i,j] <= ringRangeC[1]):
						flag = 1
					if (flag == 1):
						[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i, j, r, c, visitID, fImg, bImg, qArr, pixelList, intensityList, qArrList)
						if (len(pixelList) >= areaRangeC[0] and len(pixelList) <= areaRangeC[1]):
							theta = 90
							X = Y = 0
							for k in pixelList:
								X += k[0]
								Y += k[1]
							centroid[0] = X*1.0/len(pixelList)
							centroid[1] = Y*1.0/len(pixelList)
							if (centroid[1] - floor(c/2.0) > 0):
								theta = atan((floor(r/2.0)-centroid[0])/(centroid[1]-floor(c/2.0))) * 180 / numpy.pi
								ID += 1
								for k in pixelList:
									IDImg[k[0], k[1]] = ID
								radii = []
								for k in range(len(pixelList)):
									radii.append(int(round(sqrt((pixelList[k][0]-centroid[0])**2 + (pixelList[k][1]-centroid[1])**2))))
#								cv.Circle(cv.fromarray(fImg1), (int(round(centroid[1])),int(round(centroid[0]))), max(radii), 255, thickness=1)
								IDpixelListTheta['ID'].append(ID)
								IDpixelListTheta['pixelList'].append(pixelList)
								IDpixelListTheta['intensityList'].append(intensityList)
								IDpixelListTheta['theta'].append(theta)
								IDpixelListTheta['area'].append(len(pixelList))
								IDpixelListTheta['sumIntensity'].append(sum(intensityList))
								IDpixelListTheta['qArrList'].append(qArrList)
								IDpixelListTheta['rTcks'].append([rTcks[qArr[int(round(centroid[0])), int(round(centroid[1]))]]])
								IDpixelListTheta['radiusList'].append(max(radii))
								IDpixelListTheta['centroidList'].append([int(round(centroid[0])), int(round(centroid[1]))])
							else:
								for k in pixelList:
									bImg[k[0], k[1]] = 0
						else:
							for k in pixelList:
								bImg[k[0], k[1]] = 0
					else:
						bImg[i,j] = 0
	return (fImg1, bImg, IDImg, IDpixelListTheta)


def connAreaID_py(bImg, qArr, areaRange=[10, 20], ringRange=[0, 1E10]):
	[r,c] = bImg.shape
	ID = 0
	IDpixelListTheta = {}
	visitID = numpy.zeros([r,c], dtype='uint8')
	IDImg = numpy.zeros([r,c], dtype='uint')

	for i in range(r):
		for j in range(c):
			pixelList = []
			if (visitID[i,j] == 0):
				visitID[i,j] = 1
				if (bImg[i,j] == 1 and qArr[i,j] >= ringRange[0] and qArr[i,j] <= ringRange[1]):
					[visitID, pixelList] = findNeighbors_py(i, j, r, c, visitID, bImg, pixelList)
					if (len(pixelList) >= areaRange[0] and len(pixelList) <= areaRange[1]):
						theta = 90
						X = Y = 0
						for k in pixelList:
							X += k[0]
							Y += k[1]
						centroid = [int(floor(r/2)) - int(round(X*1.0/len(pixelList))), int(round(Y*1.0/len(pixelList))) - int(floor(c/2))]
						if (centroid[1] > 0):
							theta = atan(centroid[0]*1.0/centroid[1]) * 180 / 3.141592653589793
							if (abs(theta) <= 5 or abs(abs(theta) - 90) <= 5):
								for k in pixelList:
									bImg[k[0], k[1]] = 0
							else:
								ID += 1
								for k in pixelList:
									IDImg[k[0], k[1]] = ID
								if (ID == 1):
									IDpixelListTheta['ID'] = [ID]
									IDpixelListTheta['pixelList'] = [pixelList]
									IDpixelListTheta['theta'] = [theta]
								else:
									IDpixelListTheta['ID'].append(ID)
									IDpixelListTheta['pixelList'].append(pixelList)
									IDpixelListTheta['theta'].append(theta)
						else:
							for k in pixelList:
								bImg[k[0], k[1]] = 0
					else:
						for k in pixelList:
							bImg[k[0], k[1]] = 0
	return (bImg, IDImg, IDpixelListTheta)


def findNeighbors_py(i,j,r,c,visitID,bImg,pixelList):
	pixelList.append([i,j])
	if (i == 0):
		if (j == 0):
			if (visitID[i,j+1] == 0):
				visitID[i,j+1] = 1
				if (bImg[i,j+1] == 1):
					[visitID, pixelList] = findNeighbors_py(i,j+1,r,c,visitID,bImg,pixelList)
			if (visitID[i+1,j] == 0):
				visitID[i+1,j] = 1
				if (bImg[i+1,j] == 1):
					[visitID, pixelList] = findNeighbors_py(i+1,j,r,c,visitID,bImg,pixelList)
		elif (j == c-1):
			if (visitID[i,j-1] == 0):
				visitID[i,j-1] = 1
				if (bImg[i,j-1] == 1):
					[visitID, pixelList] = findNeighbors_py(i,j-1,r,c,visitID,bImg,pixelList)
			if (visitID[i+1,j] == 0):
				visitID[i+1,j] = 1
				if (bImg[i+1,j] == 1):
					[visitID, pixelList] = findNeighbors_py(i+1,j,r,c,visitID,bImg,pixelList)
		else:
			if (visitID[i,j-1] == 0):
				visitID[i,j-1] = 1
				if (bImg[i,j-1] == 1):
					[visitID, pixelList] = findNeighbors_py(i,j-1,r,c,visitID,bImg,pixelList)
			if (visitID[i+1,j] == 0):
				visitID[i+1,j] = 1
				if (bImg[i+1,j] == 1):
					[visitID, pixelList] = findNeighbors_py(i+1,j,r,c,visitID,bImg,pixelList)
			if (visitID[i,j+1] == 0):
				visitID[i,j+1] = 1
				if (bImg[i,j+1] == 1):
					[visitID, pixelList] = findNeighbors_py(i,j+1,r,c,visitID,bImg,pixelList)
	elif (i == r-1):
		if (j == 0):
			if (visitID[i-1,j] == 0):
				visitID[i-1,j] = 1
				if (bImg[i-1,j] == 1):
					[visitID, pixelList] = findNeighbors_py(i-1,j,r,c,visitID,bImg,pixelList)
		if (visitID[i,j+1] == 0):
				visitID[i,j+1] = 1
				if (bImg[i,j+1] == 1):
					[visitID, pixelList] = findNeighbors_py(i,j+1,r,c,visitID,bImg,pixelList)
		elif (j == c-1):
			if (visitID[i,j-1] == 0):
				visitID[i,j-1] = 1
				if (bImg[i,j-1] == 1):
					[visitID, pixelList] = findNeighbors_py(i,j-1,r,c,visitID,bImg,pixelList)
			if (visitID[i-1,j] == 0):
				visitID[i-1,j] = 1
				if (bImg[i-1,j] == 1):
					[visitID, pixelList] = findNeighbors_py(i-1,j,r,c,visitID,bImg,pixelList)
		else:
			if (visitID[i,j-1] == 0):
				visitID[i,j-1] = 1
				if (bImg[i,j-1] == 1):
					[visitID, pixelList] = findNeighbors_py(i,j-1,r,c,visitID,bImg,pixelList)
			if (visitID[i-1,j] == 0):
				visitID[i-1,j] = 1
				if (bImg[i-1,j] == 1):
					[visitID, pixelList] = findNeighbors_py(i-1,j,r,c,visitID,bImg,pixelList)
			if (visitID[i,j+1] == 0):
				visitID[i,j+1] = 1
				if (bImg[i,j+1] == 1):
					[visitID, pixelList] = findNeighbors_py(i,j+1,r,c,visitID,bImg,pixelList)
	elif (j == 0):
		if (visitID[i-1,j] == 0):
				visitID[i-1,j] = 1
				if (bImg[i-1,j] == 1):
					[visitID, pixelList] = findNeighbors_py(i-1,j,r,c,visitID,bImg,pixelList)
		if (visitID[i,j+1] == 0):
				visitID[i,j+1] = 1
				if (bImg[i,j+1] == 1):
					[visitID, pixelList] = findNeighbors_py(i,j+1,r,c,visitID,bImg,pixelList)
		if (visitID[i+1,j] == 0):
				visitID[i+1,j] = 1
				if (bImg[i+1,j] == 1):
					[visitID, pixelList] = findNeighbors_py(i+1,j,r,c,visitID,bImg,pixelList)
	elif (j == c-1):
		if (visitID[i-1,j] == 0):
				visitID[i-1,j] = 1
				if (bImg[i-1,j] == 1):
					[visitID, pixelList] = findNeighbors_py(i-1,j,r,c,visitID,bImg,pixelList)
		if (visitID[i,j-1] == 0):
				visitID[i,j-1] = 1
				if (bImg[i,j-1] == 1):
					[visitID, pixelList] = findNeighbors_py(i,j-1,r,c,visitID,bImg,pixelList)
		if (visitID[i+1,j] == 0):
				visitID[i+1,j] = 1
				if (bImg[i+1,j] == 1):
					[visitID, pixelList] = findNeighbors_py(i+1,j,r,c,visitID,bImg,pixelList)
	else:
		if (visitID[i,j-1] == 0):
				visitID[i,j-1] = 1
				if (bImg[i,j-1] == 1):
					[visitID, pixelList] = findNeighbors_py(i,j-1,r,c,visitID,bImg,pixelList)
		if (visitID[i-1,j] == 0):
				visitID[i-1,j] = 1
				if (bImg[i-1,j] == 1):
					[visitID, pixelList] = findNeighbors_py(i-1,j,r,c,visitID,bImg,pixelList)
		if (visitID[i,j+1] == 0):
				visitID[i,j+1] = 1
				if (bImg[i,j+1] == 1):
					[visitID, pixelList] = findNeighbors_py(i,j+1,r,c,visitID,bImg,pixelList)
		if (visitID[i+1,j] == 0):
				visitID[i+1,j] = 1
				if (bImg[i+1,j] == 1):
					[visitID, pixelList] = findNeighbors_py(i+1,j,r,c,visitID,bImg,pixelList)
	return visitID, pixelList


def findNeighbors_C(int i, int j, int r, int c, numpy.ndarray[numpy.uint8_t, ndim=2] visitID, numpy.ndarray[numpy.double_t, ndim=2] gImg, numpy.ndarray[numpy.uint8_t, ndim=2] bImg, numpy.ndarray[numpy.int_t, ndim=2] qArr, pixelList, intensityList, qArrList):
	assert visitID.dtype == numpy.uint8 and gImg.dtype == numpy.double and bImg.dtype == numpy.uint8 and qArr.dtype == numpy.int
	pixelList.append([i,j])
	intensityList.append(gImg[i,j])
	qArrList.append(qArr[i,j])
	if (i == 0):
		if (j == 0):
			if (visitID[i,j+1] == 0):
				visitID[i,j+1] = 1
				if (bImg[i,j+1] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i,j+1,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
			if (visitID[i+1,j] == 0):
				visitID[i+1,j] = 1
				if (bImg[i+1,j] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i+1,j,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
		elif (j == c-1):
			if (visitID[i,j-1] == 0):
				visitID[i,j-1] = 1
				if (bImg[i,j-1] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i,j-1,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
			if (visitID[i+1,j] == 0):
				visitID[i+1,j] = 1
				if (bImg[i+1,j] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i+1,j,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
		else:
			if (visitID[i,j-1] == 0):
				visitID[i,j-1] = 1
				if (bImg[i,j-1] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i,j-1,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
			if (visitID[i+1,j] == 0):
				visitID[i+1,j] = 1
				if (bImg[i+1,j] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i+1,j,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
			if (visitID[i,j+1] == 0):
				visitID[i,j+1] = 1
				if (bImg[i,j+1] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i,j+1,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
	elif (i == r-1):
		if (j == 0):
			if (visitID[i-1,j] == 0):
				visitID[i-1,j] = 1
				if (bImg[i-1,j] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i-1,j,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
		if (visitID[i,j+1] == 0):
				visitID[i,j+1] = 1
				if (bImg[i,j+1] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i,j+1,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
		elif (j == c-1):
			if (visitID[i,j-1] == 0):
				visitID[i,j-1] = 1
				if (bImg[i,j-1] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i,j-1,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
			if (visitID[i-1,j] == 0):
				visitID[i-1,j] = 1
				if (bImg[i-1,j] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i-1,j,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
		else:
			if (visitID[i,j-1] == 0):
				visitID[i,j-1] = 1
				if (bImg[i,j-1] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i,j-1,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
			if (visitID[i-1,j] == 0):
				visitID[i-1,j] = 1
				if (bImg[i-1,j] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i-1,j,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
			if (visitID[i,j+1] == 0):
				visitID[i,j+1] = 1
				if (bImg[i,j+1] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i,j+1,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
	elif (j == 0):
		if (visitID[i-1,j] == 0):
				visitID[i-1,j] = 1
				if (bImg[i-1,j] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i-1,j,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
		if (visitID[i,j+1] == 0):
				visitID[i,j+1] = 1
				if (bImg[i,j+1] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i,j+1,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
		if (visitID[i+1,j] == 0):
				visitID[i+1,j] = 1
				if (bImg[i+1,j] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i+1,j,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
	elif (j == c-1):
		if (visitID[i-1,j] == 0):
				visitID[i-1,j] = 1
				if (bImg[i-1,j] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i-1,j,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
		if (visitID[i,j-1] == 0):
				visitID[i,j-1] = 1
				if (bImg[i,j-1] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i,j-1,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
		if (visitID[i+1,j] == 0):
				visitID[i+1,j] = 1
				if (bImg[i+1,j] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i+1,j,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
	else:
		if (visitID[i,j-1] == 0):
				visitID[i,j-1] = 1
				if (bImg[i,j-1] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i,j-1,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
		if (visitID[i-1,j] == 0):
				visitID[i-1,j] = 1
				if (bImg[i-1,j] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i-1,j,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
		if (visitID[i,j+1] == 0):
				visitID[i,j+1] = 1
				if (bImg[i,j+1] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i,j+1,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
		if (visitID[i+1,j] == 0):
				visitID[i+1,j] = 1
				if (bImg[i+1,j] == 255):
					[visitID, pixelList, intensityList, qArrList] = findNeighbors_C(i+1,j,r,c,visitID,gImg,bImg,qArr, pixelList,intensityList,qArrList)
	return (visitID, pixelList, intensityList, qArrList)


def medianFilter2D_C(numpy.ndarray[numpy.uint8_t, ndim=2] gImg, size):
	assert gImg.dtype == numpy.uint8

	cdef int row, col
	cdef numpy.ndarray[numpy.int32_t, ndim=1] freq = numpy.zeros([256], dtype=numpy.int32)
	cdef numpy.ndarray[numpy.int32_t, ndim=1] size_median = numpy.zeros([2], dtype=numpy.int32)
	cdef int i, j, k, ii, jj, kk, midPoint, cumFreq

	row = gImg.shape[0]
	col = gImg.shape[1]
	size_median[0] = size[0]
	size_median[1] = size[1]

	cdef numpy.ndarray[numpy.uint8_t, ndim=2] gImgPad = numpy.zeros([row+2*size_median[0], col+2*size_median[1]], dtype=numpy.uint8)
	cdef numpy.ndarray[numpy.uint8_t, ndim=2] gImgSharp = numpy.zeros([row, col], dtype=numpy.uint8)

	gImgPad[size_median[0]:row+size_median[0], size_median[1]:col+size_median[1]] = gImg
	for i in range(0, size_median[1]):
		gImgPad[:,i] = gImgPad[:,size_median[1]+(size_median[1]-i)]
	for i in range(col+size_median[1], col+2*size_median[1]):
		gImgPad[:,i] = gImgPad[:,(col+size_median[1]-1)-(i-(col+size_median[1]-1))]
	for i in range(0, size_median[0]):
		gImgPad[i,:] = gImgPad[size_median[0]+(size_median[0]-i),:]
	for i in range(row+size_median[0], row+2*size_median[0]):
		gImgPad[i,:] = gImgPad[(row+size_median[0]-1)-(i-(row+size_median[0]-1)),:]

	midPoint = (2*size_median[0]+1)*(2*size_median[1]+1)/2 + 1
	for j in range(size_median[1], col+size_median[1]):
		for i in range(size_median[0], row+size_median[0]):
			freq[:] = 0
			cumFreq = 0
			for ii in range(-size_median[0],size_median[0]+1,1):
				for jj in range(-size_median[1],size_median[1]+1,1):
					freq[gImgPad[i+ii,j+jj]] += 1
			for ii in range(256):
				cumFreq += freq[ii]
				if (cumFreq >= midPoint):
					gImgSharp[i-size_median[0], j-size_median[1]] = ii
					break
	return gImgSharp


def medianFilter3D_py(gImgStack, size):
	[row, col, numFrames] = gImgStack.shape
	gImgStackSharp = numpy.zeros([row, col, numFrames], dtype='uint8')
	gImgStackPad = numpy.zeros([row+2*size[0], col+2*size[1], numFrames+2*size[2]], dtype='uint8')
	freq = numpy.zeros(256, dtype='int')

	print "Creating a padded image stack for median filtering"
	gImgStackPad[size[0]:row+size[0], size[1]:col+size[1], size[2]:numFrames+size[2]] = gImgStack
	for i in range(0, size[2]):
		gImgStackPad[:,:,i] = gImgStackPad[:,:,size[2]+(size[2]-i)]
	for i in range(numFrames+size[2], numFrames+2*size[2]):
		gImgStackPad[:,:,i] = gImgStackPad[:,:,(numFrames+size[2]-1)-(i-(numFrames+size[2]-1))]
	for i in range(0, size[1]):
		gImgStackPad[:,i,:] = gImgStackPad[:,size[1]+(size[1]-i),:]
	for i in range(col+size[1], col+2*size[1]):
		gImgStackPad[:,i,:] = gImgStackPad[:,(col+size[1]-1)-(i-(col+size[1]-1))]
	for i in range(0, size[0]):
		gImgStackPad[i,:,:] = gImgStackPad[size[0]+(size[0]-i),:,:]
	for i in range(row+size[0], row+2*size[0]):
		gImgStackPad[i,:,:] = gImgStackPad[(row+size[0]-1)-(i-(row+size[0]-1)),:,:]

	print "Performing median filtering"
	midPoint = int((2*size[0]+1)*(2*size[1]+1)*(2*size[2]+1)/2) + 1
	for k in range(size[2], numFrames+size[2]):
		print "Calculating median for frame", k-size[2]
		for j in range(size[1], col+size[1]):
			for i in range(size[0], row+size[0]):
				freq[:] = 0
				cumFreq = 0
				for ii in range(-size[0],size[0]+1,1):
					for jj in range(-size[1],size[1]+1,1):
						for kk in range(-size[2],size[2]+1,1):
							freq[gImgStackPad[i+ii,j+jj,k+kk]] += 1
				for ii in range(256):
					cumFreq += freq[ii]
					if (cumFreq >= midPoint):
						gImgStackSharp[i-size[0], j-size[1], k-size[2]] = ii
						break
	return gImgStackSharp


def medianFilter3D_C(numpy.ndarray[numpy.uint8_t, ndim=3] gImgStack, size):
	assert gImgStack.dtype == numpy.uint8

	cdef int row, col, numFrames
	cdef numpy.ndarray[numpy.int32_t, ndim=1] freq = numpy.zeros([256], dtype=numpy.int32)
	cdef numpy.ndarray[numpy.int32_t, ndim=1] size_median = numpy.zeros([3], dtype=numpy.int32)
	cdef int i, j, k, ii, jj, kk, midPoint, cumFreq

	row = gImgStack.shape[0]
	col = gImgStack.shape[1]
	numFrames = gImgStack.shape[2]
	size_median[0] = size[0]
	size_median[1] = size[1]
	size_median[2] = size[2]

	cdef numpy.ndarray[numpy.uint8_t, ndim=3] gImgStackPad = numpy.zeros([row+2*size_median[0], col+2*size_median[1], numFrames+2*size_median[2]], dtype=numpy.uint8)
	cdef numpy.ndarray[numpy.uint8_t, ndim=3] gImgStackSharp = numpy.zeros([row, col, numFrames], dtype=numpy.uint8)

	print "Creating a padded image stack for median filtering"
	gImgStackPad[size_median[0]:row+size_median[0], size_median[1]:col+size_median[1], size_median[2]:numFrames+size_median[2]] = gImgStack
	for i in range(0, size_median[2]):
		gImgStackPad[:,:,i] = gImgStackPad[:,:,size_median[2]+(size_median[2]-i)]
	for i in range(numFrames+size_median[2], numFrames+2*size_median[2]):
		gImgStackPad[:,:,i] = gImgStackPad[:,:,(numFrames+size_median[2]-1)-(i-(numFrames+size_median[2]-1))]
	for i in range(0, size_median[1]):
		gImgStackPad[:,i,:] = gImgStackPad[:,size_median[1]+(size_median[1]-i),:]
	for i in range(col+size_median[1], col+2*size_median[1]):
		gImgStackPad[:,i,:] = gImgStackPad[:,(col+size_median[1]-1)-(i-(col+size_median[1]-1))]
	for i in range(0, size_median[0]):
		gImgStackPad[i,:,:] = gImgStackPad[size_median[0]+(size_median[0]-i),:,:]
	for i in range(row+size_median[0], row+2*size_median[0]):
		gImgStackPad[i,:,:] = gImgStackPad[(row+size_median[0]-1)-(i-(row+size_median[0]-1)),:,:]

	print "Performing median filtering"
	midPoint = (2*size_median[0]+1)*(2*size_median[1]+1)*(2*size_median[2]+1)/2 + 1
	for k in range(size_median[2], numFrames+size_median[2]):
		print "Calculating median for frame", k-size_median[2]
		for j in range(size_median[1], col+size_median[1]):
			for i in range(size_median[0], row+size_median[0]):
				freq[:] = 0
				cumFreq = 0
				for ii in range(-size_median[0],size_median[0]+1,1):
					for jj in range(-size_median[1],size_median[1]+1,1):
						for kk in range(-size_median[2],size_median[2]+1,1):
							freq[gImgStackPad[i+ii,j+jj,k+kk]] += 1
				for ii in range(256):
					cumFreq += freq[ii]
					if (cumFreq >= midPoint):
						gImgStackSharp[i-size_median[0], j-size_median[1], k-size_median[2]] = ii
						break
	return gImgStackSharp


def removeEdges(numpy.ndarray[numpy.uint8_t, ndim=2]img, numpy.ndarray[numpy.uint8_t, ndim=2]grad, numpy.ndarray[numpy.uint8_t, ndim=1]exclude):
	assert img.dtype == numpy.uint8 and grad.dtype==numpy.uint8 and exclude.dtype == numpy.uint8
	cdef int row, col, i ,j, excludeSize

	row = img.shape[0]
	col = img.shape[1]
	excludeSize = exclude.size
	for i in range(1,row-1):
		for j in range(1,col-1):
			for k in exclude:
				if (img[i-1][j-1]==k or img[i-1][j]==k or img[i-1][j+1]==k or img[i][j-1]==k or img[i][j]==k or img[i][j+1]==k or img[i+1][j-1]==k or img[i+1][j]==k or img[i+1][j+1]==k):
					grad[i,j]=0
	return grad


def gradient(numpy.ndarray[numpy.double_t, ndim=2] img, numpy.ndarray[numpy.uint8_t, ndim=1] exclude):
	assert img.dtype == numpy.double and exclude.dtype == numpy.uint8
	cdef int row, col, i ,j, excludeSize, flagR, flagC, ii
	cdef double gradR, gradC

	row = img.shape[0]
	col = img.shape[1]
	excludeSize = exclude.size
	cdef numpy.ndarray[numpy.double_t, ndim=2] grad = numpy.zeros([row, col], dtype=numpy.double)

	for i in range(row):
		for j in range(col):
			gradR = gradC = 0.0
			flagR = flagC = 0
			if (i == row-1):
				for ii in range(excludeSize):
					if ((img[i,j] == exclude[ii]) or (img[i-1, j] == exclude[ii])):
						flagR = 1
						break
				if (flagR == 0):
					gradR = (img[i, j] - img[i-1, j])**2
			else:
				for ii in range(excludeSize):
					if ((img[i,j] == exclude[ii]) or (img[i+1, j] == exclude[ii])):
						flagR = 1
						break
				if (flagR == 0):
					gradR = (img[i, j] - img[i+1, j])**2
			if (j == col-1):
				for ii in range(excludeSize):
					if ((img[i,j] == exclude[ii]) or (img[i, j-1] == exclude[ii])):
						flagC = 1
						break
				if (flagC == 0):
					gradC = (img[i, j] - img[i, j-1])**2
			else:
				for ii in range(excludeSize):
					if ((img[i,j] == exclude[ii]) or (img[i, j+1] == exclude[ii])):
						flagC = 1
						break
				if (flagC == 0):
					gradC = (img[i, j] - img[i, j+1])**2
			grad[i,j] = sqrt(gradR + gradC)
	grad = ((grad - grad.min())/(grad.max() - grad.min())*255.0)
	return grad


def cleanContour(numpy.ndarray[numpy.uint8_t, ndim=2] contour, method='none', int minNeighbors=0):
	assert contour.dtype == numpy.uint8

	cdef int i, j, numNeighbors
	cdef int row, col

	if (method == 'neighbors'):
		row = contour.shape[0]
		col = contour.shape[1]
		contourClean = contour.copy()
		for i in range(1,row-1):
			for j in range(1,col-1):
				if (contour[i,j] > 0):
					numNeighbors = contour[i,j-1]*1+contour[i,j+1]*1+contour[i-1,j]*1+contour[i+1,j]*1 + contour[i-1,j-1]*1+contour[i-1,j+1]*1+contour[i+1,j+1]*1+contour[i+1,j-1]*1
					if (numNeighbors <= minNeighbors):
						contourClean[i,j] = 0
		return contourClean.astype('bool')
	elif (method == 'edges'):
		row = contour.shape[0]
		col = contour.shape[1]
		for i in range(row):
			if (contour[i, 0] > 0):
				contour = removeContour(contour, row, col, i, 0)
			if (contour[i, col-1] > 0):
				contour = removeContour(contour, row, col, i, col-1)
		for i in range(col):
			if (contour[0, i] > 0):
				contour = removeContour(contour, row, col, 0, i)
			if (contour[row-1, i] > 0):
				contour = removeContour(contour, row, col, row-1, i)
		return contour.astype('bool')
	else:
		return contour


def removeContour(numpy.ndarray[numpy.uint8_t, ndim=2] contour, int row, int col, int i, int j):
	assert contour.dtype == numpy.uint8
	cdef int primaryNeighbors, secondaryNeighbors, numNeighbors
	contour[i,j] = False
	if (i == 0):
		if (j == 0):
			primaryNeighbors = contour[i,j+1]*1+contour[i+1,j]*1
			secondaryNeighbors = contour[i+1,j+1]*1
			numNeighbors = primaryNeighbors+secondaryNeighbors
			if (numNeighbors <= 2):
				if (primaryNeighbors == 1 or primaryNeighbors == 2):
					if (contour[i,j+1] == True):
						contour = removeContour(contour,row,col,i,j+1)
					if (contour[i+1,j] == True):
						contour = removeContour(contour,row,col,i,j+1)
				elif (secondaryNeighbors == 1):
					if (contour[i+1,j+1] == True):
						contour = removeContour(contour,row,col,i,j+1)
		elif (j == col-1):
			primaryNeighbors = contour[i,j-1]*1+contour[i+1,j]*1
			secondaryNeighbors = contour[i+1,j-1]*1
			numNeighbors = primaryNeighbors+secondaryNeighbors
			if (numNeighbors <= 2):
				if (primaryNeighbors == 1 or primaryNeighbors == 2):
					if (contour[i,j-1] == True):
						contour = removeContour(contour,row,col,i,j-1)
					if (contour[i+1,j] == True):
						contour = removeContour(contour,row,col,i+1,j)
				elif (secondaryNeighbors == 1):
					if (contour[i+1,j-1] == True):
						contour = removeContour(contour,row,col,i+1,j-1)
		else:
			primaryNeighbors = contour[i,j-1]*1+contour[i,j+1]+contour[i+1,j]*1
			secondaryNeighbors = contour[i+1,j-1]*1+contour[i+1,j+1]*1
			numNeighbors = primaryNeighbors+secondaryNeighbors
			if (numNeighbors <= 2):
				if (primaryNeighbors == 1 or primaryNeighbors == 2):
					if (contour[i,j-1] == True):
						contour = removeContour(contour,row,col,i,j-1)
					if (contour[i,j+1] == True):
						contour = removeContour(contour,row,col,i,j+1)
					if (contour[i+1,j] == True):
						contour = removeContour(contour,row,col,i+1,j)
				elif (secondaryNeighbors == 1):
					if (contour[i+1,j-1] == True):
						contour = removeContour(contour,row,col,i+1,j-1)
					if (contour[i+1,j+1] == True):
						contour = removeContour(contour,row,col,i+1,j+1)
	elif (i == row-1):
		if (j == 0):
			primaryNeighbors = contour[i-1,j]*1+contour[i,j+1]*1
			secondaryNeighbors = contour[i-1,j+1]*1
			numNeighbors = primaryNeighbors+secondaryNeighbors
			if (numNeighbors <= 2):
				if (primaryNeighbors == 1 or primaryNeighbors == 2):
					if (contour[i-1,j] == True):
						contour = removeContour(contour,row,col,i-1,j)
					if (contour[i,j+1] == True):
						contour = removeContour(contour,row,col,i,j+1)
				elif (secondaryNeighbors == 1):
					if (contour[i-1,j+1] == True):
						contour = removeContour(contour,row,col,i-1,j+1)
		elif (j == col-1):
			primaryNeighbors = contour[i,j-1]*1+contour[i-1,j]*1
			secondaryNeighbors = contour[i-1,j-1]*1
			numNeighbors = primaryNeighbors+secondaryNeighbors
			if (numNeighbors <= 2):
				if (primaryNeighbors == 1 or primaryNeighbors == 2):
					if (contour[i,j-1] == True):
						contour = removeContour(contour,row,col,i,j-1)
					if (contour[i-1,j] == True):
						contour = removeContour(contour,row,col,i-1,j)
				elif (secondaryNeighbors == 1):
					if (contour[i-1,j-1] == True):
						contour = removeContour(contour,row,col,i-1,j-1)
		else:
			primaryNeighbors = contour[i,j-1]*1+contour[i,j+1]+contour[i-1,j]*1
			secondaryNeighbors = contour[i-1,j-1]*1+contour[i-1,j+1]*1
			numNeighbors = primaryNeighbors+secondaryNeighbors
			if (numNeighbors <= 2):
				if (primaryNeighbors == 1 or primaryNeighbors == 2):
					if (contour[i,j-1] == True):
						contour = removeContour(contour,row,col,i,j-1)
					if (contour[i,j+1] == True):
						contour = removeContour(contour,row,col,i,j+1)
					if (contour[i-1,j] == True):
						contour = removeContour(contour,row,col,i-1,j)
				elif (secondaryNeighbors == 1):
					if (contour[i-1,j-1] == True):
						contour = removeContour(contour,row,col,i-1,j-1)
					if (contour[i-1,j+1] == True):
						contour = removeContour(contour,row,col,i-1,j+1)
	elif (j == 0):
		primaryNeighbors = contour[i-1,j]*1+contour[i+1,j]*1+contour[i,j+1]*1
		secondaryNeighbors = contour[i-1,j+1]*1+contour[i+1,j+1]*1
		numNeighbors = primaryNeighbors+secondaryNeighbors
		if (numNeighbors <= 2):
			if (primaryNeighbors == 1 or primaryNeighbors == 2):
				if (contour[i-1,j] == True):
					contour = removeContour(contour,row,col,i-1,j)
				if (contour[i+1,j] == True):
					contour = removeContour(contour,row,col,i+1,j)
				if (contour[i,j+1] == True):
					contour = removeContour(contour,row,col,i,j+1)
			elif (secondaryNeighbors == 1):
				if (contour[i-1,j+1] == True):
					contour = removeContour(contour,row,col,i-1,j+1)
				if (contour[i+1,j+1] == True):
					contour = removeContour(contour,row,col,i+1,j+1)
	elif (j == col-1):
		primaryNeighbors = contour[i-1,j]*1+contour[i+1,j]*1+contour[i,j-1]*1
		secondaryNeighbors = contour[i-1,j-1]*1+contour[i+1,j-1]*1
		numNeighbors = primaryNeighbors+secondaryNeighbors
		if (numNeighbors <= 2):
			if (primaryNeighbors == 1 or primaryNeighbors == 2):
				if (contour[i-1,j] == True):
					contour = removeContour(contour,row,col,i-1,j)
				if (contour[i+1,j] == True):
					contour = removeContour(contour,row,col,i+1,j)
				if (contour[i,j-1] == True):
					contour = removeContour(contour,row,col,i,j-1)
			elif (secondaryNeighbors == 1):
				if (contour[i-1,j-1] == True):
					contour = removeContour(contour,row,col,i-1,j-1)
				if (contour[i+1,j-1] == True):
					contour = removeContour(contour,row,col,i+1,j-1)
	else:
		primaryNeighbors = contour[i,j-1]*1+contour[i,j+1]*1+contour[i-1,j]*1+contour[i+1,j]*1
		secondaryNeighbors = contour[i-1,j-1]*1+contour[i-1,j+1]*1+contour[i+1,j+1]*1+contour[i+1,j-1]*1
		numNeighbors = primaryNeighbors+secondaryNeighbors
		if (numNeighbors <= 2):
			if (primaryNeighbors == 1 or primaryNeighbors == 2):
				if (contour[i,j-1] == True):
					contour = removeContour(contour,row,col,i,j-1)
				if (contour[i,j+1] == True):
					contour = removeContour(contour,row,col,i,j+1)
				if (contour[i-1,j] == True):
					contour = removeContour(contour,row,col,i-1,j)
				if (contour[i+1,j] == True):
					contour = removeContour(contour,row,col,i+1,j)
			elif (secondaryNeighbors == 1):
				if (contour[i-1,j-1] == True):
					contour = removeContour(contour,row,col,i-1,j-1)
				if (contour[i-1,j+1] == True):
					contour = removeContour(contour,row,col,i-1,j+1)
				if (contour[i+1,j+1] == True):
					contour = removeContour(contour,row,col,i+1,j+1)
				if (contour[i+1,j-1] == True):
					contour = removeContour(contour,row,col,i+1,j-1)
	return contour


def removeBoundaryParticles(numpy.ndarray[numpy.uint8_t, ndim=2] bImg, bool flag=True):
	assert bImg.dtype == numpy.uint8
	cdef int row, col, numLabel, i, ii, jj, label

	if (flag==False):
		return bImg.astype('bool')

	row = bImg.shape[0]
	col = bImg.shape[1]

	cdef numpy.ndarray[numpy.int32_t, ndim=2] labelImg = numpy.zeros([row, col], dtype=numpy.int32)
	labelImg, numLabel = ndimage.label(bImg.astype('bool'))
	cdef numpy.ndarray[numpy.int32_t, ndim=1] bdryLabel = numpy.zeros([numLabel+1], dtype=numpy.int32)

	for i in range(row):
		bdryLabel[labelImg[i,0]] = 1
		bdryLabel[labelImg[i,col-1]] = 1
	for i in range(1,col-1):
		bdryLabel[labelImg[0,i]] = 1
		bdryLabel[labelImg[row-1,i]] = 1
	for i in range(row):
		for j in range(col):
			if (bdryLabel[labelImg[i,j]] == 1):
				bImg[i,j] = 0
	return bImg.astype('bool')


def areaThreshold(numpy.ndarray[numpy.uint8_t, ndim=2] bImg, numpy.ndarray[numpy.float64_t, ndim=1] areaRange, flag=True):
	assert bImg.dtype == numpy.uint8 and areaRange.dtype == numpy.float64
	cdef int i, j, numLabel, row, col

	if (flag==False):
		return bImg.astype('bool')

	row = bImg.shape[0]
	col = bImg.shape[1]

	cdef numpy.ndarray[numpy.int32_t, ndim=2] labelImg = numpy.zeros([row, col], dtype=numpy.int32)
	labelImg, numLabel = ndimage.label(bImg,structure=[[1,1,1],[1,1,1],[1,1,1]])
	cdef numpy.ndarray[numpy.int32_t, ndim=1] area = numpy.zeros([numLabel+1], dtype=numpy.int32)

	for i in range(row):
		for j in range(col):
			area[labelImg[i,j]] += 1
	for i in range(row):
		for j in range(col):
			if (area[labelImg[i,j]] < areaRange[0] or area[labelImg[i,j]] > areaRange[1]):
				bImg[i,j] = 0
	return bImg.astype('bool')


def circularThreshold(numpy.ndarray[numpy.uint8_t, ndim=2] bImg, numpy.ndarray[numpy.float64_t, ndim=1] circularityRange, flag=True):
	assert bImg.dtype == numpy.uint8 and circularityRange.dtype == numpy.float64
	cdef int i, numLabel, row, col, area
	cdef double circularity, pmeter

	if (flag==False):
		return bImg.astype('bool')

	row = bImg.shape[0]
	col = bImg.shape[1]

	cdef numpy.ndarray[numpy.int32_t, ndim=2] labelImg = numpy.zeros([row, col], dtype=numpy.int32)
	cdef numpy.ndarray[numpy.uint8_t, ndim=2] bImgLabelN = numpy.zeros([row, col], dtype=numpy.uint8)


	labelImg, numLabel = ndimage.label(bImg,structure=[[1,1,1],[1,1,1],[1,1,1]])
	for i in range(1, numLabel+1):
		bImgLabelN = (labelImg == i).astype('uint8')
		area = bImgLabelN.sum()
		pmeter = measure.perimeter(bImgLabelN)
		circularity = (4*numpy.pi*area)/(pmeter**2)
		if (circularity > 1):
			circularity = 1.0/circularity
		if (circularity<circularityRange[0] or circularity>circularityRange[1]):
			bImg -= bImgLabelN
	return bImg.astype('bool')


def dimensionThreshold(numpy.ndarray[numpy.uint8_t, ndim=2] bImg, numpy.ndarray[numpy.float64_t, ndim=1] widthRange, numpy.ndarray[numpy.float64_t, ndim=1] heightRange):
	assert bImg.dtype == numpy.uint8 and widthRange.dtype == numpy.float64 and heightRange.dtype == numpy.float64
	cdef int i, j, numLabel, row, col, width, height

	row = bImg.shape[0]
	col = bImg.shape[1]

	cdef numpy.ndarray[numpy.int32_t, ndim=2] labelImg = numpy.zeros([row, col], dtype=numpy.int32)
	labelImg, numLabel = ndimage.label(bImg, structure=[[1,1,1],[1,1,1],[1,1,1]])

	for i in range(1, numLabel+1):
		bImgLabelN = (labelImg == i).astype('uint8')
		indices = numpy.nonzero(bImgLabelN)
		height = indices[0].max()-indices[0].min()
		width = indices[1].max()-indices[1].min()
		if (width<widthRange[0] or width>widthRange[1] or height<heightRange[0] or height>heightRange[1]):
			bImg-=bImgLabelN
	return bImg.astype('bool')


def encapsule(numpy.ndarray[numpy.uint8_t, ndim=2] bImg, numpy.ndarray[numpy.uint8_t, ndim=2] gImg, numpy.ndarray[numpy.int32_t, ndim=2] qArr, numpy.ndarray[numpy.int32_t, ndim=2] ringRange, mode='circle', int radius=5):
	assert bImg.dtype == numpy.uint8 and gImg.dtype == numpy.uint8 and qArr.dtype == numpy.int32 and ringRange.dtype == numpy.int32

	cdef int row, col, i, j, ii, jj, flag, counter = 0
	row = bImg.shape[0]
	col = bImg.shape[1]

	cdef numpy.ndarray[numpy.uint8_t, ndim=2] bImgEncapsulated = numpy.zeros([row, col], dtype=numpy.uint8)
	cdef numpy.ndarray[numpy.uint8_t, ndim=2] test = numpy.zeros([row, col], dtype=numpy.uint8)
	cdef numpy.ndarray[numpy.uint8_t, ndim=2] temp = numpy.zeros([row, col], dtype=numpy.uint8)

	if (mode == 'circle'):
		rowIndices = numpy.nonzero(numpy.logical_and(qArr>ringRange[0][0], qArr<ringRange[len(ringRange)-1][1]))[0]
		colIndices = numpy.nonzero(numpy.logical_and(qArr>ringRange[0][0], qArr<ringRange[len(ringRange)-1][1]))[1]
		for i, j in zip(rowIndices, colIndices):
#			cv.Circle(cv.fromarray(temp), (i,j), radius, 1, thickness=-1)
			flag = 0
			for ii in range(i-radius,i+radius+1):
				for jj in range(j-radius,j+radius+1):
					if (temp[jj,ii]):
						if (bImg[jj,ii] == 0):#!= temp[jj,ii]):
							flag = 1
			if (flag == 0):
				counter += 1
				for ii in range(i-radius,i+radius+1):
					for jj in range(j-radius,j+radius+1):
						if (temp[jj,ii]):
							bImgEncapsulated[jj,ii] = temp[jj,ii]
			temp[:] = 0
	return bImgEncapsulated

def createMask(numpy.ndarray[numpy.int32_t, ndim=2] qArr, numpy.ndarray[numpy.int32_t, ndim=2] ringRange):
	assert qArr.dtype == numpy.int32 and ringRange.dtype == numpy.int32
	cdef int i, j, k, row, col
	row = qArr.shape[0]; col = qArr.shape[1]
	cdef numpy.ndarray[numpy.uint8_t, ndim=2] mask = numpy.zeros([row, col], dtype=numpy.uint8)
	cdef numpy.ndarray[numpy.uint8_t, ndim=2] tempMask = numpy.ones([row, col], dtype=numpy.uint8)

	for k in range(len(ringRange)):
		tempMask[qArr < ringRange[k][0]] = 0
		tempMask[qArr > ringRange[k][1]] = 0
		mask = numpy.logical_or(mask, tempMask).astype('uint8')
		tempMask[:] = 1
	return mask


def thresholdAdaptive1(numpy.ndarray[numpy.uint8_t, ndim=2] img, numpy.ndarray[numpy.uint8_t,ndim=2] mask, method='none', bool excludeZero=False, int strelSize=0):
	assert img.dtype==numpy.uint8 and mask.dtype==numpy.uint8
	cdef int r, c, row, col
	cdef double threshold
	row = img.shape[0]; col = img.shape[1]
	cdef numpy.ndarray[numpy.uint8_t, ndim=2] bImg = numpy.zeros([row,col], dtype=numpy.uint8)
	cdef numpy.ndarray[numpy.uint8_t, ndim=2] padImg = numpy.zeros([row+2*strelSize,col+2*strelSize], dtype=numpy.uint8)
	cdef numpy.ndarray[numpy.uint8_t, ndim=2] imgROI = numpy.zeros([2*strelSize+1,2*strelSize+1], dtype=numpy.uint8)
	if (method not in ['mean','median','otsu','none']):
		print 'WARNING: THRESHOLDING METHOD ', method, ' IS NOT A VALID OPTION. USE ONE OF \'mean\',\'median\',\'otsu\',\'none\'. USING DEFAULT METHOD \'none\''
		return img
	if (method=='none'):
		return img
	else:
		padImg = numpy.pad(img,pad_width=strelSize,mode='reflect')
		for r in range(strelSize,row+strelSize):
			for c in range(strelSize,col+strelSize):
				if (mask[r-strelSize,c-strelSize]):
					imgROI = padImg[r-strelSize:r+strelSize+1,c-strelSize:c+strelSize+1]
					if (method=='mean'):
						if (excludeZero==True):
							threshold = numpy.mean(imgROI[imgROI>0])
						else:
							threshold = numpy.mean(imgROI)
						bImg[r-strelSize,c-strelSize] = (img[r-strelSize,c-strelSize]>=threshold)*1
					elif (method=='median'):
						if (excludeZero==True):
							threshold = numpy.median(imgROI[imgROI>0])
						else:
							threshold = numpy.median(imgROI)
						bImg[r-strelSize,c-strelSize] = (img[r-strelSize,c-strelSize]>=threshold)*1
					elif (method=='otsu'):
						if (excludeZero==True):
							threshold = skimage.filter.threshold_otsu(imgROI[imgROI>0])
						else:
							threshold = skimage.filter.threshold_otsu(imgROI)
						bImg[r-strelSize,c-strelSize] = (img[r-strelSize,c-strelSize]>=threshold)*1
	return bImg.astype('bool')


def thresholdAdaptive2(numpy.ndarray[numpy.uint8_t, ndim=2]img, numpy.ndarray[numpy.uint8_t,ndim=2] mask, method='none', bool excludeZero=False, int strelSize=0):
	assert img.dtype==numpy.uint8 and mask.dtype==numpy.uint8
	cdef int r, c, row, col
	cdef double threshold
	row = img.shape[0]; col = img.shape[1]
	cdef numpy.ndarray[numpy.uint8_t, ndim=2] bImg = numpy.zeros([row+2*strelSize,col+2*strelSize], dtype=numpy.uint8)
	cdef numpy.ndarray[numpy.uint8_t, ndim=2] padImg = numpy.zeros([row+2*strelSize,col+2*strelSize], dtype=numpy.uint8)
	cdef numpy.ndarray[numpy.uint8_t, ndim=2] imgROI = numpy.zeros([2*strelSize+1,2*strelSize+1], dtype=numpy.uint8)
	if (method not in ['mean','median','otsu','none']):
		print 'WARNING: THRESHOLDING METHOD ', method, ' IS NOT A VALID OPTION. USE ONE OF \'mean\',\'median\',\'otsu\',\'none\'. USING DEFAULT METHOD \'none\''
		return img
	if (method=='none'):
		return img
	else:
		padImg = numpy.pad(img,pad_width=strelSize,mode='reflect')
		for r in range(strelSize,row+strelSize):
			for c in range(strelSize,col+strelSize):
				if (mask[r-strelSize,c-strelSize]):
					imgROI = padImg[r-strelSize:r+strelSize+1,c-strelSize:c+strelSize+1]
					if (method=='mean'):
						if (excludeZero==True):
							threshold = numpy.mean(imgROI[imgROI>0])
						else:
							threshold = numpy.mean(imgROI)
						bImg[r-strelSize:r+strelSize+1,c-strelSize:c+strelSize+1]=(numpy.logical_or((bImg[r-strelSize:r+strelSize+1,c-strelSize:c+strelSize+1]).astype('bool'),imgROI>=threshold))*1
					elif (method=='median'):
						if (excludeZero==True):
							threshold = numpy.median(imgROI[imgROI>0])
						else:
							threshold = numpy.median(imgROI)
						bImg[r-strelSize:r+strelSize+1,c-strelSize:c+strelSize+1]=(numpy.logical_or((bImg[r-strelSize:r+strelSize+1,c-strelSize:c+strelSize+1]).astype('bool'),imgROI>=threshold))*1
					elif (method=='otsu'):
						if (excludeZero==True):
							threshold = skimage.filter.threshold_otsu(imgROI[imgROI>0])
						else:
							threshold = skimage.filter.threshold_otsu(imgROI)
						bImg[r-strelSize:r+strelSize+1,c-strelSize:c+strelSize+1]=(numpy.logical_or((bImg[r-strelSize:r+strelSize+1,c-strelSize:c+strelSize+1]).astype('bool'),imgROI>=threshold))*1
	return numpy.logical_and(bImg[strelSize:row+strelSize,strelSize:col+strelSize].astype('bool'), mask.astype('bool'))


###############################################################
def calculateMinMaxDist(numpy.ndarray[numpy.int_t, ndim=1] r1, numpy.ndarray[numpy.int_t, ndim=1] c1, numpy.ndarray[numpy.int_t, ndim=1] r2, numpy.ndarray[numpy.int_t, ndim=1] c2):
	assert r1.dtype == numpy.int and c1.dtype == numpy.int and r1.dtype == numpy.int and c2.dtype == numpy.int

	cdef double dist, dMin, dMax
	cdef int len1 = r1.shape[0]
	cdef int len2 = r2.shape[0]
	cdef int i, j

	for i in range(len1):
		for j in range(len2):
			dist = numpy.sqrt((r1[i]-r2[j])**2 + (c1[i]-c2[j])**2)
			if (i==0 and j==0):
				dMin=dist; dMax=dist
			if (dist<dMin):
				dMin=dist
			if (dist>dMax):
				dMax=dist
	return (dMin,dMax)
###############################################################


###############################################################
def threshold_kapur(numpy.ndarray[numpy.uint8_t, ndim=1] intensities):
	assert intensities.dtype==numpy.uint8

	cdef numpy.ndarray[numpy.double_t, ndim=1] entropyTotal = numpy.zeros([256], dtype=numpy.double)
	cdef double entropyLeft, entropyRight
	cdef int i, s, totalFreq, leftFreq, rightFreq

	freq, bin_edges = numpy.histogram(intensities, bins=range(0,257), normed=False)

	totalFreq = numpy.sum(freq)
	for s in range(0,256):
		entropyLeft=0.0
		entropyRight=0.0
		leftFreq = 0
		for i in range(0,s+1):
			leftFreq += freq[i]
		rightFreq = totalFreq-leftFreq

		if (leftFreq>0):
			for i in range(0,s+1):
				if (freq[i]>0):
					entropyLeft += 1.0*freq[i]/leftFreq * numpy.log(1.0*freq[i]/leftFreq)
		if (rightFreq>0):
			for i in range(s+1,256):
				if (freq[i]>0):
					entropyRight += 1.0*freq[i]/rightFreq * numpy.log(1.0*freq[i]/rightFreq)

		entropyRight=-entropyRight; entropyLeft=-entropyLeft
		entropyTotal[s] = entropyRight+entropyLeft

	return numpy.argmax(entropyTotal)
###############################################################


###############################################################
def createRGBImage(numpy.ndarray[numpy.double_t, ndim=2] hexSymScore, numpy.ndarray[numpy.double_t, ndim=2] intensity, numpy.ndarray[numpy.double_t, ndim=1] hueRange, numpy.ndarray[numpy.double_t, ndim=1] valRange, double minHex, double maxHex, double minVal, double maxVal):
	assert hexSymScore.dtype==numpy.double and intensity.dtype==numpy.double and hueRange.dtype==numpy.double and valRange.dtype==numpy.double
	cdef int row,col,ROW,COL,H,S
	cdef double V
	
	cdef double c,x,m,r0,b0,g0,h,s,v
	
	ROW, COL = hexSymScore.shape[0], hexSymScore.shape[1]
	
	cdef numpy.ndarray[numpy.uint8_t, ndim=3] RGBImage = numpy.zeros([ROW,COL,3], dtype=numpy.uint8)
	
	for row in range(ROW):
		for col in range(COL):
			if (intensity[row,col]==0):
				H,S,V = 0,0,0
				RGBImage[row,col,:] = [0,0,0]
			else:
				H = int((hexSymScore[row,col]-minHex)/(maxHex-minHex)*(hueRange[1]-hueRange[0]))+hueRange[0]
				S = 1
				V = (intensity[row,col]-minVal)/(maxVal-minVal)*(valRange[1]-valRange[0]) + valRange[0]
				
				h, s, v = numpy.double(H), numpy.double(S), numpy.double(V)
				if (h<0): h=0
				if (h>359): h=359
				if (s<0): s=0
				if (s>1): s=1
				if (v<0): v=0
				if (v>1): v=1
				
				c = v*s
				x = c*(1-numpy.abs((h/60.0)%2-1))
				m = v-c
				if (h>=0 and h<60):
					r0,g0,b0 = c,x,0
				elif (h>=60 and h<120):
					r0,g0,b0 = x,c,0
				if (h>=120 and h<180):
					r0,g0,b0 = 0,c,x
				if (h>=180 and h<240):
					r0,g0,b0 = 0,x,c
				if (h>=240 and h<300):
					r0,g0,b0 = x,0,c
				if (h>=300 and h<360):
					r0,g0,b0 = c,0,x
					
				RGBImage[row,col,:] = [numpy.int((r0+m)*255),numpy.int((g0+m)*255),numpy.int((b0+m)*255)]
	return RGBImage
###############################################################


###############################################################
def hsv2rgb(double h, double s,double v):
	cdef double c, x, m, r0, b0, g0, r, g, b
	
	if (h<0): h=0
	if (h>359): h=359
	if (s<0): s=0
	if (s>1): s=1
	if (v<0): v=0
	if (v>1): v=1
	
	c = v*s
	x = c*(1-numpy.abs((h/60.0)%2-1))
	m = v-c
	if (h>=0 and h<60):
		r0,g0,b0 = c,x,0
	elif (h>=60 and h<120):
		r0,g0,b0 = x,c,0
	if (h>=120 and h<180):
		r0,g0,b0 = 0,c,x
	if (h>=180 and h<240):
		r0,g0,b0 = 0,x,c
	if (h>=240 and h<300):
		r0,g0,b0 = x,0,c
	if (h>=300 and h<360):
		r0,g0,b0 = c,0,x
	r,g,b = r0+m,g0+m,b0+m
#	r,g,b = int(r*255),int(g*255),int(b*255)
	
	return (int(r*255),int(g*255),int(b*255))
###############################################################
