import numpy
import cv2
from scipy import ndimage
from skimage.morphology import disk, white_tophat
from mahotas.polygon import fill_convexhull
from skimage import measure

##############################################################
##############################################################
def drawLine(bImg,x0,y0,m):
	[row,col] = bImg.shape
	line = numpy.zeros([row,col], dtype='bool')
	for x1 in range(col):
		y1 = int(m*(x1-x0)+y0)
		if (y1>=0 and y1<row):
			line[row-1-y1,x1] = True
	for y1 in range(row):
		x1 = int((y1-y0)/m+x0)
		if (x1>=0 and x1<col):
			line[row-1-y1,x1] = True
	return line
##############################################################
##############################################################
