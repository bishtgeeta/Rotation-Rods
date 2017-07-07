import numpy
import cv2
from scipy import ndimage
from skimage.morphology import disk, white_tophat
from mahotas.polygon import fill_convexhull
from skimage import measure
from skimage import feature
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time

class FindAngleHelper(object):
    """A helper class for finding slopes of two particles in an image
    It plots the input data and waits for the user to draw two lines
    in the image, the slopes of these two lines is saved in attributes
    `first_slope` and `second_slope` of the instance of the class
    
    Parameters: data: numpy array
    """
    def __init__(self, data, line_length=None):
        self.press = False
        self.data = data
        if line_length is None:
            self.line_length = data.shape[0]*1.0 / 3
        
        
        self.first_line = None
        self.first_slope = numpy.NaN
        self.second_slope = numpy.NaN
        self.intersection_angle = numpy.NaN
        
        self.figure = plt.figure()
        self.ax1 = plt.subplot2grid((5,2), (0,0), colspan=2, rowspan=4)
        self.ax1.imshow(self.data, cmap='gray')
        self.ax1.set_xlim([0, self.data.shape[1]])
        self.ax1.set_ylim([0, self.data.shape[0]])
        self.line,  = self.ax1.plot([1,1], [1,1])
        
        self.ax2 = plt.subplot2grid((5,2), (4,0))
        self.ax3 = plt.subplot2grid((5,2), (4,1))
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        self.ax3.set_xticks([])
        self.ax3.set_yticks([])
        plt.tight_layout()
        
        self.button_accept = Button(self.ax2, 'Accept')
        self.button_accept.on_clicked(self.accept)
        self.button_retry = Button(self.ax3, 'Try again')
        self.button_retry.on_clicked(self.retry)
        
 
    def on_click(self, event):
        self.press = True
        self.x0, self.y0 = event.xdata, event.ydata
        self.x1, self.y1 = event.xdata, event.ydata
        
    def on_release(self, event):
        self.press = False
        if (self.x0 - self.x1)**2 + (self.y0 - self.y1)**2  > self.line_length**2:
            self.ax1.plot([self.x0, self.x1], [self.y0, self.y1])
            if self.first_line is None:
                self.point1 = numpy.array([self.x0, self.y0])
                self.point2 = numpy.array([self.x1, self.y1])
                self.first_line = [self.x0, self.x1, self.y0, self.y1]
                self.first_slope = self._get_slope(self.first_line)
            else:
                self.point3 = numpy.array([self.x0, self.y0])
                self.point4 = numpy.array([self.x1, self.y1])
                self.second_line = [self.x0, self.x1, self.y0, self.y1]
                self.second_slope = self._get_slope(self.second_line)
        
    def on_move(self, event):
        if self.press:
            self.x1, self.y1 = event.xdata, event.ydata
            self.line.set_xdata([self.x0, self.x1])
            self.line.set_ydata([self.y0, self.y1])
            self.line.figure.canvas.draw()

    def connect(self):
        self.cidpress = self.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.cidrelease = self.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.figure.canvas.mpl_connect('motion_notify_event', self.on_move)

    def _get_slope(self, line):
        if line[1] == line[0]:
            return 90
        tan_theta = (line[3] - line[2]) * 1.0 / (line[1] - line[0])
        return numpy.rad2deg(numpy.arctan(tan_theta))
        
    def _get_intersection_angle(self):
        try:
            ray1 = self.point2 - self.point1
            ray2 = self.point4 - self.point3
        except AttributeError:
            return numpy.NaN
        cos_theta = numpy.dot(ray1, ray2) / (numpy.linalg.norm(ray1) * numpy.linalg.norm(ray2))
        theta = numpy.rad2deg(numpy.arccos(cos_theta))
        self.intersection_angle_sign = numpy.sign(numpy.cross(ray1, ray2))
        if theta < 0:
            self.intersection_angle = self.intersection_angle_sign * (180 + theta)
        else:
            self.intersection_angle = self.intersection_angle_sign * theta
            
        return self.intersection_angle

    
    def accept(self, event):
        _ = self._get_intersection_angle()
        time.sleep(0.5)
        plt.close('all')
        
    def retry(self, event):
        self.first_line = None
        self.ax1.clear()
        self.ax1.imshow(self.data, cmap='gray')
        self.ax1.set_xlim([0, self.data.shape[1]])
        self.ax1.set_ylim([0, self.data.shape[0]])
        self.line, = self.ax1.plot([1,1], [1,1])
        
        
#############################################################################
#For Automatically Calculating Angle for Rods and Bipyramids
##############################################################################        
        
        
def get_intersection_point(center1, orientation1, center2, orientation2):
    m1 = numpy.tan(orientation1)
    m2 = numpy.tan(orientation2)
    point1 = center1 + numpy.array([1, m1])
    point2 = center2 + numpy.array([1, m2])
    
    da = numpy.array([1, m1])
    db = numpy.array([1, m2])
    dp = numpy.array(center1) - numpy.array(center2)
    
    dap = numpy.array([-m1, 1])
    denom = numpy.dot(dap, db)
    num = numpy.dot( dap, dp)
    
    return (num / denom.astype(float))*db + center2

    
    

def get_intersection_angle(center1, orientation1, center2, orientation2):
    orientation1 = numpy.deg2rad(orientation1)
    orientation2 = numpy.deg2rad(orientation2)
    intersection_point = get_intersection_point(center1, orientation1, 
                                                  center2, orientation2)

    ray1 = intersection_point - center1
    ray2 = intersection_point - center2
    cos_theta = numpy.dot(ray1, ray2) / (numpy.linalg.norm(ray1) * numpy.linalg.norm(ray2))
    theta = numpy.rad2deg(numpy.arccos(cos_theta))
    sign = numpy.sign(numpy.cross(ray1, ray2))
    
    if theta < 0:
        return sign * (180 + theta)
    else:
        return sign * theta    
#######################################################################
# NORMALIZE AN 8 BIT GRAYSCALE IMAGE
#######################################################################
def normalize(gImg, min=0, max=255):
    if (gImg.max() > gImg.min()):
        gImg = 1.0*(max-min)*(gImg - gImg.min())/(gImg.max() - gImg.min())
        gImg=gImg+min
    elif (gImg.max() > 0):
        gImg[:] = max
    gImg=gImg.astype('uint8')
    return gImg
#######################################################################


#######################################################################
# INVERT A GRAYSCALE IMAGE
#######################################################################
def invertImage(gImg):
    return 255-gImg
#######################################################################


#######################################################################
# PERFORM BACKGROUND SUBTRACTION USING THE TOP-HAT TRANSFORM
#######################################################################
def subtractBackground(gImg,sigma,radius):
    gImgBlur = ndimage.gaussian_filter(gImg, sigma=sigma)
    gImgTHT=white_tophat(gImgBlur, selem=disk(radius))
    gImg = normalize(gImgTHT)
    return gImg
#######################################################################


#######################################################################
# PERFORM 3D FILTERS ON IMAGE STACK
#######################################################################
def filter3D(gImgStack,size=3,method='mean',mode='reflect'):
    if (method=='mean'):
        return ndimage.uniform_filter(gImgStack,size=size,mode=mode)
    elif (method=='median'):
        return ndimage.median_filter(gImgStack,size=size,mode=mode)
    elif (method=='gauss'):
        return ndimage.gaussian_filter(gImgStack,sigma=size,mode=mode)
#######################################################################



#######################################################################
# FILL UP THE HOLES IN A BINARY IMAGE
#######################################################################
def fillHoles(bImg):
    return ndimage.binary_fill_holes(bImg)
#######################################################################


#######################################################################
# DRAW A CONVEX HULL AND FILL IT AROUND INDIVIDUAL CONNECTED OBJECTS
#######################################################################
def convexHull(bImg):
    label,numLabel=ndimage.label(bImg)
    bImg[:]=False
    for i in range(1,numLabel+1):
        bImgN=label==i
        bImgN=fill_convexhull(bImgN)
        bImg=numpy.logical_or(bImg,bImgN)
    return bImg
#######################################################################


#######################################################################
# FIND OUT THE BOUNDARY OF CONNECTED OBJECTS IN A BINARY IMAGE
#######################################################################
def boundary(bImg):
    bImgErode = ndimage.binary_erosion(bImg)
    bImgBdry = (bImg - bImgErode).astype('bool')
    return bImgBdry
#######################################################################


#######################################################################
# BINARY OPENING OPERATION
#######################################################################
def binary_opening(bImg, iterations=1):
    bImg = ndimage.binary_erosion(bImg, iterations=iterations)
    bImg = ndimage.binary_dilation(bImg, iterations=iterations)
    return bImg
#######################################################################


#######################################################################
# BINARY CLOSING OPERATION
#######################################################################
def binary_closing(bImg, iterations=1):
    bImg = ndimage.binary_dilation(bImg, iterations=iterations)
    bImg = ndimage.binary_erosion(bImg, iterations=iterations)
    return bImg
#######################################################################


#######################################################################
# CANNY EDGE DETECTION
#######################################################################
def canny_detection(gImgProc, sigma=6):
    bImg = feature.canny(gImgProc, sigma=sigma)
    return bImg
#######################################################################


#######################################################################
# BINARY DILATION OPERATION
#######################################################################
def binary_dilation(bImg, iterations=1):
    bImg = ndimage.binary_dilation(bImg, iterations=iterations)
    return bImg
#######################################################################


#######################################################################
# BINARY EROSION OPERATION
#######################################################################
def binary_erosion(bImg, iterations=1):
    bImg = ndimage.binary_erosion(bImg, iterations=iterations)
    return bImg
#######################################################################


def labelRegionProps(labelImg, structure=[[1,1,1],[1,1,1],[1,1,1]], area=False, perimeter=False, circularity=False, orientation=False, pixelList=False, bdryPixelList = False, centroid=False, intensityList=False, sumIntensity=False, avgIntensity=False, maxIntensity=False, effRadius=False, radius=False, theta=False, rTick=False, qTick=False, circumRadius=False, inRadius=False, radiusOFgyration=False, rTickMMM=False, thetaMMM=False):
    labels = numpy.unique(labelImg)[1:]
    numLabel = labels[-1]
    
    dictionary = {}
    dictionary['id'] = []
    if (area == True):
        dictionary['area'] = []
    if (perimeter == True):
        dictionary['perimeter'] = []
    if (circularity == True):
        dictionary['circularity'] = []
    if (orientation == True):
        dictionary['orientation'] = []
    if (pixelList == True):
        dictionary['pixelList'] = []
    if (bdryPixelList == True):
        dictionary['bdryPixelList'] = []
    if (centroid == True):
        dictionary['centroid'] = []
    if (intensityList == True):
        dictionary['intensityList'] = []
    if (sumIntensity == True):
        dictionary['sumIntensity'] = []
    if (avgIntensity == True):
        dictionary['avgIntensity'] = []
    if (maxIntensity == True):
        dictionary['maxIntensity'] = []
    if (effRadius == True):
        dictionary['effRadius'] = []
    if (radius == True):
        dictionary['radius'] = []
    if (circumRadius == True):
        dictionary['circumRadius'] = []
    if (inRadius == True):
        dictionary['inRadius'] = []
    if (radiusOFgyration == True):
        dictionary['radiusOFgyration'] = []
    if (rTick == True):
        dictionary['rTick'] = []
    if (qTick == True):
        dictionary['qTick'] = []
    if (rTickMMM == True):
        dictionary['rTickMean'] = []
        dictionary['rTickMin'] = []
        dictionary['rTickMax'] = []
    if (theta == True):
        dictionary['theta'] = []
    if (thetaMMM == True):
        dictionary['thetaMean'] = []
        dictionary['dThetaP'] = []
        dictionary['dThetaM'] = []
        
    for label in labels:
        bImgLabelN = labelImg == label
        dictionary['id'].append(label)
        if (area == True):
            Area = bImgLabelN.sum()
            dictionary['area'].append(Area)
        if (centroid == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = [numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1])]
            dictionary['centroid'].append(centerRC)
            
    return labelImg, numLabel, dictionary

#######################################################################
# FIND OUT THE REGION PROPERTIES OF CONNECTED OBJECTS IN A BINARY IMAGE
#######################################################################
def regionProps(bImg, gImg=0, structure=[[1,1,1],[1,1,1],[1,1,1]], area=False, perimeter=False, circularity=False, orientation=False, pixelList=False, bdryPixelList = False, centroid=False, intensityList=False, sumIntensity=False, avgIntensity=False, maxIntensity=False, effRadius=False, radius=False, theta=False, rTick=False, qTick=False, circumRadius=False, inRadius=False, radiusOFgyration=False, rTickMMM=False, thetaMMM=False):
    [labelImg, numLabel] = ndimage.label(bImg, structure=structure)
    [row, col] = bImg.shape
    dictionary = {}
    dictionary['id'] = []
    if (area == True):
        dictionary['area'] = []
    if (perimeter == True):
        dictionary['perimeter'] = []
    if (circularity == True):
        dictionary['circularity'] = []
    if (orientation == True):
        dictionary['orientation'] = []
    if (pixelList == True):
        dictionary['pixelList'] = []
    if (bdryPixelList == True):
        dictionary['bdryPixelList'] = []
    if (centroid == True):
        dictionary['centroid'] = []
    if (intensityList == True):
        dictionary['intensityList'] = []
    if (sumIntensity == True):
        dictionary['sumIntensity'] = []
    if (avgIntensity == True):
        dictionary['avgIntensity'] = []
    if (maxIntensity == True):
        dictionary['maxIntensity'] = []
    if (effRadius == True):
        dictionary['effRadius'] = []
    if (radius == True):
        dictionary['radius'] = []
    if (circumRadius == True):
        dictionary['circumRadius'] = []
    if (inRadius == True):
        dictionary['inRadius'] = []
    if (radiusOFgyration == True):
        dictionary['radiusOFgyration'] = []
    if (rTick == True):
        dictionary['rTick'] = []
    if (qTick == True):
        dictionary['qTick'] = []
    if (rTickMMM == True):
        dictionary['rTickMean'] = []
        dictionary['rTickMin'] = []
        dictionary['rTickMax'] = []
    if (theta == True):
        dictionary['theta'] = []
    if (thetaMMM == True):
        dictionary['thetaMean'] = []
        dictionary['dThetaP'] = []
        dictionary['dThetaM'] = []
        
    for i in range(1, numLabel+1):
        bImgLabelN = labelImg == i
        dictionary['id'].append(i)
        if (area == True):
            Area = bImgLabelN.sum()
            dictionary['area'].append(Area)
        if (perimeter == True):
            pmeter = measure.perimeter(bImgLabelN)
            dictionary['perimeter'].append(pmeter)
        if (circularity == True):
            Area = bImgLabelN.sum()
            pmeter = measure.perimeter(bImgLabelN)
            circlarity = (4*numpy.pi*Area)/(pmeter**2)
            if (circlarity>1):
                circlarity=1-(circularity-1)
            dictionary['circularity'].append(circlarity)
        if (orientation == True):
            regions = measure.regionprops(bImgLabelN.astype('uint8'))
            for props in regions:
                dictionary['orientation'].append(numpy.rad2deg(props.orientation))
        if (pixelList == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            dictionary['pixelList'].append([pixelsRC[0].tolist(),pixelsRC[1].tolist()])
        if (bdryPixelList == True):
            bdry = boundary(bImgLabelN)
            pixelsRC = numpy.nonzero(bdry)
            dictionary['bdryPixelList'].append([pixelsRC[0].tolist(),pixelsRC[1].tolist()])
        if (centroid == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = [numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1])]
            dictionary['centroid'].append(centerRC)
        if (intensityList == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            intensities = gImg[pixelsRC]
            dictionary['intensityList'].append(intensities)
        if (sumIntensity == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            sumInt = numpy.sum(gImg[pixelsRC])
            dictionary['sumIntensity'].append(sumInt)
        if (avgIntensity == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            avgInt = numpy.mean(gImg[pixelsRC])
            dictionary['avgIntensity'].append(avgInt)
        if (maxIntensity == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            maxInt = numpy.max(gImg[pixelsRC])
            dictionary['maxIntensity'].append(maxInt)
        if (radius == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = (numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1]))
            radii = numpy.max(numpy.sqrt((pixelsRC[0]-centerRC[0])**2 + (pixelsRC[1]-centerRC[1])**2))
            dictionary['radius'].append(radii)
        if (effRadius == True):
            Area = bImgLabelN.sum()
            effRadii = numpy.sqrt(Area/numpy.pi)
            dictionary['effRadius'].append(effRadii)
        if (circumRadius == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = (numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1]))
            bdryPixelsRC = numpy.nonzero(boundary(bImgLabelN))
            radii = numpy.max(numpy.sqrt((bdryPixelsRC[0]-centerRC[0])**2 + (bdryPixelsRC[1]-centerRC[1])**2))
            dictionary['circumRadius'].append(radii)
        if (inRadius == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = (numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1]))
            bdryPixelsRC = numpy.nonzero(boundary(bImgLabelN))
            radii = numpy.min(numpy.sqrt((bdryPixelsRC[0]-centerRC[0])**2 + (bdryPixelsRC[1]-centerRC[1])**2))
            dictionary['inRadius'].append(radii)
        if (radiusOFgyration == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = (numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1]))
            gyration = numpy.sqrt(numpy.average((pixelsRC[0]-centerRC[0])**2 + (pixelsRC[1]-centerRC[1])**2))
            dictionary['radiusOFgyration'].append(gyration)
        if (rTick == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = [numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1])]
            center = [row/2,col/2]
            if (row<=col):
                sc = 1.0*col/row
                qArrScale = col
                dist = numpy.sqrt((sc*(centerRC[0]-center[0]))**2 + (centerRC[1]-center[1])**2)
            else:
                sc = 1.0*row/col
                qArrScale = row
                dist = numpy.sqrt((centerRC[0]-center[0])**2 + (sc*(centerRC[1]-center[1]))**2)
            if (dist==0):
                rTck = 0
            else:
                rTck = qArrScale/dist
            dictionary['rTick'].append(rTck)
        if (qTick == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = [numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1])]
            center = [row/2,col/2]
            if (row<=col):
                sc = 1.0*col/row
                qArrScale = col
                qTck = numpy.sqrt((sc*(centerRC[0]-center[0]))**2 + (centerRC[1]-center[1])**2)
            else:
                sc = 1.0*row/col
                qArrScale = row
                qTck = numpy.sqrt((centerRC[0]-center[0])**2 + (sc*(centerRC[1]-center[1]))**2)
            dictionary['qTick'].append(qTck)
        if (rTickMMM == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = [numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1])]
            center = [row/2,col/2]
            if (row<=col):
                sc = 1.0*col/row
                qArrScale = col
                dist = numpy.sqrt((sc*(centerRC[0]-center[0]))**2 + (centerRC[1]-center[1])**2)
                distAll = numpy.sqrt((sc*(pixelsRC[0]-center[0]))**2 + (pixelsRC[1]-center[1])**2)
            else:
                sc = 1.0*row/col
                qArrScale = row
                dist = numpy.sqrt((centerRC[0]-center[0])**2 + (sc*(centerRC[1]-center[1]))**2)
                distAll = numpy.sqrt((pixelsRC[0]-center[0])**2 + (sc*(pixelsRC[1]-center[1]))**2)
            if (dist==0):
                rTck = 0
            else:
                rTck = qArrScale/dist
            rTckAll = qArrScale/distAll
            dictionary['rTickMean'].append(rTck)
            dictionary['rTickMin'].append(rTckAll.min())
            dictionary['rTickMax'].append(rTckAll.max())
        if (theta == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = (numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1]))
            center = [row/2,col/2]
            angle = numpy.arctan2(center[0]-centerRC[0], centerRC[1]-col/2.0)*180/numpy.pi
            if (angle<0):
                angle = 360+angle
            dictionary['theta'].append(angle)
        if (thetaMMM == True):
            pixelsRC = numpy.nonzero(bImgLabelN)
            centerRC = [numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1])]
            center = [row/2,col/2]
            angle = numpy.arctan2(center[0]-centerRC[0], centerRC[1]-center[1])*180/numpy.pi
            angleAll = numpy.arctan2(center[0]-pixelsRC[0], pixelsRC[1]-center[1])*180/numpy.pi
            if (angle<0):
                angle = 360+angle
            for i in range(len(angleAll)):
                if (angleAll[i]<0):
                    angleAll[i] = 360+angleAll[i]
            if (numpy.max([angleAll<10])==True and numpy.max(angleAll>350)==True):
                if (angle < 180):
                    dThetaP = numpy.max(angleAll[angleAll<180])-angle
                    dThetaM = angle-(numpy.min(angleAll[angleAll>180])-360)
                else:
                    dThetaP = numpy.max(angleAll[angleAll<180])-(angle-360)
                    dThetaM = angle-numpy.min(angleAll[angleAll>180])
            else:
                dThetaP = numpy.max(angleAll)-angle
                dThetaM = angle-numpy.min(angleAll)
            dictionary['thetaMean'].append(angle)
            dictionary['dThetaP'].append(dThetaP)
            dictionary['dThetaM'].append(dThetaM)
    return labelImg, numLabel, dictionary
#######################################################################


#######################################################################
# AREA THRESHOLD FOR LABELLED PARTICLES
#######################################################################
def areaThresholdLabels(labelImg, areaRange):
    particleList = numpy.unique(labelImg[1:])
    for particle in particleList:
        area = numpy.sum(labelImg==particle)
        if not(area>=areaRange[0] and area<=areaRange[1]):
            labelImg[labelImg==particle]=0
    return labelImg
#######################################################################


#######################################################################
# CIRCULARITY THRESHOLD FOR LABELLED PARTICLES
#######################################################################
def circularThresholdLabels(labelImg, circularRange):
    particleList = numpy.unique(labelImg[1:])
    for particle in particleList:
        bImgLabelN = labelImg==particle
        circularity = 0
        area = bImgLabelN.sum()
        perimeter = measure.perimeter(bImgLabelN)
        if (perimeter>0):
            circularity = (4*numpy.pi*area)/(perimeter**2)
        if (circularity>1):
            circularity=1/circularity
        if not(circularity>=circularRange[0] and circularity<=circularRange[1]):
            labelImg[bImgLabelN]=0
    return labelImg
#######################################################################


#######################################################################
# WRITE TEXT ON RGB IMAGE
#######################################################################
def textOnRGBImage(RGBImg, text, position, fontScale=1, color=(255,0,0), thickness=1):
    cv2.putText(RGBImg, text, (position[1],position[0]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=color, thickness=thickness, bottomLeftOrigin=False)
    return RGBImg
#######################################################################


#######################################################################
# WRITE TEXT ON GRAYSCALE IMAGE
#######################################################################
def textOnGrayImage(gImg, text, position, fontScale=1, color=127, thickness=1):
    cv2.putText(gImg, text, (position[1],position[0]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=color, thickness=thickness, bottomLeftOrigin=False)
    return gImg
#######################################################################


#######################################################################
# CONVERT A GRAYSCALE IMAGE TO RGB
#######################################################################
def gray2rgb(gImg):
    [row,col] = gImg.shape
    rgbImg = numpy.zeros([row, col, 3], dtype='uint8')
    rgbImg[:,:,0] = gImg
    rgbImg[:,:,1] = gImg
    rgbImg[:,:,2] = gImg
    return rgbImg
#######################################################################


#######################################################################
# CONVERT A RGB IMAGE TO BGR
#######################################################################
def RGBtoBGR(rgbImg):
    bgrImg = rgbImg.copy()
    bgrImg[:,:,0] = rgbImg[:,:,2]
    bgrImg[:,:,2] = rgbImg[:,:,0]
    return bgrImg
#######################################################################
