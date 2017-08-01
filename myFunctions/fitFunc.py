import numpy
import cv2
from scipy import optimize
from numpy import isnan

numpy.seterr(over='raise')


################################################################################
def gaussSkewPlane(data, a0, a1, a2, height, muR, muC, sigmaR, sigmaC, rho):
    f = a0 + a1*data[:,0] + a2*data[:,1] + height*numpy.e**(-1.0/(2*(1-rho**2)) * (((data[:,0]-muR)/sigmaR)**2 + ((data[:,1]-muC)/sigmaC)**2 - 2*rho*(data[:,0]-muR)*(data[:,1]-muC)/(sigmaR*sigmaC)))
    return f
################################################################################


################################################################################
def initialGuess(data):
    data = data.astype('double')
    a0 = data[:,2].min()
    a1 = 0.0
    a2 = 0.0
    height = data[:,2].max()
    total = data[:,2].sum()
    muR = 1.0*(data[:,0]*data[:,2]).sum()/total
    muC = 1.0*(data[:,1]*data[:,2]).sum()/total
    subDataR = data[data[:,1]==int(muC),:]
    subDataC = data[data[:,0]==int(muR),:]
    sigmaR = numpy.sqrt(numpy.sum((subDataR[:,0]-muR)**2 * subDataR[:,2])/numpy.sum(subDataR[:,2]))
    sigmaC = numpy.sqrt(numpy.sum((subDataC[:,1]-muC)**2 * subDataC[:,2])/numpy.sum(subDataC[:,2]))
    rho = 0.0
    return a0,a1,a2,height,muR,muC,sigmaR,sigmaC,rho
################################################################################


################################################################################
def fitting(rawData,slope):
    [row,col] = rawData.shape
    mask = numpy.zeros([row,col],dtype='uint8')
    R,C = numpy.indices(rawData.shape)
    data = numpy.column_stack((numpy.ndarray.flatten(R), numpy.ndarray.flatten(C), numpy.ndarray.flatten(rawData)))
    flag = True
    error = 1e10
    k = 1
    
    [a0,a1,a2,height,muR,muC,sigmaR,sigmaC,rho] = initialGuess(data)
    if (slope<0):
        rho=-0.5
    elif (slope>0):
        rho=0.5
        
    try:
        [params, pcov] = optimize.curve_fit(gaussSkewPlane, data[:,:2], data[:,2], [a0,a1,a2,height,muR,muC,sigmaR,sigmaC,rho])
    except:
        flag = False
        
    if (flag==True):
        a0     = params[0]
        a1     = params[1]
        a2     = params[2]
        height = params[3]
        muR    = params[4]
        muC    = params[5]
        sigmaR = numpy.abs(params[6])
        sigmaC = numpy.abs(params[7])
        rho    = params[8]

        if (sigmaR<0.5 or sigmaC<0.5):
            flag=False
        else:
            fitData = a0 + a1*R + a2*C + height*numpy.e**(-1.0/(2*(1-rho**2)) * (((R-muR)/sigmaR)**2 + ((C-muC)/sigmaC)**2 - 2*rho*(R-muR)*(C-muC)/(sigmaR*sigmaC)))
            error = numpy.sqrt(numpy.sum((fitData-rawData)**2))
            
            covMatrix = numpy.array([[sigmaR**2, rho*sigmaR*sigmaC],[rho*sigmaR*sigmaC, sigmaC**2]])
            eigVals, eigVecs = numpy.linalg.eig(covMatrix)

            majorIndex = numpy.argmax(eigVals); majorAxis = numpy.sqrt(k * eigVals[majorIndex])*numpy.sqrt(2)
            minorIndex = numpy.argmin(eigVals); minorAxis = numpy.sqrt(k * eigVals[minorIndex])*numpy.sqrt(2)
            theta = numpy.arctan2(eigVecs[majorIndex][1], eigVecs[majorIndex][0])
            if (theta<0):
                theta += 2*numpy.pi
            theta -= numpy.pi/2
            
            if (isnan(muC) or isnan(muR) or isnan(majorAxis) or isnan(minorAxis) or isnan(theta)):
                pass
            else:
                cv2.ellipse(mask,center=(int(numpy.round(muC)),int(numpy.round(muR))),\
                            axes=(int(numpy.round(majorAxis)),int(numpy.round(minorAxis))),\
                            angle=numpy.rad2deg(theta),startAngle=0,endAngle=360,\
                            color=1,thickness=-1)
    return mask.astype('bool'), flag, error
    #muR, muC, majorAxis, minorAxis, theta
################################################################################
