import cv2
import numpy
from numpy import isfinite
import os.path
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.interpolate import interp1d

sys.path.append(os.path.abspath('../myFunctions'))
#import myPythonFunc
import plotFunc

import matplotlib.patches as mpatches


##############################################################
def power(x,p0,p1) :
    # A power law fit with:
    #   Normalization                : p0
    #   Offset                       : p1
    #   Constant                     : p2
    #return p0*(x)**-1+p1
    return p0*(x)**-1
##############################################################

##############################################################
def line(x,m):
    return m*x
##############################################################



class trajectoryAnalysis(object):
    ##############################################################
    def __init__(self, fileName, outputDir, pixInNM, imgRow, imgCol, skiprows=0, measureInPix=False):
        print "READING THE ODS DATA FILES"
        self.fileName = fileName
        self.outputDir = outputDir
        self.pixInNM = pixInNM
        self.imgRow = imgRow
        self.imgCol = imgCol
        self.numMeasures = 5
        self.data = numpy.loadtxt(fileName,skiprows=skiprows)
        [self.row,self.col] = self.data.shape
        for colNum in range(self.col):
            if 0 < numpy.isnan(self.data[:, colNum]).sum() < self.data.shape[0]:
                self.data[:, colNum] = self.fill_nan(self.data[:, colNum])
        self.numParticles = (self.col-1)/self.numMeasures
        self.particleList = range(1,self.numParticles+1)
        if (measureInPix==True):
            self.pixel2nm()
        self.fps = int(1.0/(self.data[1,0]-self.data[0,0]))
        self.frameList = (self.fps*self.data[:,0]).astype('int')
    ##############################################################
    
    @staticmethod
    def fill_nan(x):
        ind = numpy.where(~numpy.isnan(x))[0]
        vals = x[ind]
        interp = interp1d(ind, vals, kind='linear')
        nan_ind = numpy.where(numpy.isnan(x))[0]
        nan_vals = interp(nan_ind)
        x[nan_ind] = nan_vals
        return x
    
    ##############################################################
    def pixel2nm(self):
        print "CONVERTING DATA IN PIXELS TO NM"
        fps=10.0
        self.data[:,0] /= fps
        for particle in self.particleList:
            self.data[:,(particle-1)*6+1] *= self.pixInNM
            self.data[:,(particle-1)*6+2] *= self.pixInNM
            self.data[:,(particle-1)*6+3] *= self.pixInNM**2
            self.data[:,(particle-1)*6+4] *= self.pixInNM
            self.data[:,(particle-1)*6+6] *= self.pixInNM
    ##############################################################
    
    
    ##############################################################
    def plotXY(self, particleList=-1):
        print "PLOTTING XY VALUES OF ALL PARTICLES"
        if (particleList==-1):
            particleList = self.particleList
        fig = plt.figure(figsize=(3,2))
        ax = fig.add_axes([0,0,1,1])
        for particle in particleList:
            x=self.data[:,(particle-1)*self.numMeasures+1]; y=self.data[:,(particle-1)*self.numMeasures+2]; t=self.data[:,0]
            x=x[numpy.isfinite(x)]; y=y[numpy.isfinite(y)]; t=t[numpy.isfinite(x)]
            ax.plot(t,x,label='x')
            ax.plot(t,y,label='y')
        ax.set_xlim([0,1.0*self.row/self.fps])
        ax.set_ylim([0,self.pixInNM*self.imgRow])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (nm)')
        ax.legend(loc=1)
        plt.savefig(self.outputDir+'/xy.png', format='png')
        plt.savefig(self.outputDir+'/xy.pdf', format='pdf')
        plt.close()
    ##############################################################
    
    
    ##############################################################
    def plotVelocity(self, particleList=-1):
        print "PLOTTING SPEED OF ALL PARTICLES"
        if (particleList==-1):
            particleList = self.particleList
        fig = plt.figure(figsize=(3,2))
        ax = fig.add_axes([0,0,1,1])
        for particle in particleList:
            x=self.data[:,(particle-1)*self.numMeasures+1]; y=self.data[:,(particle-1)*self.numMeasures+2]; t=self.data[:,0]
            x=x[numpy.isfinite(x)]; y=y[numpy.isfinite(y)]; t=t[numpy.isfinite(x)]
            dx, dy, dt = numpy.diff(x,n=1), numpy.diff(y,n=1), numpy.diff(t,n=1)
            dS = numpy.sqrt(dx**2+dy**2)
            v = dS/dt
            ax.plot(t[:-1],v,color='#FF0000')
        ax.set_xlim([0,1.0*self.row/self.fps])
        ax.set_xlabel(r'Time (s)')
        ax.set_ylabel(r'Speed (nm/s)')
        plt.savefig(self.outputDir+'/speed.png', format='png')
        plt.savefig(self.outputDir+'/speed.pdf', format='pdf')
        plt.close()
    ##############################################################
    
    
    ##############################################################
    def plotTrajectory(self, particleList=-1, offset=False):
        print "PLOTTING TRAJECTORY OF ALL PARTICLES"
        if (particleList==-1):
            particleList = self.particleList
        if (offset==True):
            fig = plt.figure(figsize=(2,2))
            ax = fig.add_axes([0,0,1,1])
            for particle in particleList:
                x=self.data[:,(particle-1)*self.numMeasures+1]; y=self.data[:,(particle-1)*self.numMeasures+2]
                x=x[numpy.isfinite(x)]; y=y[numpy.isfinite(y)]
                if (x.shape[0]>0):
                    x=x-x[0]; y=y-y[0]
                    ax.plot(x,y,color='#000000')
            ax.plot(0,0,marker='o',color='#FF0000')
            ax.set_xlabel('X (nm)')
            ax.set_ylabel('Y (nm)')
            plt.savefig(self.outputDir+'/trajectoryOffset.png', format='png')
            plt.savefig(self.outputDir+'/trajectoryOffset.pdf', format='pdf')
            plt.close()
        elif (offset==False):
            fig = plt.figure(figsize=(2,2))
            ax = fig.add_axes([0,0,1,1])
            for particle in particleList:
                x=self.data[:,(particle-1)*self.numMeasures+1]; y=self.data[:,(particle-1)*self.numMeasures+2]
                x=x[numpy.isfinite(x)]; y=y[numpy.isfinite(y)]
                ax.plot(x,y)
            ax.set_xlim([0,self.pixInNM*self.imgCol])
            ax.set_ylim([0,self.pixInNM*self.imgRow])
            ax.set_xlabel('X (nm)')
            ax.set_ylabel('Y (nm)')
            plt.savefig(self.outputDir+'/trajectory.png', format='png')
            plt.savefig(self.outputDir+'/trajectory.pdf', format='pdf')
            plt.close()
    ##############################################################
    
    
    ##############################################################
    def plotTrajectoryCMAP(self, gImg, frame=1, particleList=-1, image=True):
        print "PLOTTING TRAJECTORY OF ALL PARTICLES STARTING FROM FIRST FRAME"
        if (particleList==-1):
            particleList = self.particleList
        fig = plt.figure(figsize=(1.1, 1.1))
        ax= fig.add_axes([0,0,1,1])
        if (image==True):
            ax.imshow(gImg, extent=[0,self.pixInNM*self.imgCol,0,self.pixInNM*self.imgRow])
            for particle in particleList:
                x=self.data[frame-1:,(particle-1)*self.numMeasures+1]; y=self.data[frame-1:,(particle-1)*self.numMeasures+2]
                #x=x[numpy.isfinite(x)]; y=y[numpy.isfinite(y)]
                if (numpy.isfinite(x[0]) and numpy.isfinite(y[0])):
                    plotFunc.colorline(ax,x,y)
        elif (image==False):
            for particle in particleList:
                x=self.data[frame-1:,(particle-1)*self.numMeasures+1]; y=self.data[frame-1:,(particle-1)*self.numMeasures+2]
                if (numpy.isfinite(x[0]) and numpy.isfinite(y[0])):
                    plotFunc.colorline(ax,x,y)
            ax.set_xlim([0,self.pixInNM*self.imgCol])
            ax.set_ylim([0,self.pixInNM*self.imgRow])
        
        image = numpy.zeros([256,2])
        for i in range(2):
            image[:,i] = numpy.linspace(0,255,256)
        ax2 = fig.add_axes([1.2,0.65,0.04,0.15])
        ax2.imshow(image, cmap='jet_r')
        plotFunc.set_axes_width(ax2, width=0)
        plotFunc.set_ticks_off(ax2)
        ax2.set_xlabel(str(1.0/self.fps)+' s')
        ax2.set_title(str(1.0*self.row/self.fps)+' s')
        ax.set_xlabel(r'X (nm)')
        ax.set_ylabel(r'Y (nm)')
        plt.savefig(self.outputDir+'/trajectoryCMAP(small).png', format='png')
        plt.savefig(self.outputDir+'/trajectoryCMAP(small).pdf', format='pdf')
        plt.close()
    ##############################################################
    
    
    ##############################################################
    def plotTrajectory3d(self, particleList=-1):
        print "PLOTTING 3D TRAJECTORY OF ALL PARTICLES"
        if (particleList==-1):
            particleList = self.particleList
        fig = plt.figure(figsize=(2,2))
        ax = plt.axes(projection='3d')
        for particle in particleList:
            x=self.data[:,(particle-1)*self.numMeasures+1]; y=self.data[:,(particle-1)*self.numMeasures+2]; t=self.data[:,0]
            x=x[numpy.isfinite(x)]; y=y[numpy.isfinite(y)]; t=t[numpy.isfinite(y)]
            ax.plot(x,y,t)
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_zlabel('Time (s)')
        ax.set_xlim([0,self.pixInNM*self.imgCol])
        ax.set_ylim([0,self.pixInNM*self.imgRow])
        ax.set_zlim([0.1,self.row/self.fps])
        plt.savefig(self.outputDir+'/trajectory3d.png', format='png')
        plt.savefig(self.outputDir+'/trajectory3d.pdf', format='pdf')
        plt.close()
    ##############################################################
    
    
    ##############################################################
    def calculateMSD(self, particleList=-1):
        print "CALCULATING MSD OF ALL PARTICLES"
        if (particleList==-1):
            particleList = self.particleList
        self.MSDDict={}
        for particle in self.particleList:
            self.MSDDict[particle] = {}
            self.MSDDict[particle]['MSD'] = []
            self.MSDDict[particle]['time'] = []
            
        for particle in particleList:
            x=self.data[:,(particle-1)*self.numMeasures+1]; y=self.data[:,(particle-1)*self.numMeasures+2]
            for dt in range(self.row):
                S=0; N=0
                for t in range(self.row-dt):
                    if (isfinite(x[t+dt]) and isfinite(x[t]) and isfinite(y[t+dt]) and isfinite(y[t])):
                        ds = (x[t+dt]-x[t])**2 + (y[t+dt]-y[t])**2
                        S+=ds; N+=1
                if (N>0):
                    self.MSDDict[particle]['MSD'].append(S/N)
                    self.MSDDict[particle]['time'].append(1.0*dt/self.fps)
    ##############################################################
    
    
    ##############################################################
    def plotMSD(self, particleList=-1):
        print "PLOTTING MSD OF ALL PARTICLES"
        if (particleList==-1):
            particleList = self.particleList
        fig = plt.figure(figsize=(3,2))
        ax = fig.add_axes([0,0,1,1])
        print self.particleList
        for particle in self.particleList:
            dT = numpy.asarray(self.MSDDict[particle]['time'])
            MSD = numpy.asarray(self.MSDDict[particle]['MSD'])
            MSD = MSD[dT<=10]
            dT = dT[dT<=10]
            m = 0
            [params, pcov] = optimize.curve_fit(line, dT, MSD, [m])
            print params
            ax.plot(dT,MSD)
            #ax.plot(self.MSDDict[particle]['time'],self.MSDDict[particle]['MSD'])
        ax.set_xlim(0,10)
        #ax.set_ylim(0,1000)
        ax.set_xlabel(r'$\Delta$T (s)')
        ax.set_ylabel(r'MSD $(nm^2)$')
        plt.savefig(self.outputDir+'/MSD.png', format='png')
        plt.savefig(self.outputDir+'/MSD.pdf', format='pdf')
        plt.close()
    ##############################################################
    
    
    ##############################################################
    def calculateDiffusion(self, particleList=-1, dimensions=2, percentile=90):
        print "CALCULATING DIFFUSION COEFFICIENT FOR ALL PARTICLES"
        if (particleList==-1):
            particleList = self.particleList
        tau = 1.0/self.fps; self.minRadius = 1E10; self.maxRadius=-1
        self.diffusionDict = {}
        for particle in particleList:
            self.diffusionDict[particle] = {}
            self.diffusionDict[particle]['diffusion'] = []
            self.diffusionDict[particle]['radius'] = []
        for particle in particleList:
            x=self.data[:,(particle-1)*self.numMeasures+1]; y=self.data[:,(particle-1)*self.numMeasures+2]; t=self.data[:,0]; radius=self.data[:,(particle-1)*self.numMeasures+6]
            for r in range(self.row-1):
                if (numpy.isfinite(x[r+1]) and numpy.isfinite(x[r]) and numpy.isfinite(y[r+1]) and numpy.isfinite(y[r])):
                    S = (x[r+1]-x[r])**2 + (y[r+1]-y[r])**2
                    diffusionCoeff = S/(2*dimensions*tau)
                    avgRadius = (radius[r+1]+radius[r])/2.0
                    if (avgRadius<self.minRadius):
                        self.minRadius=avgRadius
                    if (avgRadius>self.maxRadius):
                        self.maxRadius=avgRadius
                    self.diffusionDict[particle]['diffusion'].append(diffusionCoeff)
                    self.diffusionDict[particle]['radius'].append(avgRadius)
        for particle in particleList:
            if (self.diffusionDict[particle]['diffusion']):
                diffusionArr = numpy.asarray(self.diffusionDict[particle]['diffusion'])
                radiusArr = numpy.asarray(self.diffusionDict[particle]['radius'])
                diffPercentile = numpy.percentile(diffusionArr, percentile)
                tempDiffArr = diffusionArr[diffusionArr<=diffPercentile]
                tempRadiusArr = radiusArr[diffusionArr<=diffPercentile]
                self.diffusionDict[particle]['diffusion'] = tempDiffArr.tolist()
                self.diffusionDict[particle]['radius'] = tempRadiusArr.tolist()
    ##############################################################
    
    
    ##############################################################
    def fitDiffusion(self, particleList=-1):
        print "FITTING THE DIFFUSION COEFFICIENT TO 'p0/R'"
        if (particleList==-1):
            particleList = self.particleList
        diffusionList, radiusList = [], []
        for particle in particleList:
            diffusionList += self.diffusionDict[particle]['diffusion']
            radiusList += self.diffusionDict[particle]['radius']
        diffusionArr = numpy.asarray(diffusionList)
        radiusArr = numpy.asarray(radiusList)
        
        #INITIAL GUESS
        p0 = diffusionArr[radiusArr==radiusArr.min()][0]
        p1 = 0
        
        #FITTING WITH POWER LAW WITH ALPHA=-1
        flag = True
        try:
            [params, pcov] = optimize.curve_fit(power, radiusArr, diffusionArr, [p0,p1])
        except:
            flag = False
            print "FITTING DOES NOT CONVERGE"
            
        if (flag==True):
            x = numpy.linspace(self.minRadius,self.maxRadius,100)
            #x = numpy.linspace(0.1,6,1000)
            y = params[0]*(x)**-1+params[1]
            print params[0], params[0]/2.0
            
        #x = x[y<=0.2]
        #y = y[y<=0.2]
            
        fig = plt.figure(figsize=(3,2))
        ax = fig.add_axes([0,0,1,1])
        ax.scatter(radiusArr, diffusionArr, s=1, facecolor='#000000', edgecolor='#000000')
        ax.plot(x,y,color='r',linewidth=1)
        ax.set_xlim(18,26)
        ax.set_ylim(0,10)
        ax.set_xlabel(r'R (nm)')
        ax.set_ylabel(r'$D_{MSD}(nm^2/s)$')
        plt.savefig(self.outputDir+'/scatterdiffusionFit.png', format='png')
        plt.savefig(self.outputDir+'/scatterdiffusionFit.pdf', format='pdf')
        plt.close()
        print "FIT PARAMETERS ARE: ", params
        return x,y, params
    ##############################################################
    
    
    ##############################################################
    def plotDiffusionMeanWithError(self,particleList=-1):
        print "CALCULATING MEAN AND DEVIATION OF DIFFUSION COEFFICIENT FOR RADIUS BINS"
        if (particleList==-1):
            particleList = self.particleList
        diffusionList = []
        radiusList = []
        self.radiusBins = numpy.arange(0,6.1,0.5)
        for r in range(1,len(self.radiusBins)):
            rMin, rMax = self.radiusBins[r-1], self.radiusBins[r]
            tempDiffusionList, tempRadiusList = [], []
            for particle in particleList:
                for i in range(len(self.diffusionDict[particle]['radius'])):
                    radius = self.diffusionDict[particle]['radius'][i]
                    diffusion = self.diffusionDict[particle]['diffusion'][i]
                    if (i==len(self.radiusBins)-1):
                        if (radius>=rMin and radius<=rMax):
                            tempDiffusionList.append(diffusion)
                            tempRadiusList.append(radius)
                    else:
                        if (radius>=rMin and radius<rMax):
                            tempDiffusionList.append(diffusion)
                            tempRadiusList.append(radius)
            diffusionList.append(tempDiffusionList)
            radiusList.append(tempRadiusList)
            
        x,y,params = self.fitDiffusion()
        fig = plt.figure(figsize=(3,2))
        ax = fig.add_axes([0,0,1,1])
        for diffusion, radius in zip(diffusionList, radiusList):
            if (diffusion):
                meanR = numpy.mean(radius);    sigmaR = numpy.std(radius)
                meanD = numpy.mean(diffusion); sigmaD = numpy.std(diffusion)
                percentile_75 = numpy.percentile(diffusion, 75)
                percentile_25 = numpy.percentile(diffusion, 25)
                ax.plot([meanR-sigmaR,meanR+sigmaR],[meanD,meanD],color='k')
                #ax.plot([meanR,meanR],[meanD-sigmaD,meanD+sigmaD],color='#0000FF')
                ax.plot([meanR,meanR],[percentile_25,percentile_75],color='k')
                #ax.scatter(meanR,meanD,s=25,color='#606060',edgecolor='k')
                ax.plot(meanR,meanD,ls='None',marker='o',markersize=6,color='#c0bfbf')
        ax.plot(x,y,color='r',linewidth=1)
        ax.set_xlabel(r'R (nm)')
        ax.set_ylabel(r'$D_{MSD}(nm^2/s)$')
        ax.set_xlim(0,6)
        ax.set_ylim(0,.2)
        plt.savefig(self.outputDir+'/diffusionFit.png', format='png')
        plt.savefig(self.outputDir+'/diffusionFit.pdf', format='pdf')
        plt.close()
    ##############################################################
    
    
    ##############################################################
    def plotAvgRadiusvsTime(self, particleList=-1):
        print "CALCULATING THE AVERAGE RADIUS OF ALL PARTICLES PRESENT IN THE FRAME AND PLOTTING IT AGAINST TIME"
        if (particleList==-1):
            particleList = self.particleList
        
        x = self.data[:,0]
        for particle in particleList:
            if (particle == particleList[0]):
                radius=self.data[:,(particle-1)*self.numMeasures+6]
            else:
                radius = numpy.column_stack((radius,self.data[:,(particle-1)*self.numMeasures+6]))
        avgRadius, stdRadius = numpy.zeros(self.row), numpy.zeros(self.row)
        for r in range(self.row):
            tempRad = radius[r,:]
            avgRadius[r] = numpy.mean(tempRad[numpy.isfinite(tempRad)])
            stdRadius[r] = numpy.std(tempRad[numpy.isfinite(tempRad)])
        
        fig = plt.figure(figsize=(3,2))
        ax = fig.add_axes([0,0,1,1])
        ax.plot(x,avgRadius)
        ax.fill_between(x,avgRadius-stdRadius,avgRadius+stdRadius,alpha=0.25)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'$R_{eff}$ (nm)')
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(0,30)
        plt.savefig(self.outputDir+'/AvgRadiusvsTime.png', format='png')
        plt.show()
    ##############################################################
    
    
    ##############################################################
    def plotAvgAreavsTime(self, particleList=-1):
        print "CALCULATING THE AVERAGE AREA OF ALL PARTICLES PRESENT IN THE FRAME AND PLOTTING IT AGAINST TIME"
        if (particleList==-1):
            particleList = self.particleList
        
        x = self.data[:,0]
        for particle in particleList:
            if (particle == particleList[0]):
                area=self.data[:,(particle-1)*self.numMeasures+3]
            else:
                area = numpy.column_stack((area,self.data[:,(particle-1)*self.numMeasures+3]))
        avgArea, stdArea = numpy.zeros(self.row), numpy.zeros(self.row)
        for r in range(self.row):
            tempArea = area[r,:]
            avgArea[r] = numpy.mean(tempArea[numpy.isfinite(tempArea)])
            stdArea[r] = numpy.std(tempArea[numpy.isfinite(tempArea)])
        fig = plt.figure(figsize=(3,2))
        ax = fig.add_axes([0,0,1,1])
        ax.plot(x,avgArea)
        ax.fill_between(x,avgArea-stdArea,avgArea+stdArea,alpha=0.25)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'Area ($nm^2$)')
        ax.set_xlim(x[0], x[-1])
        plt.savefig(self.outputDir+'/AvgAreavsTime.png', format='png')
        plt.close()
    ##############################################################
    
    
    ##############################################################
    def plotParticleRadiusvsTime(self, particleList=-1):
        print "PLOTTING THE RADIUS OF ALL PARTICLES AGAINST TIME"
        if (particleList==-1):
            particleList = self.particleList
        
        fig = plt.figure(figsize=(3,2))
        ax = fig.add_axes([0,0,1,1])
        x = self.data[:,0]
        for particle in particleList:
            radius = self.data[:,(particle-1)*self.numMeasures+6]
            ax.plot(x[numpy.isfinite(radius)],radius[numpy.isfinite(radius)])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'$R_{eff}$ (nm)')
        ax.set_xlim(x[0], x[-1])
        #ax.set_ylim(0,30)
        plt.savefig(self.outputDir+'/ParticleRadiusvsTime.png', format='png')
        plt.savefig(self.outputDir+'/ParticleRadiusvsTime.pdf', format='pdf')
        plt.close()
        
        
        #print "PLOTTING THE RADIUS OF ALL PARTICLES AGAINST TIME"
        #if (particleList==-1):
            #particleList = self.particleList
        
        #fig = plt.figure(figsize=(1.05,1.05))
        #ax = fig.add_axes([0,0,1,1])
        #x = self.data[:,0]
        #for particle in particleList:
            #radius = self.data[:,(particle-1)*6+6]
            #xTemp = x[x<=70]
            #radiusTemp = radius[x<=70]
            #ax.plot(xTemp[numpy.isfinite(radiusTemp)],radiusTemp[numpy.isfinite(radiusTemp)])
        #ax.set_xlabel('t (s)')
        #ax.set_ylabel(r'R (nm)')
        #ax.set_xlim(0,70)
        #ax.set_ylim(5,25)
        #ax.set_xticks([0,20,40,60])
        #ax.set_yticks([5,15,25])
        #plt.savefig(self.outputDir+'/ParticleRadiusvsTime.png', format='png')
        #plt.savefig(self.outputDir+'/ParticleRadiusvsTime.pdf', format='pdf')
        #plt.close()
    ##############################################################
    
    
    ##############################################################
    def plotParticleAreavsTime(self, particleList=-1):
        print "PLOTTING THE AREA OF ALL PARTICLES AGAINST TIME"
        if (particleList==-1):
            particleList = self.particleList
        
        fig = plt.figure(figsize=(3,2))
        ax = fig.add_axes([0,0,1,1])
        x = self.data[:,0]
        for particle in particleList:
            area = self.data[:,(particle-1)*6+3]
            ax.plot(x[numpy.isfinite(area)],area[numpy.isfinite(area)])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'Area ($nm^2$)')
        ax.set_xlim(x[0], x[-1])
        #ax.set_ylim(0,3000)
        plt.savefig(self.outputDir+'/ParticleAreavsTime.png', format='png')
        plt.savefig(self.outputDir+'/ParticleAreavsTime.pdf', format='pdf')
        plt.close()
    ##############################################################
    
    
    ##############################################################
    def plotNumberofPoints(self, particleList=-1):
        print "PLOTTING THE NUMBER OF PARTICLES FOR EACH TIME POINT"
        if (particleList==-1):
            particleList = self.particleList
        
        for particle in particleList:
            if (particle == particleList[0]):
                radius=self.data[:,(particle-1)*6+6]
            else:
                radius = numpy.column_stack((radius,self.data[:,(particle-1)*6+6]))
        numObs = numpy.zeros(self.row)
        for r in range(self.row):
            tempRad = radius[r,:]
            numObs[r] = len(tempRad[numpy.isfinite(tempRad)])
            
        fig = plt.figure(figsize=(3,2))
        ax = fig.add_axes([0,0,1,1])
        x = self.data[:,0]
        ax.plot(x,numObs)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'Pillar Count')
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(116,130)
        plt.savefig(self.outputDir+'/PillarCount.png', format='png')
        plt.close()
    ##############################################################
    
    
    ##############################################################
    def plotRadiusHistogram(self, particleList=-1, bins=10):
        '''
        bins = numpy.arange(self.minRadius,self.maxRadius+binSize,binSize)
        '''
        print "PLOTTING THE RADIUS HISTOGRAM"
        if (particleList==-1):
            particleList = self.particleList
        radiusList=[]
        for particle in particleList:
            radiusList += self.diffusionDict[particle]['radius']
            
        print min(radiusList), max(radiusList), len(radiusList)
        
        fig = plt.figure(figsize=(3,2))
        ax = fig.add_axes([0,0,1,1])
        n, bins, patches = ax.hist(radiusList, bins=bins)#, facecolor='#c0bfbf', edgecolor='#000000')
        self.radiusBins = bins
        ax.set_xlabel(r'R (nm)')
        ax.set_ylabel(r'Count')
        
        numpy.save(r'radiusList.npy', radiusList)
        
        #ax.set_xlim(bins.min(), bins.max())
        ax.set_xlim(bins.min(), bins.max())
        plt.savefig(self.outputDir+'/radiusHistogram.png', format='png')
        plt.savefig(self.outputDir+'/radiusHistogram.pdf', format='pdf')
        plt.close()
    ##############################################################
    
    
    ##############################################################
    def plotDiffusionHistogram(self, particleList=-1, bins=10):
        '''
        bins = numpy.arange(self.minDiffusion,self.maxDiffusion+binSize,binSize)
        '''
        print "PLOTTING THE DIFFUSION HISTOGRAM"
        if (particleList==-1):
            particleList = self.particleList
        diffusionList=[]
        for particle in particleList:
            diffusionList += self.diffusionDict[particle]['diffusion']
            
        print min(diffusionList), max(diffusionList)
        
        fig = plt.figure(figsize=(3,2))
        ax = fig.add_axes([0,0,1,1])
        n, bins, patches = ax.hist(diffusionList, bins=bins, facecolor='#fba910', edgecolor='#ed1c24')
        ax.set_xlabel(r'$D_{MSD}(nm^2/s)$')
        ax.set_ylabel(r'Count')
        ax.set_xlim(0,1)
        plt.savefig(self.outputDir+'/diffusionHistogram.png', format='png')
        plt.savefig(self.outputDir+'/diffusionHistogram.pdf', format='pdf')
        plt.close()
    ##############################################################
    
    
    ##############################################################
    def plotAverageDiffusionHistogram(self, particleList=-1, bins=10):
        '''
        bins = numpy.arange(self.minRadius,self.maxRadius+binSize,binSize)
        '''
        print "PLOTTING THE AVERAGE DIFFUSION HISTOGRAM"
        if (particleList==-1):
            particleList = self.particleList
        meanDiffusionList=[]
        for particle in particleList:
            if (self.diffusionDict[particle]['diffusion']):
                meanDiffusionList.append(numpy.mean(self.diffusionDict[particle]['diffusion']))
        print min(meanDiffusionList), max(meanDiffusionList)
        fig = plt.figure(figsize=(3,2))
        ax = fig.add_axes([0,0,1,1])
        n, buckets, patches = ax.hist(meanDiffusionList, bins=bins)#, facecolor='#fba919', edgecolor='#ed1c24')
        self.diffusionBins = bins
        #ax.set_xlim(0,0.02)
        ax.set_xlabel(r'$\bar{D}_{MSD} (nm^2/s)$')
        ax.set_ylabel(r'Particle Count')
        plt.savefig(self.outputDir+'/averageDiffusionHistogram.png', format='png')
        plt.savefig(self.outputDir+'/averageDiffusionHistogram.pdf', format='pdf')
        plt.close()
    ##############################################################
    
    
    ##############################################################
    def overlayImages(self, dataDir1, dataDir2, outputDir, frameList):
        print "OVERLAYING BINARY IMAGES OVER RAW IMAGES"
        fig = plt.figure(figsize=(2,2))
        ax = fig.add_axes([0,0,1,1])
        for frame in frameList:
            img1 = numpy.load(dataDir1+'/'+str(frame)+'.npy')
            img2 = numpy.load(dataDir2+'/'+str(frame)+'.npy')
            ax.imshow(img1, extent=[0,self.pixInNM*self.imgCol,0,self.pixInNM*self.imgRow])
            ax.imshow(img2, cmap='Blues', alpha=0.3, extent=[0,self.pixInNM*self.imgCol,0,self.pixInNM*self.imgRow])
            ax.set_title(str(1.0*frame/self.fps)+' s')
            ax.set_xlabel('x (nm)')
            ax.set_ylabel('y (nm)')
            plt.savefig(outputDir+'/'+str(frame)+'.png', format='png')
            ax.cla()
        plt.close()
    ##############################################################
    
    
    ##############################################################
    def etchRate(self, startTime=0.0, particleList=-1):
        if (particleList==-1):
            particleList = self.particleList
        x = self.data[:,0]
        rateList = []
        for particle in self.particleList:
            time = self.data[:,0]
            radius = self.data[:,(particle-1)*6+6]
            
            time = time[numpy.isfinite(radius)]
            radius = radius[numpy.isfinite(radius)]
            
            radius = radius[time>=startTime]
            time = time[time>=startTime]
            
            if (1.0*len(time)/self.row>0.4):
                p = numpy.polyfit(time,radius,1)
                rateList.append(-p[0])
        print "Average etch rate =", numpy.mean(rateList), ", ", len(rateList)
    ##############################################################
