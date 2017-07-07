import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.collections import LineCollection
import numpy

def cm2inch(x):
	return x*0.393701

def set_axes_color(ax, color='#000000'):
	ax.spines['bottom'].set_color(color)
	ax.spines['top'].set_color(color)
	ax.spines['left'].set_color(color)
	ax.spines['right'].set_color(color)
	return ax

def set_axes_width(ax, width=0.75):
	ax.spines['bottom'].set_linewidth(width)
	ax.spines['top'].set_linewidth(width)
	ax.spines['left'].set_linewidth(width)
	ax.spines['right'].set_linewidth(width)
	return ax

def set_tick_color(ax, color='#000000'):
	ax.tick_params(axis='x', colors='#000000')
	ax.tick_params(axis='y', colors='#000000')
	return ax
	
def set_axes_off(ax):
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	return ax
	
def set_label_color(ax, color='#000000'):
	ax.yaxis.label.set_color('#000000')
	ax.xaxis.label.set_color('#000000')
	return ax

def set_title_color(ax, color='#000000'):
	ax.title.set_color('red')
	return ax

def set_tickLabels_off(ax):
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	return ax

def set_ticks_off(ax):
	ax.set_xticks([])
	ax.set_yticks([])
	return ax

def set_minorTicks_on(ax):
	ax.xaxis.set_minor_locator(AutoMinorLocator(5))
	ax.yaxis.set_minor_locator(AutoMinorLocator(5))
	return ax


# Data manipulation:

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = numpy.array([x, y]).T.reshape(-1, 1, 2)
    segments = numpy.concatenate([points[:-1], points[1:]], axis=1)

    return segments

def colorline(ax, x, y, z=None, cmap='jet', norm=plt.Normalize(0.0, 1.0), linewidth=0.75, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = numpy.linspace(0.0, 1.0, len(x))
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input
        z = numpy.array([z])
    z = numpy.asarray(z)

    segments = make_segments(x, y)
    #for s in segments:
		#print s
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    ax.add_collection(lc)
    return lc



###############################################################
###############################################################
## PLOTTING
###############################################################
###############################################################
#kB=1.38e-23
#T=300
#R=4e-9
#density=19300
#A=numpy.pi*R**2
#m=4./3*numpy.pi*R**3
#D1=2e-20; D2=2e-20

#viscosity1=(kB*T)/(6*numpy.pi*D1*R)
#viscosity2=(kB*T)/(6*numpy.pi*D2*R)

#N1=6.023e23; V1=18e-6
#N2=60; V2=1e-27
##viscosity1=1568813.010477254; viscosity2=15688130.10477254

#rho1=N1/V1; rho2=N2/V2
#Fd1=rho1*kB*T*A; Fd2=rho2*kB*T*A

#c1_1=Fd1/(6*numpy.pi*viscosity1*R)
#c2_1=(6*numpy.pi*viscosity1*R)/m

#c1_2=Fd2/(6*numpy.pi*viscosity2*R)
#c2_2=(6*numpy.pi*viscosity2*R)/m

#t1=numpy.linspace(1e-30,1e-23,10000)#1e-23,10000)
##t2=numpy.linspace(1e-22,1e-20,1000)
#t=t1
##t=numpy.concatenate((t1,t2))

#v1=c1_1*(1-numpy.exp(1)**(-c2_1*t))*1e9
#v2=c1_2*(1-numpy.exp(1)**(-c2_2*t))*1e9


#fig = plt.figure()
#ax1 = fig.add_subplot(111)
##ax.set_xlim(-2,0); ax.set_ylim(0,5)
##ax.set_ylim(0,0.5e-20)

#ax1.plot(t,v1,color='#FF0000',alpha=1,linewidth=2.0,label='Water')
#ax1.plot(t,v2,color='#000000',alpha=1,linewidth=2.0,label='1mM CTAB')
#ax1.set_xlim(0,1e-23)

##ax2=fig.add_subplot(212)
##ax2.plot(numpy.log10(t),numpy.log10(v1),color='#FF0000',alpha=1,linewidth=2.0,label='Water')
##ax2.plot(numpy.log10(t),numpy.log10(v2),color='#000000',alpha=1,linewidth=2.0,label='1mM CTAB')
###ax2.set_ylim(-6,-14)

##final = numpy.column_stack((t,v1,v2))
##numpy.savetxt('terminalVelocity.csv', final, fmt='%.18e', delimiter=',', newline='\n')


##x=[]; xerr=[]; y=[]; yerr=[]
##for col in range(cols3):
	##x1=data3[:,col]; x2=data1[:,col]
	##x1=myPythonFunc.percentileRange(x1,min=25,max=75); x2=myPythonFunc.percentileRange(x2,min=25,max=75)

	##mu=myPythonFunc.average(x1)
	##sigma=myPythonFunc.variance(x1)
	##x.append(mu)
	##xerr.append(0)
	###xerr.append(sigma)

	##mu=myPythonFunc.average(x2)
	##sigma=myPythonFunc.variance(x2)
	##y.append(mu)
	##yerr.append(0)
	###yerr.append(sigma)

##ax.plot(x,y,marker='o',markersize=20,ls='none',color='#0000FF',fillstyle='none',label='Water')

##x=[]; xerr=[]; y=[]; yerr=[]
##for col in range(cols4):
	##x1=data4[:,col]; x2=data2[:,col]
	##x1=myPythonFunc.percentileRange(x1,min=25,max=75); x2=myPythonFunc.percentileRange(x2,min=25,max=75)

	##mu=myPythonFunc.average(x1)
	##sigma=myPythonFunc.variance(x1)
	##x.append(mu)
	##xerr.append(0)
	###xerr.append(sigma)

	##mu=myPythonFunc.average(x2)
	##sigma=myPythonFunc.variance(x2)
	##y.append(mu)
	##yerr.append(0)
	###yerr.append(sigma)
##ax.plot(x,y,marker='o',markersize=20,ls='none',color='#FF00CC',fillstyle='none',label='1mM CTAB')


##ax.plot(t,v1,color='#0000FF',alpha=1,linewidth=2.0,linestyle='-',label='Water')
##boxes=[data1]#,data7]
##ax.hist(data1[:], 7, facecolor='#00FFFF', alpha=0.5,label='Water')
##ax.hist(data2[:], 10, facecolor='#FFCCCC', alpha=0.5,label='1mM CTAB')
##ax.boxplot(boxes)


##ax.plot(data1[:,0],data1[:,1],marker='o',markersize=20,ls='none',color='#0000FF',fillstyle='none',label='Water')
##ax.plot(data1[:,0],data1[:,2],marker='o',markersize=20,ls='none',color='#0000FF',fillstyle='none')
##ax.plot(data1[:,3],data1[:,4],marker='o',markersize=20,ls='none',color='#FF00CC',fillstyle='none',label='1mM CTAB')
##ax.plot(data1[:,3],data1[:,5],marker='o',markersize=20,ls='none',color='#FF00CC',fillstyle='none')

##for col in range(2,cols1-1):
	##ax.plot(data1[:,0],data1[:,col],color='#00FFFF',alpha=0.7,linewidth=1.0,linestyle='-')
##ax.plot(data1[:,0],data1[:,cols1-1],color='#00FFFF',alpha=0.7,linewidth=1.0,linestyle='-',label='Water')
##ax.plot(data1[:,0],data1[:,1],color='#0000FF',linewidth=2.0,linestyle='-')
##for col in range(2,cols2-1):
	##ax.plot(data2[:,0],data2[:,col],color='#FFCCCC',alpha=0.7,linewidth=1.0,linestyle='-')
##ax.plot(data2[:,0],data2[:,cols2-1],color='#FFCCCC',alpha=0.7,linewidth=1.0,linestyle='-',label='1mM CTAB')
##ax.plot(data2[:,0],data2[:,1],color='#FF00CC',linewidth=2.0,linestyle='-')

##mean1=numpy.zeros([rows1],dtype='float64'); mean2=numpy.zeros([rows2],dtype='float64')
##for row in range(rows1):
	##mean1[row]=myPythonFunc.average(data1[row,1:])
##for row in range(rows2):
	##mean2[row]=myPythonFunc.average(data2[row,1:])
##ax.plot(data1[:,0],mean1,color='#0000FF',linewidth=2.0,linestyle='-')
##ax.plot(data2[:,0],mean2,color='#FF00CC',linewidth=2.0,linestyle='-')

#ax1.set_xlabel(r'T (s)'); ax1.set_ylabel(r'$v$ (nm/s)')
##ax2.set_xlabel(r'log(T)'); ax2.set_ylabel(r'log($v$)')
##ax.set_ylabel(r'MSD (nm$^{2}$')
##plt.subplots_adjust(bottom=0.25)
###############################################################
###############################################################
##ax1.legend(loc=4)
##ax2.legend(loc=4)

#plt.show()
###############################################################
###############################################################
