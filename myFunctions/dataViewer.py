import numpy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

def displayImageStack(gImgStack):
    [row,col,numFrames] = gImgStack.shape
    
    fig = plt.figure(figsize=(10,5))
    axImage = fig.add_axes([0.20,0.20,0.60,0.60])
    axSlide = fig.add_axes([0.1,0.1,0.8,0.05])
    frame=1
    im = axImage.imshow(gImgStack[:,:,frame-1])
    axImage.set_xticks([]), axImage.set_yticks([])
    sFrame = Slider(axSlide, 'Frame', 1, numFrames, valinit=1)
    
    def update(val):
        frame = sFrame.val
        im.set_data(gImgStack[:,:,frame-1])
        im.axes.figure.canvas.draw()
        #axImage.imshow(gImgStack[:,:,frame-1])
        #fig.canvas.draw_idle()
    sFrame.on_changed(update)
    
    plt.show()
    return 0
