from os import PathLike
import cv2
import numpy as np
import matplotlib.pyplot as plt

def uint8_to_float32(img):
    return (img/255).astype("float32")

def float32_to_uint8(img):
    return (img*255).astype("uint8")

def imread(p: PathLike) -> np.ndarray:
    
    im = cv2.imread(p)
    #im = normalize_image(im)
    return (im/im.max()).astype("float32")

def normalize_image(im: np.ndarray):

    im = (im-im.min())/(im.max()-im.min())
    return im

def Plot3D(im):
    
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    X = range(im.shape[0])
    Y = range(im.shape[1])
    X, Y = np.meshgrid(X, Y)
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, im, 50, cmap='binary')
    plt.show()
    
    
def gridPlot(ims, labels=None, target=None, sz=(10,10), vmin=0, vmax=1, save=None, plot=True):
    
    fig, axs = plt.subplots(sz[0], sz[1], figsize=(3*sz[0], 3*sz[1]))
    
    for n, (ax, im) in enumerate(zip(axs.ravel(), ims[:sz[0]*sz[1]])):
        ax.imshow(im, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        if isinstance(labels, (list, np.ndarray)) and isinstance(target, (list, np.ndarray)):
            ax.set_title([labels[n], target[n]], color="white")
        elif isinstance(labels, (list, np.ndarray)):
            ax.set_title(labels[n], color="white")
        else:
            ax.set_title(n)
        
    if isinstance(save, str):
        plt.savefig(save)
        plt.close(fig)
    if plot: 
        plt.show()
