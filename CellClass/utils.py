from os import PathLike
import cv2
import numpy as np

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