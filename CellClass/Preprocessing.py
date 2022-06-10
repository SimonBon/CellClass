import cv2
import numpy as np
from skimage.morphology import reconstruction
from .utils import float32_to_uint8, uint8_to_float32

def h_dome(img, h=0.7):

    seed = img - h
    background = reconstruction(seed, img)
    hdome = img - background
    
    return hdome
    
def apply_clahe(img, clipLimit=6):

    img = float32_to_uint8(img)
    print(img.dtype, img.min(), img.max())
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
    equalized = clahe.apply(img)

    return uint8_to_float32(equalized)
    

def illumination(inp:np.ndarray, n=8) -> tuple:

    if not max(inp.shape) < 512:
        r = 512/max(inp.shape)
        ret = cv2.resize(inp, (int(r*inp.shape[0]), int(r*inp.shape[1])))
        print(f"Resizing to {ret.shape}")

    ret = np.copy(inp)

    for d_i in range(n):

        d_i=int((d_i//2)*2+1)
        tmp_2 = cv2.copyMakeBorder(ret, d_i, d_i, d_i, d_i, cv2.BORDER_REPLICATE)

        for x in range(d_i,ret.shape[0]+d_i):
            for y in range(d_i,ret.shape[1]+d_i):
                c_0 = (tmp_2[x-d_i,y-d_i] + tmp_2[x+d_i,y+d_i])/2
                c_1 = (tmp_2[x+d_i,y-d_i] + tmp_2[x-d_i,y+d_i])/2
                c_2 = (tmp_2[x-d_i,y] + tmp_2[x+d_i,y])/2
                c_3 = (tmp_2[x,y-d_i] + tmp_2[x,y+d_i])/2
                c_4 = tmp_2[x,y]

                tmp_2[x,y] = min([c_0, c_1, c_2, c_3, c_4])
                
        tmp_2 = cv2.GaussianBlur(tmp_2, ksize=(d_i, d_i), sigmaX=d_i, sigmaY=d_i)
        ret = tmp_2[d_i:ret.shape[0]+d_i, d_i:ret.shape[1]+d_i]
    
    ret = cv2.resize(ret, (inp.shape[1], inp.shape[0]))

    return inp-ret, ret
