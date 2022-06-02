from xmlrpc.client import boolean
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

def get_cell_patches(im: np.ndarray, masks: np.ndarray, size=64):
    
    centers = get_cell_centers(masks)
    
    patches = extract_patches(im, masks, centers, size)
    
    return patches

        
def get_cell_centers(mask: np.ndarray) -> np.ndarray:
    
    centers = []  
    for n in tqdm(range(1,mask.max()+1)):
        tmp = np.copy(mask)
        tmp[mask != n] = 0
        tmp = tmp.astype(bool)
        y, x = np.where(tmp != 0)
        y_min, y_max = y.min(), y.max()+1
        x_min, x_max = x.min(), x.max()+1
        
        if y_min == 0 or y_max == mask.shape[0] or x_min==0 or x_max==mask.shape[1]:
            continue
        
        cell = tmp[y_min:y_max, x_min:x_max]
        y, x = calc_center(cell)
        y_center, x_center = np.round(y+y_min,0), np.round(x+x_min,0)
        centers.append([y_center, x_center, n])
        
    return np.array(centers).astype(np.uint16)
        
def calc_center(bin):
    
    M00 = np.sum(bin)
    M10 = np.sum(np.array(range(bin.shape[0])) * np.sum(bin, axis=1))
    M01 = np.sum(np.array(range(bin.shape[1])) * np.sum(bin, axis=0))
    
    return M10/M00, M01/M00

def extract_patches(im, masks, centers, size):
    
    if im.ndim == 2:
        tmp_im = np.pad(im, ((size//2, size//2),(size//2, size//2)), mode="constant")
        
    elif im.ndim == 3:
        tmp_im = np.pad(im, ((size//2, size//2),(size//2, size//2), (0,0)), mode="constant")
        
    tmp_masks = np.pad(masks, ((size//2, size//2),(size//2, size//2)), mode="constant")
    
    patches = []
    for y,x,n in tqdm(centers):
        
        y += size//2; x += size//2
        w_y, w_x = (y-size//2, y+size//2),(x-size//2, x+size//2)

        cell_mask = np.copy(tmp_masks[w_y[0]:w_y[1], w_x[0]:w_x[1]])
        cell_mask[cell_mask != n] = 0
        cell_mask = cell_mask.astype(bool)
        cell_mask = dilate_mask(cell_mask, 3)
        
        marker_im = np.copy(tmp_im[w_y[0]:w_y[1], w_x[0]:w_x[1], ...])
        marker_im[cell_mask == 0] = 0
        
        patches.append(marker_im)
        
    return np.array(patches).astype("float32")

        
def dilate_mask(mask, s=3):
    
    k = np.ones((s,s)).astype(np.uint8)
    ret = cv2.dilate(mask.astype(np.uint8), k)
    return ret.astype(bool)