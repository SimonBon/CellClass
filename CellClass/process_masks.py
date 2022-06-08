from xmlrpc.client import boolean
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from CellClass.MCImage import MCImage
from typing import Union

def get_cell_patches(MCIm: MCImage, masks: np.ndarray, channels=["B", "G", "R"], size=64):
    
    centers = get_cell_centers(masks)
    
    patches = extract_patches(MCIm, masks, centers, size, channels)
    
    return patches

        
def get_cell_centers(mask: np.ndarray) -> np.ndarray:
    
    centers = []  
    for n in tqdm(range(1,mask.max()+1)):
        tmp = np.copy(mask)
        tmp[mask != n] = 0
        tmp = tmp.astype(float)
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
    
    M = cv2.moments(bin)
    
    return M["m10"]/M["m00"], M["m01"]/M["m00"]
    
    # M00 = np.sum(bin)
    # M10 = np.sum(np.array(range(bin.shape[0])) * np.sum(bin, axis=1))
    # M01 = np.sum(np.array(range(bin.shape[1])) * np.sum(bin, axis=0))
    # return M10/M00, M01/M00

def extract_patches(MCIm, masks, centers, size, channels):
    
    if len(channels) == 1:
        im = getattr(MCIm, channels[0])
    else:
        im = np.stack([getattr(MCIm, x) for x in channels], axis=-1)
        
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
        marker_all = np.copy(marker_im)
        marker_im[cell_mask == 0] = 0
        
        if cell_mask.any():
            patch = Patch(cell_mask, marker_im, marker_all, channels, y, x, n)
            patches.append(patch)
        
    return patches
        
def dilate_mask(mask, s=3):
    
    k = np.ones((s,s)).astype(np.uint8)
    ret = cv2.dilate(mask.astype(np.uint8), k)
    return ret.astype(bool)


class Patch():
    
    def __init__(self, mask, masked, not_masked, channels, y_pos, x_pos, idx):
        
        for n, c in enumerate(channels):
            setattr(self, c + "_masked", masked[..., n])
            setattr(self, c + "_not_masked", not_masked[..., n])
           
        if hasattr(self, "R_masked") and hasattr(self, "G_masked") and hasattr(self, "B_masked"):
            self.RGB_masked = np.stack((self.R_masked, self.G_masked, self.B_masked), axis=-1)
            self.RGB_not_masked = np.stack((self.R_not_masked, self.G_not_masked, self.B_not_masked), axis=-1)
            self.overlay = np.clip(self.RGB_not_masked + np.stack((0.2*mask, 0.2*mask, 0.2*mask), axis=-1), 0 ,1)
        
        else:
            self.masked = masked
            self.not_masked = not_masked
           
        self.shape = masked.shape         
        self.mask = mask
        self.y_pos = y_pos
        self.x_pos = x_pos
        self.idx = idx
        
        self.y_size ,self.x_size = self.get_size()

        self.area = np.sum(mask)

    def get_size(self):
        
        y, x = np.where(self.mask != 0)
        
        y_min, y_max = y.min(), y.max()+1
        x_min, x_max = x.min(), x.max()+1
            
        
        return y_max-y_min, x_max-x_min
    
    