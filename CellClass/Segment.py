from deepcell.applications import NuclearSegmentation
from deepcell.utils.plot_utils import make_outline_overlay, create_rgb_image
import numpy as np

class Segmentation():
    
    def __init__(self, *args, **kwargs):
        
        self.app = NuclearSegmentation(*args, **kwargs)
    
    def __call__(self, im, *args, return_outline=False, **kwargs):
    
        if im.ndim == 2:
            tmp = np.expand_dims(im, axis=0)
            tmp = np.expand_dims(tmp, axis=-1)
            
        elif im.ndim == 3:
            tmp = np.expand_dims(im, axis=-1)
            
        elif im.ndim == 4:
            tmp = np.copy(im)
            
        masks = self.app.predict(tmp, *args, **kwargs)
        
        if return_outline:
            
            outline = self.create_outline(tmp, masks)
            return im, masks.squeeze(), outline
            
        else:
            return im, masks.squeeze()
        
            
    @staticmethod
    def create_outline(im, mask):
        
        rgb = create_rgb_image(im, ["blue"])
        outline = make_outline_overlay(rgb, mask)
        
        return outline.squeeze()
        
        
         
            
        