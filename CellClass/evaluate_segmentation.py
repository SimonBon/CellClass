import cv2
import os
import numpy as np
from tqdm import tqdm


def _process_illustrator_output(filename, output_dir, image_size=(1496, 2048)):
    
    name = filename.split("/")[-1].split(".")[0]
    inp = cv2.imread(filename,0)
    inp = cv2.resize(inp, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
    
    inp[inp < 0.99*inp.max()] = 0
    inp[inp >= 0.99*inp.max()] = 1
    _, inp, _, _ = cv2.connectedComponentsWithStats(inp, 4)
    
    inp = inp.astype(np.uint16)
    cv2.imwrite(os.path.join(output_dir, f"{name}_procecced.png"), inp)


def load_segmentation(filepath):
    return cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)


def random_color(im):
 
    ret = np.zeros((*im.shape, 3))
    for c in range(1,im.max()):
        col = np.random.uniform(0, 1, 3)
        ret[im == c, :] = col
        
    return ret
    
    
def remove_boarder(mask, border_margin=5):
    
    for c in range(1, mask.max()):
        idx = np.where(mask == c)
        if idx[0].min() < border_margin or idx[1].min() < border_margin:
            mask[mask == c] = 0
        if idx[0].max() > mask.shape[0]-border_margin or idx[1].max() > mask.shape[1]-border_margin:
            mask[mask == c] = 0
        
    return mask

def additive_blend(im0, im1):
    
    im0 = np.array(im0)
    im0 = np.clip(im0, 0, 1)

    im1 = np.array(im1)
    im1 = np.clip(im1, 0, 1)

    rgb = np.zeros((*im0.shape, 3))
    rgb[:, :, 0] = im0
    rgb[:, :, 1] = im1

    return rgb

def dice(target, pred):
    
    A = target.astype(bool)
    B = pred.astype(bool)
    
    A_n_B = np.sum(A * B)
    
    return 2*A_n_B/(np.sum(A)+np.sum(B))
    
def jaccard(target, pred):
    
    A = target.astype(bool)
    B = pred.astype(bool)
    
    A_n_B = np.sum(A * B)
    A_u_B = np.sum(A | B)

    return A_n_B/A_u_B

def precision_recall(target, pred):
    
    A = target.astype(bool)
    B = pred.astype(bool)
    
    TP = np.sum(A * B)
    FP = np.sum(np.invert(B) * A)
    FN = np.sum(np.invert(A) * B)
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    return precision, recall

def find_corresponding(target, pred):
    
    cell_idxs = np.unique(pred)
    cell_idxs = cell_idxs[cell_idxs != 0]
    
    if len(cell_idxs) == 0:
        return 0
    if len(cell_idxs) == 1:
        return cell_idxs[0]
    else:
        ret = [np.sum(pred == c) for c in cell_idxs]
        return cell_idxs[np.where(ret == max(ret))][0]
    

def compare_segmentations(seg0, seg1):

    ret = []
    for c in tqdm(range(1, seg0.max()+1)):
        
        tmp_seg0 = np.copy(seg0)
        tmp_seg0[tmp_seg0 != c] = 0
        
        if not tmp_seg0.any():
            continue

        tmp_seg1 = np.copy(seg1)
        tmp_seg1[tmp_seg0 == 0] = 0
        
        cell_idx = find_corresponding(tmp_seg0, tmp_seg1)
        
        if cell_idx != 0:
            
            tmp_seg1 = np.copy(seg1)
            tmp_seg1[tmp_seg1 != cell_idx] = 0

            vprecision, vrecall = precision_recall(tmp_seg0, tmp_seg1)
            vdice = dice(tmp_seg0, tmp_seg1)
            vjaccard = jaccard(tmp_seg0, tmp_seg1)
            
        else:
            
            vprecision, vrecall = 0, 0
            vdice = 0
            vjaccard = 0
                
        for val, modularity in zip([vdice, vjaccard, vprecision, vrecall], ["dice", "jaccard", "precision", "recall"]): 
            ret.append({
                    "val": val,
                    "mod":  modularity,
                })
            
    return ret