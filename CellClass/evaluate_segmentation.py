import cv2
import os
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt


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
    
    for c in np.unique(mask):
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
    
def get_false_negatives(seg0, seg1):

    FP_counter = 0
    colors_to_remove = []
    for c in tqdm(np.unique(seg0)):
        if c==0:
            continue
        
        true = np.zeros_like(seg0)
        true[seg0==c]=1
        sz_true = np.sum(true)
        
        tmp1 = np.copy(seg1)
        tmp1[true==0] = 0
        tmp1[tmp1 != 0] = 1
        sz_pred = np.sum(tmp1)
        
        if (sz_pred/sz_true) < 0.2 or sz_true < 25:
            if sz_true > 25:
                FP_counter += 1
            colors_to_remove.append(c)

    return FP_counter, colors_to_remove

def get_false_positives(seg0, seg1):
    
    FN_counter = 0
    colors_to_remove = []
    for c in tqdm(np.unique(seg1)):
        if c==0:
            continue
        
        true = np.zeros_like(seg1)
        true[seg1==c]=1
        sz_true = np.sum(true)
        
        tmp0 = np.copy(seg0)
        tmp0[true==0] = 0
        tmp0[tmp0 != 0] = 1
        sz_pred = np.sum(tmp0)
        
        if (sz_pred/sz_true) < 0.2 or sz_true < 25:
            if sz_true > 25:
                FN_counter += 1
            colors_to_remove.append(c)

    return FN_counter, colors_to_remove


def remove_colors(seg, colors):
    
    seg = np.copy(seg)
    for c in colors:
        seg[seg==c] = 0
    return seg
    

def compare_segmentations(seg0, seg1):

    fp_counter, seg1_colors_to_remove = get_false_positives(seg0, seg1)
    fn_counter, seg0_colors_to_remove = get_false_negatives(seg0, seg1)

    print(f"False Positives: {fp_counter}\nFalse Negatives: {fn_counter}")
    seg0 = remove_colors(seg0, seg0_colors_to_remove)
    seg1 = remove_colors(seg1, seg1_colors_to_remove)

    ret = []
    for c in tqdm(np.unique(seg0)):
        
        if c == 0:
            continue
        
        tmp_seg0 = np.copy(seg0)
        tmp_seg0[tmp_seg0 != c] = 0
        
        if not tmp_seg0.any():
            continue

        tmp_seg1 = np.copy(seg1)
        tmp_seg1[tmp_seg0 == 0] = 0
        
        cell_idx = find_corresponding(tmp_seg0, tmp_seg1)
        tmp_seg1 = np.copy(seg1)
        tmp_seg1[tmp_seg1 != cell_idx] = 0

        vprecision, vrecall = precision_recall(tmp_seg0, tmp_seg1)
        vdice = dice(tmp_seg0, tmp_seg1)
        vjaccard = jaccard(tmp_seg0, tmp_seg1)
                
        for val, modularity in zip([vdice, vjaccard, vprecision, vrecall], ["dice", "jaccard", "precision", "recall"]): 
            ret.append({
                    "val": val,
                    "mod":  modularity,
                })
            
    return ret, fp_counter, fn_counter