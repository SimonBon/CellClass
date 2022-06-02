import os 
import argparse
from pathlib import Path
import re
import cv2
from natsort import natsorted
import numpy as np
import tifffile
from tqdm import tqdm


import matplotlib.pyplot as plt

def parse():

    p = argparse.ArgumentParser()
    p.add_argument("-i", "--in_path")
    p.add_argument("-o", "--out_path")
    args = p.parse_args()
    return args


def get_files(p: os.PathLike) -> list:

    files = [os.path.join(p,f) for f in os.listdir(p) if ".tif" in f.lower() and not f.startswith(".")]
    img_idxs = natsorted(list(set([re.findall(r"-\d+-", f)[0] for f in files])))
    BGR_images_paths = [list(filter(lambda x: n in x, files)) for n in img_idxs]
    return BGR_images_paths


def as_BGR(files: list) -> np.ndarray:
    
    B = cv2.imread([f for f in files if "b.tif" in f.lower()][0],0)
    G = cv2.imread([f for f in files if "g.tif" in f.lower()][0],0)
    R = cv2.imread([f for f in files if "r.tif" in f.lower()][0],0)

    BGR_img = np.stack([R, G, B], axis=-1)

    return BGR_img


def save_BGR(paths: list, out_path: os.PathLike, sample: str):
    
    for i, files in enumerate(tqdm(paths)):
        img = as_BGR(files)
        tifffile.imwrite(os.path.join(out_path, f"{sample}_{i}.tif"), img)
        
        
if __name__ == "__main__":

    args = parse()

    sample = args.in_path.split(".")[-1]
    files = get_files(args.in_path)
    save_BGR(files, args.out_path, sample)
    
