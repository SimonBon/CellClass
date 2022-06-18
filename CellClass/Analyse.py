import imaplib
import cv2
import argparse
import pickle as pkl
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def parseargs():

    parser = argparse.ArgumentParser(description='Analyse Patches')
    parser.add_argument('-i','--input', type=str, help='Input path to a .ptch-File')
    parser.add_argument('-v', '--visualize', action='store_true', default=False, help='Flag if Results should be visualized (default: False)')

    return parser.parse_args()


def load_patches(f):
    
    with open(f, "rb") as fin:
        patches = pkl.load(fin)

    return patches


def get_spots(im, mask=None):
    
    binary = get_binary(im, mask)
    
    connectivity = 8
    output = cv2.connectedComponentsWithStats(binary, connectivity)
    
    num = output[0]-1
    coords = output[-1][1:,:]
    
    return num, coords


def get_binary(im, mask, t=0.2):
    
    #m = cv2.erode(p.mask.astype("uint8"), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e))).astype(bool)
    dog_im = DoG(im)
    #dog_im[~m] = 0
    dog_im[dog_im > t] = 1
    dog_im[dog_im < t] = 0
    
    if isinstance(mask, np.ndarray):
        dog_im[~mask] = 0
    
    return dog_im.astype("uint8")


def DoG(im, s0=1, s1=1.6):
    sig0 = cv2.GaussianBlur(im, (31,31), s0, s0)
    sig1 = cv2.GaussianBlur(im, (31,31), s1, s1)
    ret = sig0-sig1
    ret[ret<0] = 0
    return (ret-ret.min())/(ret.max()-ret.min())


def visualize(nums, coords, ims, masks=None):
    
    if not isinstance(masks, list):
        masks = [None for n in nums]
    
    fig, axs = plt.subplots(10, 10, figsize=(30, 30))
    for ax, num, coord, im, mask in zip(axs.ravel(), nums, coords, ims, masks):
        ax.imshow(im, cmap="jet")
        if isinstance(mask, np.ndarray):
            ax.imshow(mask, cmap="gray", alpha=0.2)
        ax.scatter(coord[:,0], coord[:,1], s=80, facecolors='none', edgecolors='black')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title([num], color="white")
        
    plt.show()
    
def get_stats(nums):
    
    nums = np.array(nums)
    
    tot = len(nums)
    hist = [(sum(nums==x), sum(nums==x)/tot) for x in range(16)]
    hist.append((sum(nums > 15), sum(nums > 15)/tot))
    hist = np.array(hist)
    mval = max(hist[:,1])
    
    plt.bar(range(17), hist[:, 1])
    plt.ylim([0, 1.1*mval])
    [plt.text(x, hist[x][1]+0.04*mval, int(hist[x][0]), ha="center", va="center") for x in range(17)]
    plt.xticks(range(17))
    plt.show()
    
    
def visualize_nspots(nums, coords, ims, masks, n, channel="R"):
     
    matches = [[num, coord, im, mask] for (num, coord, im, mask) in zip(nums, coords, ims, masks) if num==n]

    fig, axs = plt.subplots(10, 10, figsize=(30, 30))
    for ax, p in zip(axs.ravel(), matches):
        ax.imshow(p[2], cmap="jet")
        ax.imshow(p[3], cmap="gray", alpha=0.2)
        ax.scatter(p[1][:,0], p[1][:,1], s=80, facecolors='none', edgecolors='black')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title([p[0]], color="white")
        
    plt.show()
    
if __name__ == "__main__":
    
    args = parseargs()
    
    patches = load_patches(args.input)
    
    spots = [(*get_spots(p.R, p.mask), p.R, p.mask) for p in tqdm(patches)]
    
    if args.visualize:
        visualize(spots)
    
    get_stats(spots)
    
    
    

    
    
    