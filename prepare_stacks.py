import os 
import queue
import threading
import argparse
from pathlib import Path
import re
import cv2
from natsort import natsorted
import numpy as np
import tifffile
from tqdm import tqdm
from collections import defaultdict
import sys

import matplotlib.pyplot as plt

NUM_WORKER = 16

def parse():

    p = argparse.ArgumentParser()
    p.add_argument("-i", "--in_path")
    p.add_argument("-o", "--out_path")
    args = p.parse_args()
    return args


def get_files(p: os.PathLike) -> list:

    files = [x for x in os.listdir(p) if ".tif" in x.lower() and not x.startswith(".") and not "r.tif" in x.lower() and not "g.tif" in x.lower()]
    
    samples = defaultdict(list)

    for f in files:
        basename, extension = os.path.splitext(f)
        if not "B" in f:
            _, idx, channel, focus = basename.split("-")
        else:
            _, idx, channel = basename.split("-")
        
        samples[idx].append(f)
        
    return samples

def stack_save_focuses(idx, files, in_p, out_p):

    r = []
    g = []
    for f in files:
        
        ff = os.path.join(in_p, f)
        
        if "R" in f:
            r.append(cv2.imread(ff,0))
        if "G" in f:
            g.append(cv2.imread(ff,0))
        if "B" in f: 
            b = cv2.imread(ff,0)
            
    ims = np.array([r, g])

    mean = np.sum(ims, axis=1)/ims.shape[1]
    

    cv2.imwrite(os.path.join(out_p, f'Img-{idx}-G.TIF'), mean[0].astype("uint8"))
    cv2.imwrite(os.path.join(out_p, f'Img-{idx}-R.TIF'), mean[1].astype("uint8"))
    cv2.imwrite(os.path.join(out_p, f'Img-{idx}-B.TIF'), b.astype("uint8"))

class Worker(threading.Thread):
    def __init__(self, in_p, out_p, q, task, list, n, *args, **kwargs):
        self.in_p = in_p
        self.out_p = out_p
        self.q = q
        self.task = task
        self.list = list
        self.n = n
        super().__init__(*args, **kwargs)

    def run(self):
        while True:
            try:
                idx, files = self.q.get_nowait()
                self.list.pop()
                self.task(idx, files, self.in_p, self.out_p)
                self.__progressBar()

            except queue.Empty:
                return

            self.q.task_done()

    def __progressBar(self, barLength=50):

        percent = int((self.n - len(self.list)) * 100 / self.n)
        arrow = '█' * int(percent/100 * barLength - 1)
        spaces = ' ' * (barLength - len(arrow))

        sys.stdout.write("\r" + f'Progress: |{arrow}{spaces}| {percent}% [{self.n-len(self.list)}/{self.n}]' + "\r")
        sys.stdout.flush()

def merge_focuses(in_p, out_p, s):
        
    Q = queue.Queue()
    for idx, files in s.items():
        Q.put_nowait((idx, files))
        
    l = list(range(len(s)))
    n = len(l)

    for _ in range(NUM_WORKER):
        Worker(
            in_p,
            out_p,
            Q,
            stack_save_focuses, 
            l, 
            n).start()

    Q.join()   
    
    
if __name__ == "__main__":

    args = parse()
    
    if not os.path.isdir(args.out_path):
        os.mkdir(args.out_path)

    samples = get_files(args.in_path)
    merge_focuses(args.in_path, args.out_path, samples)