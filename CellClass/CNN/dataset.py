from torch.utils.data import Dataset
import os
from tqdm import tqdm 
from CellClass import Analyse as an
import numpy as np
import torch

class MYCNdataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, images, targets, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = images
        if isinstance(targets, (np.ndarray, list)):
            self.targets = targets
        else:

            self.targets = ["NO Class" for _ in images]


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = self.images[idx]
        label = self.targets[idx]
        
        sample = (image, label)

        return sample
    

def load_training_patches(base, negative_sample, positive_sample, n=None):
    
    negative = [os.path.join(base, x) for x in os.listdir(base) if negative_sample in x]
    positive = [os.path.join(base, x) for x in os.listdir(base) if positive_sample in x]

    if n:
    
        negatives = []
        for f in tqdm(negative[:n]):
            negatives.extend(an.load_patches(f))
            
        positives = []
        for f in tqdm(positive[:n]):
            positives.extend(an.load_patches(f))
            
    else: 
        
        negatives = []
        for f in tqdm(negative):
            negatives.extend(an.load_patches(f))
            
        positives = []
        for f in tqdm(positive):
            positives.extend(an.load_patches(f))
            
        
    return positives, negatives


def create_dataset(base, negative_sample, positive_sample, split_ratios=[0.8, 0.1, 0.1], n=None):
    
    positives, negatives = load_training_patches(base, negative_sample, positive_sample, n=n)

    train_stack = []       
    for p in positives:
        rgb = p.RGB
        rgb[~p.mask] = 0
        train_stack.append(rgb)
    train_p = np.stack(train_stack)
    target_p = np.ones(train_p.shape[0])

    train_stack = []       
    for p in negatives:
        rgb = p.RGB
        rgb[~p.mask] = 0
        train_stack.append(rgb)
    train_n = np.stack(train_stack)
    target_n = np.zeros(train_n.shape[0])

    train_p = np.transpose(train_p, axes=[0, 3, 1, 2])
    train_n = np.transpose(train_n, axes=[0, 3, 1, 2])
    
    train = np.concatenate((train_p, train_n))
    target = np.concatenate((target_p, target_n))
    
    dataset = MYCNdataset(train, target)
    
    return split_dataset(dataset, split_ratios)
    
    
def split_dataset(dataset, split_ratios):
    
    n_train = int(len(dataset)*split_ratios[0])
    n_val = int(len(dataset)*split_ratios[1])
    n_test = len(dataset)-(n_train+n_val)
    
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])
    return train_set, val_set, test_set

def load_dilution(base, sample, n=None, verbose=True):
        
    samples = [os.path.join(base, x) for x in os.listdir(base) if sample in x]

    if n:
        images = []
        if verbose:
            iterator = tqdm(samples[:n])
        else:
            iterator = samples[:n]
        for f in iterator:
            images.extend(an.load_patches(f))
    else:
        images = []
        if verbose:
            iterator = tqdm(samples)
        else:
            iterator = samples
        for f in iterator:
            images.extend(an.load_patches(f))
        
    stack = []       
    for p in images:
        rgb = p.RGB
        rgb[~p.mask] = 0
        stack.append(rgb)
        
    images = np.stack(stack)
    images = np.transpose(images, axes=[0, 3, 1, 2])
    
    dataset = MYCNdataset(images, None)
    
    return dataset