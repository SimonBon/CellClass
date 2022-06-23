import os  
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pickle as pkl
import random 
import numpy as np
from CellClass.CNN import transformations as T


class PatchDataset(Dataset):
    def __init__(self, files, pos="S19_", neg="S29_", transform=None, eval=False):
        super().__init__()
        self.eval = eval
        self.pos = pos
        self.neg = neg
        self.transform = transform
        self.files = files
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        sample_name = self.files[idx].split("/")[-1]
        true_class = self.get_true_class(sample_name, idx)

        with open(self.files[idx], "rb") as fin:
            patch = pkl.load(fin)
            masked = self.get_masked(patch)
            
            sample = {"image": masked,
                      "true_class": true_class}
            
            if self.transform:
                sample = self.transform(sample)
            else:
                self.basic_transform = transforms.Compose([T.ToTensor(), T.Normalize()])
                sample = self.basic_transform(sample)
                
            if self.eval: 
                sample = {"image": sample["image"]}
                return sample
            else:
                return sample
            
            
    def get_masked(self, patch):
        tmp = patch.RGB
        tmp[~patch.mask]=0
        return tmp
    
    def get_true_class(self, sample_name, idx):
        
        if self.pos in sample_name:
            label = 1
        elif self.neg in sample_name:
            label = 0
        else:
            label = -1
            
        return label
    
  
class MYCNTrainingSet(Dataset):
    
    def __init__(self, base, transform, pos="S19_", neg="S29_", split=[0.8, 0.1, 0.1], n=False):
        super().__init__()
        self.base = base
        self.pos = pos
        self.neg = neg
        self.split = split
        self.transform = transform
        self.files = np.array(self._get_files(self.pos) + self._get_files(self.neg))
        if n:
            self.files = self.files[:1000]
    
        assert sum(self.split) == 1, "Split must sum up to 1"
        
        train_files, val_files, test_files = self.get_split_indices()
        
        self.train_dataset = PatchDataset(train_files, transform=self.transform)
        self.val_dataset = PatchDataset(val_files)
        self.test_dataset = PatchDataset(test_files)
        
    def get_split_indices(self):
        
        total = len(self.files)
        train_num = int(len(self.files)*self.split[0])
        val_num = int(len(self.files)*self.split[1])
        test_num = len(self.files)-(train_num+val_num)
        idxs = list(range(len(self.files)))
        random.shuffle(idxs)
        
        train_idxs = idxs[:train_num]
        val_idxs = idxs[train_num:train_num+val_num]
        test_idxs = idxs[-test_num:]
        
        return self.files[train_idxs], self.files[val_idxs], self.files[test_idxs]
          
    def _get_files(self, pattern):
        return [os.path.join(self.base, x) for x in os.listdir(self.base) if pattern in x]
    
    
    