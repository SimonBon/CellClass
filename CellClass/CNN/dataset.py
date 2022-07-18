import os  
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pickle as pkl
import random 
import numpy as np
from CellClass.CNN import transformations as T
from CellClass.CNN.utils import log

class PatchDataset(Dataset):
    """Args:
        files (list): list of filepaths to patches used in the dataset
        pos (str, optional): Samplename for positive Samples. Defaults to "S19_".
        neg (str, optional): Samplename for negative Samples. Defaults to "S29_".
        transform (torchvision.transforms, optional): Transformations that you want to be applied to the dataset. Defaults to None.
        eval (bool, optional): Define if the Dataset is used for evaluation, meaning there are no target samples. Defaults to False.
        rescale_intensity (bool, optional): Defines if the image channels are rescaled between [0, 1]. Defaults to True.
    """
    def __init__(self, files, pos="S19_", neg="S29_", transform=None, eval=False, rescale_intensity=True):
        
        super().__init__()
        self.eval = eval
        self.pos = pos
        self.neg = neg
        self.transform = transform
        self.files = files
        self.rescale_intensity = rescale_intensity
        
        if self.rescale_intensity:
            log("debugger", "using rescaled intensities")
        else:
            log("debugger", "not using rescaled intensities")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        """Return a sample from the given list of patches

        Args:
            idx (int): index into the files list

        Returns:
            sample (dict): 
                ["image"]: stored image data
                ["true_class"]:  stored true class of the respective image
        """
        sample_name = self.files[idx].split("/")[-1]
        true_class = self.get_true_class(sample_name)

        with open(self.files[idx], "rb") as fin:
            patch = pkl.load(fin)
            masked = self.get_masked(patch)
            
            if self.rescale_intensity:
                image = self.rescale(masked)
            else:
                image = np.copy(masked)
            
            sample = {"image": image,
                      "true_class": true_class}
            
            if self.transform:
                sample = self.transform(sample)
            else:
                #if no transformations are given atleast normalize the image and return it as a tensdor
                self.basic_transform = transforms.Compose([T.ToTensor(), T.Normalize(self.rescale_intensity)])
                sample = self.basic_transform(sample)
                
            if self.eval: 
                #if in evaluation just return the image and not the true class because there is none
                sample = {"image": sample["image"]}
                return sample
            
            else:
                return sample
            
    def rescale(self, im):
        """
        Return rescaled image between [0, 1] for each of the 3 channels
        """ 
        ret = np.copy(im)       
        for i in range(ret.shape[-1]):
            ret[..., i] = (ret[..., i]-ret[..., i].min())/(ret[..., i].max()-ret[..., i].min())
            
        return ret
                
    def get_masked(self, patch):
        """
        get the RGB image from patch masked by the segmentation provided in the Patch-Class
        """
        tmp = patch.RGB
        tmp[~patch.mask]=0
        return tmp
    
    def get_true_class(self, sample_name):
        """Get the true class based on the sample name.

        Args:
            sample_name (str): name of the specific sample e.g. "S19_34_120"

        Returns:
            bool: respective class
        """
        if self.pos in sample_name:
            label = 1
        elif self.neg in sample_name:
            label = 0
        else:
            label = -1
            
        return label
    
  
class MYCNTrainingSet(Dataset):
    """_summary_

    Args:
        base (str): directory in which all patches are stored
        transform (torchvision.transforms): transformations that are applied to the training samples
        pos (str, optional): Samplename for positive Samples. Defaults to "S19_".
        neg (str, optional): Samplename for negative Samples. Defaults to "S29_".
        split (list, optional): ratios in training validation and test split. Defaults to [0.8, 0.1, 0.1].
        n (bool, int, optional): Define if you want to use only a number of training samples (for debugging purposes). Defaults to False.
        equal (bool, optional): define if you want to use equal amounts of true and false datasamples. Defaults to True.
        rescale_intensity (bool, optional): . Define if you want to rescale image channels in [0, 1]. Defaults to True.
    """
    def __init__(self, base, transform, pos="S19_", neg="S29_", split=[0.8, 0.1, 0.1], n=False, equal=True, rescale_intensity=True):

        super().__init__()
        self.base = base
        self.pos = pos
        self.neg = neg
        self.split = split
        self.transform = transform
        
        self.neg_files = self._get_files(self.neg)
        self.pos_files = self._get_files(self.pos)
        if equal:
            equalize_num = min(len(self.neg_files), len(self.pos_files))
            self.neg_files = self.neg_files[:equalize_num]
            self.pos_files = self.pos_files[:equalize_num]
            
        self.files = np.array(self.pos_files + self.neg_files)
        
        if n:
            random.shuffle(self.files)
            self.files = self.files[:n]
            
    
        assert sum(self.split) == 1, "Split must sum up to 1"
        
        train_files, val_files, test_files = self.get_split_indices()
        
        self.train_dataset = PatchDataset(train_files, transform=self.transform, rescale_intensity=rescale_intensity)
        self.val_dataset = PatchDataset(val_files, rescale_intensity=rescale_intensity)
        self.test_dataset = PatchDataset(test_files, rescale_intensity=rescale_intensity)
        
        
    def get_split_indices(self):
        #get the list of files for training, validation and testing based on 'split' value
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
        #get files from directory that match pattern of given sample
        return [os.path.join(self.base, x) for x in os.listdir(self.base) if pattern in x]
    
    
    