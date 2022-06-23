import torch
from torchvision import transforms

class ToTensor():
    
    def __call__(self, sample):
        
        image, true_class = sample['image'], sample['true_class']        
        image = image.transpose((2, 0, 1))
        

        return {'image': torch.from_numpy(image),
                'true_class': torch.tensor(true_class)}


class RandomAffine():
    
    def __init__(self):
        self.transformer = transforms.RandomAffine(degrees=(0,360), translate=(0.1, 0.1), scale=(0.9, 1.1), fill=0)

    def __call__(self, sample):
        
        image, true_class = sample['image'], sample['true_class'] 
        
        return {'image': self.transformer(image),
                'true_class': true_class}
  
class RandomFlip():
    
    def __init__(self):
        self.vert = transforms.RandomVerticalFlip()
        self.hor = transforms.RandomHorizontalFlip()
    
    def __call__(self, sample):
        
        image, true_class = sample['image'], sample['true_class'] 
        
        image = self.vert(image)
        image = self.hor(image)
        
        return {'image': image,
                'true_class': true_class}
       
class Normalize():
    
    def __init__(self):
        self.means = torch.tensor([0.019045139, 0.033935968, 0.06551865])
        self.stds = torch.tensor([0.05771165, 0.11356566, 0.17648968])
        self.transformer = transforms.Normalize(self.means, self.stds)
        
    def __call__(self, sample):
        
        image, true_class = sample['image'], sample['true_class'] 
        
        return {'image': self.transformer(image),
                'true_class': true_class}
  