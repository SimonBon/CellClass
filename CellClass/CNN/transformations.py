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
    
    def __init__(self, rescaled): 
        if rescaled:
            self.means = torch.tensor([0.0335621, 0.048920628, 0.09400295])
            self.stds = torch.tensor([0.098687746, 0.14546247, 0.24435566])
        else:
            self.means = torch.tensor([0.019170875, 0.03451792, 0.0660516])
            self.stds = torch.tensor([0.058210537, 0.11543953, 0.17737558])
            
        self.transformer = transforms.Normalize(self.means, self.stds)
        
    def __call__(self, sample):
        
        image, true_class = sample['image'], sample['true_class'] 
        
        return {'image': self.transformer(image),
                'true_class': true_class}
        
        
class DeNormalize():
    
    def __init__(self, rescaled):
        if rescaled:
            self.means = torch.tensor([-0.0335621, -0.048920628, -0.09400295])
            self.stds = torch.tensor([1/0.098687746, 1/0.14546247, 1/0.24435566])
        else:
            self.means = torch.tensor([-0.019170875, -0.03451792, -0.0660516])
            self.stds = torch.tensor([1/0.058210537, 1/0.11543953, 1/0.17737558])
            
        self.transformer = transforms.Compose([transforms.Normalize([0,0,0], self.stds), transforms.Normalize(self.means, [1,1,1])])
        
    def __call__(self, image):

        return self.transformer(image)
