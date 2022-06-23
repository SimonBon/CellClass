from inspect import getargvalues
import torch
from torch import nn
import torch.nn.functional as F

__version__ = "0.0.1"
    
class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
    def forward(self, X):
        
        return self.block(X)
       
       
class ClassificationCNN(nn.Module):
    def __init__(self, layers=[3, 16, 64, 128, 256], in_shape=[128, 128]):
        super().__init__()
        
        self.layers = layers
        self.in_shape = in_shape
        
        self.features = torch.nn.Sequential()
        for i in range(len(layers)-1):
            self.features.add_module(f"conv{i}", ConvBlock(layers[i], layers[i+1]))
        
        self.last_nodes = int(layers[-1]*(in_shape[0]/2**(len(layers)-1))**2)
        
            
        self.fc =  nn.Sequential(
            nn.Linear(self.last_nodes, 100),
            nn.Linear(100, 2)
        )
        
        
    def forward(self, X):
        
        for mod in self.features:
            X = mod(X)
            
        X = X.view(-1, self.last_nodes)
        return self.fc(X)



class HookWrapper(nn.Module):
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.activation = {}
        self.handles = []
         
    def hook_layers(self):
        
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Conv2d):
                self.handles.append(mod.register_forward_hook(self.getActivation(name)))
            
            if isinstance(mod, nn.Linear):
                print("linear")
             
             
    def getActivation(self, name):
        
        def hook(model, input, output):    
            self.activation[name] = output.detach()
            
        return hook
            
    def unhook_layers(self):
        
        if len(self.handles) == 0:
            print("No attached hooks")
        else:
            for h in self.handles:
                h.remove()
            print("All Hooks unhooked")