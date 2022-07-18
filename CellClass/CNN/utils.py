from torchvision import models
from torch import nn
import logging
import os

def get_model(type, pretrained):
    
    model_fn = getattr(models, type)
    model = model_fn(pretrained=pretrained)
    
    if pretrained:
        for param in model.parameters():
                param.requires_grad = False
    
    if isinstance(model, models.ResNet):
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=1)

    else:
        model.classifier[6] = nn.Linear(in_features=4096, out_features=1)
    
    return model

def setup_logger(args, save_dir):
    
    if args.log:
        print("Logging")
        level = logging.INFO
    else:
        print("No Logging")
        level = logging.NOTSET
    
    l = logging.getLogger("training")
    fileHandler = logging.FileHandler(os.path.join(save_dir, "log.txt"), mode='w')
    streamHandler = logging.StreamHandler()
    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)    
    
    l = logging.getLogger("debugger")
    fileHandler = logging.FileHandler(os.path.join(save_dir, "debug.txt"), mode='w')
    streamHandler = logging.StreamHandler()
    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)    

def log(name, message):

    logger = logging.getLogger(name)
    logger.info(message)


if __name__ == "__main__":
    model = get_model("vgg16_bn")
    
    for param in model.parameters():
        print(param.requires_grad)