#!/usr/bin/python3
import os
import logging
import argparse
import pickle as pkl
from torch import nn
from CellClass import CNN
from torch.optim import Adam
from datetime import datetime
import matplotlib.pyplot as plt
from CellClass.CNN.dataset import MYCNTrainingSet
from CellClass.CNN import transformations as trans
from torchvision import transforms
from CellClass.CNN import training as T
from CellClass.CNN import utils
from torch.utils.data import DataLoader

def save_parameters_to_txt(save_dir, args, dataset, model):

    with open(os.path.join(save_dir, "training_parameters.txt"), "w+") as fout:
        for arg in vars(args):
            fout.write(f"{arg}: {getattr(args, arg)}\n")
        
        fout.write(f"Number Positive Samples: {len(dataset.pos_files)}\nNumber Negative Samples: {len(dataset.neg_files)}\n")
        fout.write(f"Train-Patches: {len(dataset.train_dataset)}\nValidation_Patches: {len(dataset.val_dataset)}\nTest_Patches: {len(dataset.test_dataset)}\n")
        if isinstance(model, CNN.ClassificationCNN):
            fout.write(f"layers: {model.layers}\n")
            fout.write(f"in_shape: {model.in_shape}\n")
        
def create_save_dir(args):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, now)
    os.mkdir(save_dir)
    return save_dir
    
    
def save_training_losses_to_png(save_dir: str, l):

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.plot(range(len(l[:,0])), l[:,0])
    ax.plot(range(len(l[:,0])), l[:,1])
    ax.plot(range(len(l[:,0])), l[:,2])

    ax.legend(["Train", "Acc", "Val"])
    ax.set_yscale('log')
    plt.savefig(os.path.join(save_dir, "losses.png"))
    
def save_losses_to_pkl(save_dir, l):
    
    with open(os.path.join(save_dir, "losses.pkl"), "wb") as fout:
        pkl.dump(l, fout)

def test_model(save_dir, model, test_loader, loss_fn):
    
    _, _, _, acc = T.test_model(model, test_loader, loss_fn)
    
    with open(os.path.join(save_dir, "test_accuracy.txt"), "w+") as fout:
        fout.write(f"Test Accuracy was {acc}%")

# get arguments for preparation of images to save them in BGR and tif format. 
def parse():

    p = argparse.ArgumentParser()
    p.add_argument("-lr", "--learning_rate", type=float, help="initial_learning_rate")
    p.add_argument("--model", type=str, help="select a model from 'VGG16_BN', 'ResNet50' and 'ResNet18', 'ClassificationCNN'", default="ClassificationCNN")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    p.set_defaults(pretrained=True)
    p.add_argument("--layers", nargs='+', type=int, default=[3, 16, 64, 128, 256])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--save_dir", type=str, help="directory to save the models to", default="/out")
    p.add_argument("--patches_dir", type=str, help="directory to all patches", default="/data")
    p.add_argument("--positives", type=str, help="Name of the positive Sample", default="S19")
    p.add_argument("--negatives", type=str, help="Name of the negative Sample", default="S29")
    p.add_argument('--log', action='store_true')
    p.add_argument('--no-log', dest='log', action='store_false')
    p.set_defaults(log=True)
    p.add_argument('--n', type=int)
    p.add_argument('--rescale', action='store_true')
    p.add_argument('--no-rescale', dest='rescale', action='store_false')
    p.set_defaults(rescale=True)
    p.add_argument('--weight_decay', type=float, help="Set the weight decay for optimizer", default=0.0001)
    return p.parse_args()

if __name__ == "__main__":
    
    args = parse()
    
    save_dir = create_save_dir(args)
    utils.setup_logger(args, save_dir)
    
    if args.model.lower() == "classificationcnn":
        model = CNN.ClassificationCNN(layers=args.layers)
    elif args.model.lower() in ['vgg16_bn', 'resnet50','resnet18']:
        delattr(args, "layers")
        model = utils.get_model(args.model, args.pretrained)
    else:
        print("f{args.model} is not available!")   
        
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(list(model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    transform=transforms.Compose([trans.ToTensor(), trans.RandomAffine(), trans.RandomFlip(), trans.Normalize(args.rescale)])
    ds = MYCNTrainingSet(args.patches_dir, neg=args.negatives, pos=args.positives, transform=transform, n=args.n, rescale_intensity=args.rescale)
    train_loader = DataLoader(ds.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(ds.val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(ds.test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    utils.log("debugger", f"Train: {len(ds.train_dataset)}, Val: {len(ds.val_dataset)}, Test: {len(ds.test_dataset)}")
    
    save_parameters_to_txt(save_dir, args, ds, model)

    l, model = T.train(model, args.epochs, train_loader, val_loader, loss_fn, optimizer, save_dir)
    
    save_training_losses_to_png(save_dir, l)
    save_losses_to_pkl(save_dir, l)
    test_model(save_dir, model, test_loader, loss_fn)