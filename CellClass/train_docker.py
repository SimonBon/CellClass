import os
import logging
import argparse
import pickle as pkl
from torch import nn
from CellClass import CNN
from torch.optim import Adam
from datetime import datetime
import matplotlib.pyplot as plt
from CellClass.CNN import dataset
from CellClass.CNN import training as T
from torch.utils.data import DataLoader


# get arguments for preparation of images to save them in BGR and tif format. 
def parse():

    p = argparse.ArgumentParser()
    p.add_argument("-lr", "--learning_rate", type=float, help="initial_learning_rate")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--save_dir", type=str, help="directory to save the models to", default="/out")
    p.add_argument("--patches_dir", type=str, help="directory to all patches", default="/data")
    p.add_argument("--positives", type=str, help="Name of the positive Sample", default="S19")
    p.add_argument("--negatives", type=str, help="Name of the negative Sample", default="S29")
    p.add_argument("--log", type=bool, help="Define if logging should be done or not", default=True)
    p.add_argument("-n", type=int, help="number of images", default=100)
    return p.parse_args()

if __name__ == "__main__":
    
    args = parse()
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, now)

    os.mkdir(save_dir)
    
    if args.log:
        print("Logging")
        level = logging.INFO
    else: 
        print("No Logging")
        level = logging.NOTSET
    
    
    logging.basicConfig(filename=os.path.join(save_dir, "log.txt"), level=level)
    logger = logging.getLogger("training")
    logger.info(save_dir)
    logger.info(args)

    epochs = args.epochs
    model = CNN.ClassificationCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(list(model.parameters()), lr=args.learning_rate, weight_decay=0.0001)
    
    train, val, test = dataset.create_dataset(args.patches_dir, args.negatives, args.positives, n=args.n)
    train_loader, val_loader, test_loader = DataLoader(train, batch_size=args.batch_size), DataLoader(val, batch_size=args.batch_size), DataLoader(test, batch_size=args.batch_size)
    
    with open(os.path.join(save_dir, "training_parameters.txt"), "w+") as fout:
        for arg in vars(args):
            fout.write(f"{arg}: {getattr(args, arg)}\n")
        
        fout.write(f"Train-Patches: {len(train)}\nValidation_Patches: {len(val)}\nTest_Patches: {len(test)}")
    
    l, model = T.train(model, epochs, train_loader, val_loader, loss_fn, optimizer, save_dir)
    
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.plot(range(len(l[:,0])), l[:,0])
    ax.plot(range(len(l[:,0])), l[:,1])
    ax.plot(range(len(l[:,0])), l[:,2])

    ax.legend(["Train", "Acc", "Val"])
    ax.set_yscale('log')
    plt.savefig(os.path.join(save_dir, "losses.png"))
    
    with open(os.path.join(save_dir, "losses.pkl"), "wb") as fout:
        pkl.dump(l, fout)
        
    _, _, _, acc = T.test_model(model, test_loader, loss_fn)
    
    with open(os.path.join(save_dir, "test_accuracy.txt"), "w+") as fout:
        fout.write(f"Test Accuracy was {acc}%")
        
    