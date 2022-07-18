
from pandas import lreshape
import torch
import logging
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime
import sys
from CellClass.CNN.utils import log
     

class TrainingScheduler():
    
    def __init__(self, model, optimizer, save_dir, min_learning_rate=10**(-6), patients=10, reduction=5):
        
        self.model = model
        self.optimizer = optimizer
        self.min_learning_rate = min_learning_rate
        self.save_dir = save_dir
        self.val_losses = np.array([])
        self.train_losses = np.array([])
        self.accuracies = np.array([])
        self.patients = patients
        self.stop_early = False
        self._saved_dict = {"models": [], "losses": []}
        self.logger = logging.getLogger("training")
        self.reduction = reduction
        
        self._patients_counter = 0
        
    def add_loss(self, epoch, train_loss, val_loss, accuracy):
        
        self.saved = False
        self.val_loss = val_loss
            
        top5 = np.sort(self.val_losses)[:5]
        if len(self.val_losses) < 5:
            self.save_model(epoch, train_loss, accuracy)
            self.saved = True
            self._patients_counter = -1
            
        elif any(self.val_loss < top5):
            self.save_model(epoch, train_loss, accuracy)
            self.remove_worst()
            self.saved = True
            self._patients_counter = -1
            
        elif self._patients_counter == self.patients:
            self.set_lr(self.get_lr()/self.reduction)
            self.logger.info(f"Reducing Learning Rate by factor of {self.reduction} to {self.get_lr():.2e}")
            self._patients_counter = -1
            if self.get_lr() < self.min_learning_rate:
                self.stop_early = True
                self.logger.info("Early Stopping")
                
        self.val_losses = np.append(self.val_losses, val_loss) 
        self.train_losses = np.append(self.train_losses, train_loss) 
        self.accuracies = np.append(self.train_losses, accuracy) 
        self._patients_counter += 1
        
    def save_model(self, epoch, train_loss, accuracy): 
        
        
        time = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "accuracy": accuracy,
                "train_loss": train_loss,
                "validation_loss": self.val_loss,
                "epoch": epoch,
                "val_losses": self.val_losses,
                "train_losses": self.train_losses,
                "accuracies": self.accuracies
                
            },
            os.path.join(self.save_dir, f"CNN_Model_{time}.pt")
        )

        self._saved_dict["losses"].append(self.val_loss)
        self._saved_dict["models"].append(os.path.join(self.save_dir, f"CNN_Model_{time}.pt"))
    

    def remove_worst(self):
        
        remove_loss = [x for x, val in enumerate(self._saved_dict["losses"]) if val == max(self._saved_dict["losses"])][0]
        os.remove(self._saved_dict["models"][remove_loss])
        self._saved_dict["losses"].pop(remove_loss)
        self._saved_dict["models"].pop(remove_loss)
        
        
    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']      
            
            
    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr           


def get_device():
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger = logging.getLogger("training")
    logger.info(f"Using {device} as Training Device")
    log("debugger", f"Device: {device}")
    return device

def train(model, epochs, train_loader, val_loader, loss_fn, optimizer, save_dir, patients=10):
    
    logger = logging.getLogger("training")
    
    device = get_device()
    model_saver = TrainingScheduler(model, optimizer, save_dir, patients=patients)
    
    model.to(device)
    
    loss_list = []
    for epoch in range(epochs):
        sys.stdout.flush()
        
        if model_saver.stop_early:
            logger.info(f"Stopping Early! No improvement over {model_saver.patients} epochs!")
            break
        
        running_loss = 0
        n = 0
        for sample in train_loader:
            
            X = sample["image"]
            y = sample["true_class"]
            y = y.unsqueeze(-1).float()

            n += X.shape[0]
            X = X.to(device)
            y = y.to(device)
            
            pred = model(X)
            loss = loss_fn(pred, y)
            
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        running_loss /= n
        val_loss, accuracy = validate(model, val_loader, device, loss_fn)
        log("debugger", f"Accuracy: {accuracy.item()}")
        
        model_saver.add_loss(epoch, running_loss, val_loss, accuracy)
        
        loss_list.append([running_loss, accuracy, val_loss])
        
        if model_saver.saved:
            logger.info(f"[{epoch}/{epochs}] Train Loss: {running_loss:.2e} // Validation Loss: {val_loss:.5e} // Accuracy: {np.round(accuracy*100,2)}% || SAVED")
        else:
            logger.info(f"[{epoch}/{epochs}] Train Loss: {running_loss:.2e} // Validation Loss: {val_loss:.5e} // Accuracy: {np.round(accuracy*100,2)}% || NOT SAVED PATIENTS[{model_saver._patients_counter}/{model_saver.patients}]")

        
    loss_list = np.array(loss_list)
    
    return loss_list, model
        
        
def validate(model, val_loader, device, loss_fn):
    
    running_loss = float(0)
    running_accuracy = float(0)
    n = 0
    with torch.no_grad():
        model.eval()
        
        for sample in val_loader:
            
            X = sample["image"]
            y = sample["true_class"]
            y = y.unsqueeze(-1).float()

            n += X.shape[0]
            
            X = X.to(device)
            y = y.to(device)
            
            pred = model(X)
            
            loss = loss_fn(pred, y)

            running_loss += loss.item()
            
            classes = pred.detach()
            classes[pred > 0] = 1
            classes[pred <= 0] = 0
            running_accuracy += sum(classes == y).cpu().detach().numpy()
            
    running_loss /= n
    running_accuracy /= n
        
    return running_loss, running_accuracy
    

def test_model(model, test_loader, loss_fn):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    running_loss = float(0)
    running_accuracy = float(0)
    n = 0
    images = []
    class_all = []
    class_target = []
    with torch.no_grad():
        model.eval()
        
        for sample in test_loader:
            
            X = sample["image"]
            y = sample["true_class"]
            y = y.unsqueeze(-1).float()
            
            n += X.shape[0]     
            class_target.extend(y.numpy())
            images.extend(np.array(X))
            
            X = X.to(device)
            y = y.to(device)
            
            pred = model(X)

            loss = loss_fn(pred, y)


            running_loss += loss.item()
            
            classes = pred.detach()
            classes[pred > 0] = 1
            classes[pred <= 0] = 0
            class_all.extend(np.array(classes.cpu().detach().numpy()))
            running_accuracy += sum(classes == y).cpu().detach().numpy()
            
    running_loss /= n
    running_accuracy /= n


    images = np.array(images)
    images = np.transpose(images, [0, 2,3,1])
    
    logger = logging.getLogger("training")
    logger.info(f"Test Accuracy is {np.round(running_accuracy*100,2)}%")   
    
    return images, class_all, class_target, np.round(running_accuracy*100,2)


def predict_dilution(model, test_loader, verbose=False, return_output=False, with_uncertainty=False):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    n = 0
    images = []
    class_all = []
    all_preds = []
    with torch.no_grad():
        model.to(device)
        model.eval()
        
        for sample in test_loader:
            
            X = sample["image"]

            n += X.shape[0]        
            images.extend(np.array(X))
            
            X = X.to(device)
            pred = model(X)
            all_preds.extend(pred.detach().clone())
            classes = pred.detach()
            classes[pred > 0] = 1
            classes[pred <= 0] = 0
            class_all.extend(np.array(classes.cpu().detach().numpy()))

    images = np.array(images)
    images = np.transpose(images, [0, 2, 3, 1])
    
    class_all = np.array(class_all)
    
    if with_uncertainty:
        
        pos = 0
        neg = 0
        unc = 0
        probability = torch.sigmoid(torch.tensor(all_preds))
        for prob in probability:
            if prob > 0.9:
                pos+=1
            elif prob < 0.1:
                neg+=1
            else:
                unc+=1
    
        if verbose:
            print(f"Dilation Prediction is {np.round((pos/(pos+neg))*100,2)}")   
            
        if return_output:
            return images, [pos, neg, unc], np.round((pos/(pos+neg))*100,2), all_preds
        
        else:
            return images, [pos, neg, unc], np.round((pos/(pos+neg))*100,2)
    
    else:
        
        if verbose:
            print(f"Dilation Prediction is {np.round(sum((class_all == 1))/len(class_all)*100,2)}")   
        
        if return_output:
            return images, list(class_all), np.round(sum((class_all == 1))/len(class_all)*100,2)[0], all_preds
        
        else:
            return images, list(class_all), np.round(sum((class_all == 1))/len(class_all)*100,2)[0]
    