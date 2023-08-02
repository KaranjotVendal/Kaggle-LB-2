import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
#from torchmetrics import F1Score

import wandb

class Trainer:
    def __init__(
        self, 
        model, 
        device, 
        optimizer, 
        criterion, 
        loss_meter, 
        score_meter,
        F1score,
        fold
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.loss_meter = loss_meter
        self.score_meter = score_meter
        self.F1score = F1score
        self.hist = {#'val_loss':[],
                     #'val_score':[],
                     #'val_f1':[],
                     'train_loss':[],
                     'train_score':[],
                     'train_f1': [],
                    }
        
        self.best_valid_score = -np.inf
        self.best_valid_loss = np.inf
        self.best_f_score = 0
        self.n_patience = 0
        self.fold = fold
        
        self.messages = {
            "epoch": "[Epoch {}: {}] loss: {:.5f}, score: {:.5f}, f1 score: {:.5f} time: {} s",
            "checkpoint": "The score improved from {:.5f} to {:.5f}. f1 score: {:.5f}. Save model to '{}'",
            "patience": "\nTrain score didn't improve last {} epochs."
        }
    
    def fit(self, epochs, train_loader, save_path, patience):        
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)
            
            train_loss, train_score, train_f_score, train_time = self.train_epoch(train_loader)
            wandb.log({
                'train_loss': train_loss,
                'train_score': train_score,
                'train_f1_score': train_f_score,
            })
            #valid_loss, valid_score, valid_time, valid_f_score = self.valid_epoch(valid_loader)
            #self.hist['val_loss'].append(valid_loss)
            self.hist['train_loss'].append(train_loss)
            #self.hist['val_score'].append(valid_score)
            self.hist['train_score'].append(train_score)
            #self.hist['val_f1'].append(valid_f_score)
            self.hist['train_f1'].append(train_f_score)
            
            self.info_message(
                self.messages["epoch"], "Train", n_epoch, train_loss, train_score, train_f_score, train_time
            )
            
            '''self.info_message(
                self.messages["epoch"], "Valid", n_epoch, valid_loss, valid_score, valid_f_score, valid_time
            )'''

            if self.best_valid_score < train_score: #valid_score:
                self.info_message(
                    self.messages["checkpoint"], self.best_valid_score, train_score, train_f_score, save_path #valid_score, valid_f_score, save_path
                )
                self.best_valid_score = train_score #valid_score
                self.best_valid_loss = train_loss #valid_loss
                self.best_f_score = train_f_score #valid_f_score
                self.save_model(n_epoch, save_path)
                self.n_patience = 0
            else:
                self.n_patience += 1
            
            if self.n_patience >= patience:
                self.info_message(self.messages["patience"], patience)
                break

        #valid_loss, valid_score, valid_time, valid_f_score = self.valid_epoch(valid_loader)
                
        return self.best_valid_loss, self.best_valid_score, self.best_f_score
            
    def train_epoch(self, train_loader):
        self.model.train()
        t = time.time()
        train_loss = self.loss_meter()
        train_score = self.score_meter()
        ff_score = self.F1score()
        #ff_score = F1Score(task='binary', num_classes=2).to(self.device)
        
        for step, batch in enumerate(train_loader, 1):
            X = batch["X"].to(self.device)
            targets = batch["y"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X).squeeze(1)
            
            loss = self.criterion(outputs, targets)
            loss.backward()

            train_loss.update(loss.detach().item())
            train_score.update(targets, outputs.detach())
            #f_score = f1_score(targets.cpu().numpy().astype(int), outputs.detach().cpu().numpy())
            ff = ff_score.update(targets, outputs.detach())
            #f_score=f_score.item()
            #f1 = F1Score(task='binary')

            self.optimizer.step()
            
            _loss, _score = train_loss.avg, train_score.avg
            message = 'Train Step {}/{}, train_loss: {:.5f}, train_score: {:.5f}, train_f1: {:.5f}'
            self.info_message(message, step, len(train_loader), _loss, _score, ff, end="\r")
        
        f_score = ff_score.get_score()
        return train_loss.avg, train_score.avg, f_score, int(time.time() - t)
    '''
    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        valid_loss = self.loss_meter()
        valid_score = self.score_meter()
        #f_score = f1_score()
        ff_score = F1Score(task='binary', num_classes=2).to(self.device)

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch["X"].to(self.device)
                targets = batch["y"].to(self.device)

                outputs = self.model(X).squeeze(1)
                loss = self.criterion(outputs, targets)

                valid_loss.update(loss.detach().item())
                valid_score.update(targets, outputs)
                #f_score = f1_score(targets.cpu().numpy().astype(int), outputs.detach().cpu().numpy())
                f_score = ff_score(outputs, targets)
                f_score=f_score.item()

                
            _loss, _score = valid_loss.avg, valid_score.avg
            message = 'Valid Step {}/{}, valid_loss: {:.5f}, valid_score: {:.5f}, valid_f1: {:.5f}'
            self.info_message(message, step, len(valid_loader), _loss, _score, f_score, end="\r")
        
        return valid_loss.avg, valid_score.avg, f_score, int(time.time() - t)'''
    
    def plot_loss(self):
        plt.title("Loss")
        plt.xlabel("Training Epochs")
        plt.ylabel("Loss")

        plt.plot(self.hist['train_loss'], label="Train")
        #plt.plot(self.hist['val_loss'], label="Validation")
        plt.legend()
        plt.savefig(f"./plots/loss/loss_{self.fold}.png")
        #plt.show()
    
    def plot_score(self):
        plt.title("Score")
        plt.xlabel("Training Epochs")
        plt.ylabel("Acc")

        plt.plot(self.hist['train_score'], label="Train")
        #plt.plot(self.hist['val_score'], label="Validation")
        plt.legend()
        plt.savefig(f"./plots/score/score_{self.fold}.png")
        #plt.show()

    def plot_fscore(self):
        plt.title("f1_score")
        plt.xlabel("Training Epochs")
        plt.ylabel("f1_score")

        plt.plot(self.hist['train_f1'], label="Train")
        #plt.plot(self.hist['val_f1'], label="Validation")
        plt.legend()
        plt.savefig(f"./plots/f1_score/f1_score_{self.fold}.png")
        #plt.show()
    
    def save_model(self, n_epoch, save_path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "best_f1_score": self.best_f_score,
                "n_epoch": n_epoch,
            },
            save_path,
        )
    
    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)