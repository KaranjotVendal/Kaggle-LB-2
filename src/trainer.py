import time
import numpy as np
import torch
import torchmetrics
import wandb
from config import CFG

class Trainer:
    def __init__(
        self, 
        model, 
        device, 
        optimizer, 
        criterion, 
        loss_meter, 
        score_meter,
        fold
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.loss_meter = loss_meter
        self.score_meter = score_meter
        self.hist = {'val_loss':[],
                     'val_acc':[],
                     'val_f1':[],
                     'val_auroc':[],
                     'train_loss':[],
                     'train_acc':[],
                     'train_f1': [],
                     'train_auroc': [],
                    }
        
        self.best_valid_score = -np.inf
        self.best_valid_loss = np.inf
        self.best_f_score = 0
        self.best_valid_auroc = 0
        self.n_patience = 0
        self.fold = fold
        
        self.messages = {
            "epoch": "[Epoch {}: {}] loss: {:.5f}, score: {:.5f}, f1 score: {:.5f}, AUROC: {:.5f} time: {} s",
            "checkpoint": "The AUROC improved from {:.5f} to {:.5f}. score: {:.5f}, f1 score: {:.5f}. Save model to '{}'",
            "patience": "\nTrain score didn't improve last {} epochs."
        }
    
    def fit(self, epochs, train_loader, valid_loader, save_path, patience):        
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)
            
            train_loss, train_score, train_f_score, train_auroc, train_time = self.train_epoch(train_loader)
            
            valid_loss, valid_score, valid_f_score, valid_auroc, valid_time = self.valid_epoch(valid_loader)
            
            self.hist['val_loss'].append(valid_loss)
            self.hist['train_loss'].append(train_loss)
            self.hist['val_acc'].append(valid_score)
            self.hist['train_acc'].append(train_score)
            self.hist['val_f1'].append(valid_f_score)
            self.hist['train_f1'].append(train_f_score)
            self.hist['val_auroc'].append(valid_f_score)
            self.hist['train_auroc'].append(train_f_score)
            
            if CFG.WANDB:
                wandb.log({'val_loss': valid_loss,
                        'val_acc': valid_score,
                        'val_f1': valid_f_score,
                        'val_auroc': valid_auroc,
                        'train_loss': train_loss,
                        'train_score': train_score,
                        'train_f1': train_f_score,
                        'train_auroc': train_auroc,
                        })
                
            self.info_message(
                self.messages["epoch"], "Train", n_epoch, train_loss, train_score, train_f_score, train_auroc, train_time
            )
            
            self.info_message(
                self.messages["epoch"], "Valid", n_epoch, valid_loss, valid_score, valid_f_score, valid_auroc, valid_time
            )

            if self.best_valid_auroc < valid_auroc:
                self.info_message(
                    self.messages["checkpoint"], self.best_valid_auroc, valid_auroc, valid_score, valid_f_score, save_path
                )
                self.best_valid_score = valid_score
                self.best_valid_loss = valid_loss
                self.best_f_score = valid_f_score
                self.best_valid_auroc = valid_auroc
                self.save_model(n_epoch, save_path)
                print(f'checkpoint saved at {save_path}')
                self.n_patience = 0
            else:
                self.n_patience += 1
            
            if self.n_patience >= patience:
                self.info_message(self.messages["patience"], patience)
                break

            
    def train_epoch(self, train_loader):
        self.model.train()
        t = time.time()
        train_loss = self.loss_meter()
        train_acc = torchmetrics.Accuracy(task="binary").to(self.device)
        train_f1 = torchmetrics.F1Score(task="binary").to(self.device)
        train_auroc = torchmetrics.AUROC(task="binary").to(self.device)
            
        for step, batch in enumerate(train_loader, 1):
            X = batch["X"].to(self.device)
            targets = batch["y"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X).squeeze(1)
            
            loss = self.criterion(outputs, targets)
            loss.backward()

            train_loss.update(loss.detach().item())
            train_acc.update(outputs.detach(), targets)
            train_f1.update(outputs.detach(), targets)
            train_auroc.update(outputs.detach(), targets)
                
            self.optimizer.step()
           
        _loss = train_loss.avg
        _acc = train_acc.compute()
        _f1 = train_f1.compute()
        _roc = train_auroc.compute()


        train_acc.reset()
        train_f1.reset()
        train_auroc.reset()
          
        return _loss, _acc, _f1, _roc, int(time.time() - t)
    
    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        valid_loss = self.loss_meter()
        val_acc = torchmetrics.Accuracy(task="binary").to(self.device)
        val_f1 = torchmetrics.F1Score(task="binary").to(self.device)       
        val_auroc = torchmetrics.AUROC(task="binary").to(self.device)

        test_targets = []
        preds = []
        
        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch["X"].to(self.device)
                targets = batch["y"].to(self.device)

                outputs = self.model(X).squeeze(1)
                loss = self.criterion(outputs, targets)

                valid_loss.update(loss.detach().item())
                
                test_targets.append(targets)
                preds.append(outputs.detach())

        test_targets = torch.cat(test_targets).flatten()
        preds = torch.cat(preds)

        _loss = valid_loss.avg
        acc = val_acc(preds, test_targets)
        f1 = val_f1(preds, test_targets)
        auroc = val_auroc(preds, test_targets)              
                
        return _loss, acc, f1, auroc, int(time.time() - t)
    
    
    def save_model(self, n_epoch, save_path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "best_valid_auroc": self.best_valid_score,
            },
            save_path,
        )
    
    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)