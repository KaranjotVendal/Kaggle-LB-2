import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchmetrics
from tqdm import tqdm
import wandb

def test(self, test_loader):
        test_time = time.time()
        test_loss = self.loss_meter()
        test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(self.device)
        test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=2, average='macro').to(self.device)       
        test_auroc = torchmetrics.AUROC(task="multiclass", num_classes=2).to(self.device)
        test_pred = []
        test_targets = []
        for idx, batch in enumerate(test_loader):
            
            self.model.eval()
            with torch.no_grad():
                features = batch['X'].to(self.device)
                targets = batch['y'].to(self.device)
                
                org = batch['org']
                print(org)
                
                logits, probs = self.model(features, org)
                test_pred.append(probs)
                test_targets.append(targets)
        
        test_pred = test_pred.flatten

                
                #loss = self.criterion(logits, targets)
            
                
                
                #test_loss.update(loss.detach().item())
                #test_acc.update(probs.detach(), targets)
                #test_f1.update(probs.detach(), targets)
                #test_auroc.update(probs.detach(), targets)
                print('------BATCH ENDING-------')

            _loss = test_loss.avg
            _acc = test_acc.compute()
            _f1 = test_f1.compute()
            _roc = test_auroc.compute()

            wandb.log({'test loss': _loss,
                      'test acc': _acc,
                      'test f1_score': _f1,
                      'test AUROC': _roc
                     })
    
            #test_acc.reset()
            #test_f1.reset()
            #test_auroc.reset()
            
            ##self.hist['test_loss'].append(_loss)
            #self.hist['test_acc'].append(_acc)
            ##self.hist['test_f1'].append(_f1)
            #self.hist['test_auroc'].append(_roc)
            

        avg_loss = torch.mean(torch.tensor(self.hist['test_loss']))
        avg_acc = torch.mean(torch.tensor(self.hist['test_acc']))
        avg_f1 = torch.mean(torch.tensor(self.hist['test_f1']))
        avg_auroc = torch.mean(torch.tensor(self.hist['test_auroc']))

        print(f"Testing Time: {(time.time() - test_time)/60:.2f} min | Avg Loss: {avg_loss:.5f} | Avg Accuracy: {avg_acc:.2f}% | Avg F1 Score: {avg_f1:.4f} | Avg AUROC: {avg_auroc:.4f}")
        
        return avg_loss, avg_acc, avg_f1, avg_auroc