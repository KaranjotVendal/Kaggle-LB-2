import time
import numpy as np
import pandas as pd

import torch
import torchmetrics

from config import CFG
#from gradcam import gradcam
import wandb


def evaluate(model, test_loader, fold, mod, device):
    test_time = time.time()

    #model = Res18GRU(config.MODEL, config.NUM_CLASSES)
    #model.to(config.DEVICE)
    checkpoint = torch.load(f'./models/weights/Efficientb0_{mod}_{fold}.pth')
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
    test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=2, average='macro')       
    test_auroc = torchmetrics.AUROC(task="multiclass", num_classes=2)

    test_targets = []
    preds = []    
    for e, batch in enumerate(test_loader):
        #print(f"{e}/{len(test_loader)}", end="\r")
        with torch.no_grad():
    
            features = batch['X'].to(device)
            targets = batch['y'].to(device)
            #print(org)
            
            logits, probs = model(features)
            #predicted_class = probs.argmax(dim=1)
            
            #test_pred.append(predicted_class)
            test_targets.append(targets)
            preds.append(probs)
            #print('probs shape:',probs.shape)
            #print('targets shape:', targets.shape)
            
            #print('------BATCH ENDING-------')

    #test_pred = torch.cat(test_pred).flatten()
    test_targets = torch.cat(test_targets).flatten()
    preds = torch.cat(preds)

    acc = test_acc(preds.cpu(), test_targets.cpu())
    f1 = test_f1(preds.cpu(), test_targets.cpu())
    auroc = test_auroc(preds.cpu(), test_targets.cpu())

    wandb.log({
                'test acc': acc,
                'test f1': f1,
                'test AUROC': auroc
                })

        #gradcam
        #gradcam(test_loader, fold)
        
    elapsed_time = time.time() - test_time
    
    print(f'\nInference complete for {mod} in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s | Accuracy: {acc:.3f}% | F1 Score: {f1:.4f} | AUROC: {auroc:.4f}')


    return acc.item(), f1.item(), auroc.item() 