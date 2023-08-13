import time
import numpy as np
import pandas as pd
import torch
from torch.utils import data as torch_data
from torch.nn import functional as F
import os
import albumentations as A
from sklearn.model_selection import StratifiedKFold
import argparse

from model import Model
from config import CFG
from utils import LossMeter, AccMeter, F1score, seed_everything, get_settings
from trainer import Trainer
from data_retriever import DataRetriever, TestDataRetriever

import wandb
wandb.login(key='a2a7828ed68b3cba08f2703971162138c680b664')
r =1
run = wandb.init(
      project="Kaggle LB-2 sanity run", 
      name=f"experiment_{1}", 
      config={
      "learning_rate": 0.0001,
      "architecture": "CNN-LSTM",
      "dataset": "MICAA MRI",
      "epochs": CFG.n_epochs,
      })
    

def main(device, settings):
    #df = pd.read_csv(os.path.join(settings['DATA_PATH'], "train_labels.csv"))
    
    dlt = []
    empty_fld = [109, 123, 709]
    df = pd.read_csv("./input/train_labels.csv")
    X = df['BraTS21ID'].values
    Y = df['MGMT_value'].values

    for i in empty_fld:
        j = np.where(X == i)
        dlt.append(j)
        X = np.delete(X, j)
        
    Y = np.delete(Y,dlt)
    
    train_transform = A.Compose([
                                A.HorizontalFlip(p=0.5),
                                A.ShiftScaleRotate(
                                    shift_limit=0.0625, 
                                    scale_limit=0.1, 
                                    rotate_limit=10, 
                                    p=0.5
                                ),
                                A.RandomBrightnessContrast(p=0.5),
                            ])
    skf = StratifiedKFold(n_splits=CFG.n_fold)
    #t = df['MGMT_value']

    start_time = time.time()
    losses = []
    scores = []
    f_scores = []
    test_fscore = []


    for fold, (train_index, val_index) in enumerate(skf.split(np.zeros(len(Y)), Y), 1):
        
        xtrain = X[train_index]
        ytrain = Y[train_index]
        xtest = X[val_index]
        ytest = Y[val_index]
        
        print('-'*30)
        print(f"Fold {fold}")

        train_retriever = DataRetriever(
            settings['TRAIN_DATA_PATH'],
            xtrain, #train_df["BraTS21ID"].values, 
            ytrain, #train_df["MGMT_value"].values,
            n_frames=CFG.n_frames,
            img_size=CFG.img_size,
            transform=train_transform
        )
        
        val_retriever = DataRetriever(
            settings['TRAIN_DATA_PATH'],
            xtest, #val_df["BraTS21ID"].values, 
            ytest, #val_df["MGMT_value"].values,
            n_frames=CFG.n_frames,
            img_size=CFG.img_size
        )

        train_loader = torch_data.DataLoader(
            train_retriever,
            batch_size=8,
            shuffle=True,
            num_workers=8,
        )
        
        valid_loader = torch_data.DataLoader(
            val_retriever, 
            batch_size=8,
            shuffle=False,
            num_workers=8,
        )
        
        model = Model(cnn_path=settings['PRETRAINED_CHECKPOINT_PATH'])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = F.binary_cross_entropy_with_logits
        trainer = Trainer(
            model, 
            device, 
            optimizer, 
            criterion, 
            LossMeter, 
            AccMeter,
            fold
        )
        
        loss, score, f_score = trainer.fit(
            CFG.n_epochs, 
            train_loader, 
            valid_loader, 
            os.path.join(settings["MODEL_CHECKPOINT_DIR"], f"best-model-{fold}.pth"), 
            100,
        )

        #losses.append(loss)
        #scores.append(score)
        #f_scores.append(f_score)
        

    elapsed_time = time.time() - start_time
    '''wandb.log({
         'Avg Test f1 score': np.mean(test_fscore),
         'Avg Train f1 score': np.mean(f_scores)
         })
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
    '''
    #print('Avg loss {:.5f}'.format(np.mean(losses)))
    #print('Avg score {:.5f}'.format(np.mean(scores)))
    #print('Avg Train f1_score {:.5f}'.format(np.mean(f_scores)))
    #print('Avg Test f1_score {:.5f}'.format(np.mean(test_fscore)))

    
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting-path', default='../settings/SETTINGS_kaggle.json')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    settings = get_settings(args.setting_path)
    device = torch.device("cuda")
    seed_everything(args.seed)
    main(device, settings)