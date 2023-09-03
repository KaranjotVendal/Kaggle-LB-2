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
from utils import LossMeter, AccMeter, seed_everything, get_settings, update_metrics, save_metrics_to_json
from trainer import Trainer
from data_retriever import DataRetriever
from eval import evaluate
from plotting import plot_train_valid_fold, plot_train_valid_all_fold, plot_test_metrics

if CFG.WANDB:
    import wandb
    wandb.login(key='a2a7828ed68b3cba08f2703971162138c680b664')

    r ="baseline"
    run = wandb.init(
        project="Kaggle LB-2", 
        name=f"experiment_{r}", 
        config={
        "learning_rate": 0.0001,
        "architecture": "CNN-LSTM",
        "dataset": "MICAA MRI",
        "epochs": CFG.n_epochs,
        "batch size": CFG.batch_size
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
    skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=123)
    
    start_time = time.time()
    
    fold_acc = []
    fold_auroc = []
    fold_f1 = []

    metrics = {}

    for fold, (train_index, val_index) in enumerate(skf.split(np.zeros(len(Y)), Y), 1):
        
        metrics[fold] = {
        'train': {'loss': [], 'acc': [], 'f1': [], 'auroc': []},
        'valid': {'loss': [], 'acc': [], 'f1': [], 'auroc': []},
        'test': {'acc': [], 'f1': [], 'auroc': []}
        }
        
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
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
        )
        
        valid_loader = torch_data.DataLoader(
            val_retriever, 
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
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
        
        trainer.fit(
            CFG.n_epochs, 
            train_loader, 
            valid_loader, 
            os.path.join(settings["MODEL_CHECKPOINT_DIR"], f'Efficientb0_{CFG.MOD}_{fold}.pth'), 
            100,
        )

        acc, f1, auroc = evaluate(model,
                                valid_loader,
                                fold,
                                CFG.MOD,
                                CFG.DEVICE)

        for value in trainer.hist['train_loss']:
            update_metrics(metrics, fold, 'train', 'loss', value)
    
        for value in trainer.hist['train_acc']:
            update_metrics(metrics, fold, 'train', 'acc', value)
    
        for value in trainer.hist['train_f1']:
            update_metrics(metrics, fold, 'train', 'f1', value)
    
        for value in trainer.hist['train_auroc']:
            update_metrics(metrics, fold, 'train', 'auroc', value)
    
        for value in trainer.hist['val_loss']:
            update_metrics(metrics, fold, 'valid', 'loss', value)
    
        for value in trainer.hist['val_acc']:
            update_metrics(metrics, fold, 'valid', 'acc', value)
    
        for value in trainer.hist['val_f1']:
            update_metrics(metrics, fold, 'valid', 'f1', value)
    
        for value in trainer.hist['val_auroc']:
            update_metrics(metrics, fold, 'valid', 'auroc', value)

        update_metrics(metrics, fold, 'test', 'acc', acc)
        update_metrics(metrics, fold, 'test', 'f1', f1)
        update_metrics(metrics, fold, 'test', 'auroc', auroc)
        
        fold_acc.append(acc)
        fold_f1.append(f1)
        fold_auroc.append(auroc)

    json_path = save_metrics_to_json(metrics, 'Efficientb0')
    
    #plotting loss
    plot_train_valid_fold(json_path, 'loss')
    plot_train_valid_all_fold(json_path, 'loss')
    
    #plotting acc
    plot_train_valid_fold(json_path, 'acc')
    plot_train_valid_all_fold(json_path, 'acc')
    plot_test_metrics(json_path, 'acc')


    #plotting f1
    plot_train_valid_fold(json_path, 'f1')
    plot_train_valid_all_fold(json_path, 'f1')
    plot_test_metrics(json_path, 'f1')

    #plotting auroc
    plot_train_valid_fold(json_path, 'auroc')
    plot_train_valid_all_fold(json_path, 'auroc')
    plot_test_metrics(json_path, 'auroc')
   
    elapsed_time = time.time() - start_time
    
    
    print('\nCross validation loop complete for {} in {:.0f}m {:.0f}s'.format(CFG.MOD, elapsed_time // 60, elapsed_time % 60))
    print('\nfold accuracy:', fold_acc)
    print('\nfold f1_score:',fold_f1)
    print('\nfold auroc:', fold_auroc)
    print('\nStd F1 score:', np.std(np.array(fold_f1)))
    print('\nAVG performance of model:', np.mean(np.array(fold_f1)))

    if CFG.WANDB:
        wandb.log({
        'Avg performance F1': np.mean(np.array(fold_f1)),
        'Std f1 score': np.std(np.array(fold_f1)),
        'Avg performance acc': np.mean(np.array(fold_acc)),
        'Std acc score': np.std(np.array(fold_acc)),
        'Avg performance auroc': np.mean(np.array(fold_auroc)),
        'Std auroc score': np.std(np.array(fold_auroc)),
        })
    
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting-path', default='./settings/SETTINGS_kaggle.json')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    settings = get_settings(args.setting_path)
    device = torch.device("cuda")
    seed_everything(args.seed)
    main(device, settings)