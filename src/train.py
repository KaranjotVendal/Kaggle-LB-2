import time
import numpy as np
import pandas as pd
import torch
from torch.utils import data as torch_data
from torch.nn import functional as F
import os
import albumentations as A
#from sklearn.model_selection import train_test_split, StratifiedKFold
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
      # Set the project where this run will be logged
      project="Kaggle LB-2 Baseline run", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"experiment_{2}", 
      # Track hyperparameters and run metadata
      config={
      "learning_rate": 0.0001,
      "architecture": "CNN-LSTM",
      "dataset": "MICAA MRI",
      "epochs": CFG.n_epochs,
      })
    

def test(device, settings, fold):
        start = time.time()
        
        model = Model()
        model.to(device)
        checkpoint = torch.load(os.path.join(settings['MODEL_CHECKPOINT_DIR'], f'best-model-{fold}.pth'))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
            
        folds_xtest = np.load('./input/folds/xtest.npy', allow_pickle=True)
        folds_ytest = np.load('./input/folds/ytest.npy', allow_pickle=True)

        xtest = folds_xtest[fold - 1]
        ytest = folds_ytest[fold - 1]

        test_data_retriever = TestDataRetriever(
            settings['TEST_DATA_PATH'],
            xtest, #submission["BraTS21ID"].values,
            ytest,
            n_frames=CFG.n_frames,
            img_size=CFG.img_size,
        )

        test_loader = torch_data.DataLoader(
            test_data_retriever,
            batch_size=4,
            shuffle=False,
            num_workers=8,
        )

        
        #pred = []
        #true = []
        ff_score = F1score()
        for e, batch in enumerate(test_loader):                                 
                                                                                                                                                        
            with torch.no_grad():
                X = batch["X"].to(device = 'cuda')
                targets = batch["y"].to(device='cuda')

                outputs = model(X).squeeze(1)
                
                f_score = ff_score.update(targets, outputs.detach())
                #f.append(f_score.item())
                #pred.append(outputs.detach().cpu().numpy())
                #true.append(targets.cpu.numpy())
    
            message = "Test step: [batch {}:{}], f1 score: {}"
            info_message(message, e, len(test_loader), f_score)
            
        f1 = ff_score.get_score()    
        elapsed = time.time() - start
        wandb.log({'test_f1_score': f1})
        print('\nTesting complete in {:.0f}m {:.0f}s'.format(elapsed // 60, elapsed % 60))
        print('f1 score for test fold {}: {:.5f}'.format(fold, f1))

        return f1

def main(device, settings):

    #df = pd.read_csv(os.path.join(settings['DATA_PATH'], "train_labels.csv"))
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
    '''skf = StratifiedKFold(n_splits=CFG.n_fold)
    t = df['MGMT_value']

    start_time = time.time()
    losses = []
    scores = []
    for fold, (train_index, val_index) in enumerate(skf.split(np.zeros(len(t)), t), 1):
        print('-'*30)
        print(f"Fold {fold}")
        
        train_df = df.loc[train_index]
        val_df = df.loc[val_index]'''
    '''
    start_time = time.time()
    
    losses = []
    scores = []
    kfold = []
    for _ in range(CFG.n_fold):
        xtrain, xtest, ytrain, ytest = train_test_split(df['BraTS21ID'], df['MGMT_value'], test_size=0.2, shuffle=True, stratify=df['MGMT_value'])
        kfold.append([xtrain, xtest, ytrain, ytest])

        fold = _
        print('-'*30)
        print(f"Fold {fold+1}")'''
    
    
    start_time = time.time()
    losses = []
    scores = []
    f_scores = []
    test_fscore = []

    for _ in range(CFG.n_fold):
        fold = _+1

        folds_xtrain = np.load('./input/folds/xtrain.npy', allow_pickle=True)
        folds_xtest = np.load('./input/folds/xtest.npy', allow_pickle=True)
        folds_ytrain = np.load('./input/folds/ytrain.npy', allow_pickle=True)
        folds_ytest = np.load('./input/folds/ytest.npy', allow_pickle=True)

        xtrain = folds_xtrain[_]
        ytrain = folds_ytrain[_]
        xtest = folds_xtest[_]
        ytest = folds_ytest[_]
        
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
        '''
        valid_loader = torch_data.DataLoader(
            val_retriever, 
            batch_size=8,
            shuffle=False,
            num_workers=8,
        )'''
        
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
            F1score,
            fold
        )
        
        loss, score, f_score = trainer.fit(
            CFG.n_epochs, 
            train_loader, 
            #valid_loader, 
            os.path.join(settings["MODEL_CHECKPOINT_DIR"], f"best-model-{fold}.pth"), 
            100,
        )
        losses.append(loss)
        scores.append(score)
        f_scores.append(f_score)
        trainer.plot_loss()
        trainer.plot_score()
        trainer.plot_fscore()


        #test
        test_f = test(device, settings, fold)
        test_fscore.append(test_f)
    elapsed_time = time.time() - start_time
    wandb.log({
         'Avg Test f1 score': np.mean(test_fscore),
         'Avg Train f1 score': np.mean(f_scores)
         })
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
    print('Avg loss {:.5f}'.format(np.mean(losses)))
    print('Avg score {:.5f}'.format(np.mean(scores)))
    print('Avg Train f1_score {:.5f}'.format(np.mean(f_scores)))
    print('Avg Test f1_score {:.5f}'.format(np.mean(test_fscore)))

    
    wandb.finish()

@staticmethod
def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting-path', default='../settings/SETTINGS_kaggle.json')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    settings = get_settings(args.setting_path)
    device = torch.device("cuda")
    seed_everything(args.seed)
    main(device, settings)