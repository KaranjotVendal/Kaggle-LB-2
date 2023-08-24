class CFG:
    WANDB = True
    
    img_size = 112
    n_frames = 10
    
    cnn_features = 256
    lstm_hidden = 32
    
    n_fold = 10
    n_epochs = 15
    
    batch_size = 8
    num_workers = 4

    MOD = 'FLAIR'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")