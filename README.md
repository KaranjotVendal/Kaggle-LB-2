# RSNA-MICCAI Brain Tumor Radiogenomic Classification 2nd place solution
[Competition link](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification)

[Solution write-up](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/280033)

This is a refactored repository from 2 notebooks:
- [Training notebook](https://www.kaggle.com/minhnhatphan/rnsa-21-cnn-lstm-train/notebook)
- [Inference notebook](https://www.kaggle.com/minhnhatphan/rnsa-21-cnn-lstm-inference)

To run this repository with ease, please refer to [this Kaggle notebook](https://www.kaggle.com/minhnhatphan/rnsa21-cnn-lstm-github/notebook)

If you want to run locally, there are certain steps you need to make:
 
1. Install dependencies via `pip install - r requirements.txt`
2. Download processed train dataset from [here](https://www.kaggle.com/jonathanbesomi/rsna-miccai-png). This is a cleaned PNG public dataset of the competition, we use this dataset in the training phase
3. Download the [dataset](https://drive.google.com/file/d/1b1cbJojNYjCLrxbz0KG8ca3tiYPt_3B0/view?usp=sharing)
4. Download pretrained EfficientNet-B0 weights [here](https://www.kaggle.com/hmendonca/efficientnet-pytorch). 
5. Change/create a setting json file that has the following keys:

    * DATA_PATH: Path to the competition dataset in step 3
    * TRAIN_DATA_PATH: Path to the train folder of the step 2 cleaned dataset
    * TEST_DATA_PATH: Path to the test folder of the step 3 competition dataset
    * MODEL_CHECKPOINT_DIR: Directory where all models are saved
    * PRETRAINED_CHECKPOINT_PATH: Location of the pretrained EfficientNet model
    * SUBMISSION_DIR: Where the submission.csv is written

    File [SETTINGS_kaggle.json](settings/SETTINGS_kaggle.json) is an example to construct your custom SETTINGS file
6. Run training script `python ./src/train.py --setting-path {PATH-TO-YOUR-SETTINGS.JSON}`


7. Plots and weights

Plots and weights of the conducted experiments can be found here. [Link](https://drive.google.com/drive/folders/1FHvKqMEwIKz-hA0cazLJTLhhqr3gAL27?usp=drive_link)

 
If you have any questions, feel free to contact me at karanjotvendal@gmail.com


## ORIGINAL Authors

- [@FirsaBaba](https://github.com/minhnhatphan/rnsa21-cnn-lstm)
nminh238@gmail.com
