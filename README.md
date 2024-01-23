# ***Chess Pieces Classification***

## **Introduction**
This repository contains a simple Convolutional Neural Network (CNN) implemented using PyTorch and PyTorch Lightning to classify chess pieces. The model is trained on a custom dataset of chess pieces images.
This project aims to classify the chess pieces (King, Queen, Bishop, Knight, Rook).
this project Files are as follows:
```
project
      |___vars
      |      |___loading_data.py
      |      |___model.py
      |      |___model_run.py
      |      |___utilities.py
      |___test
      |___Data
      |___README.md
      |___ENVIRONMENT.yml
      |___LICENCE.md
      |___RubberDuck.md
                     
```
* some files have been hidden beacuse of security reasons  the codes related to such files are commented.

## **1. code implementation explained**
this project consists of four main files that all are mainly inside [vars](./vars/) folder:
* loading_data : 

* model : 
   
* model_run :
* utilities :
another important part of this project is unit testing files that are including in [test](./test/) folder. other files include [data](./Data/) folder that have been disscussed in the following sectons
   ### 1.1. CNN Model:
   ### 1.2. Pytorch_lightning


## **2. Setup**
To set up the project, follow one of these steps:
- Download
- Copy code
- use git clone:
    ```bash
    git clone git@github.com:Mehdialmoo/s5602288_Software_Eng.git
    cd s5602288_Software_Eng
    ```
 - Install the required packages:
 - use the Environment file:
   ```bash
   conda create ENVIRONMENT.yml
   conda activate SE_ENV
   ```
 - In addition to the environment this model has been trained on GPU, to train and fit the model on GPU its better to unistall the normal pytorch and install pytorch with GPU support:
   + to do such run the following commands :
      * firstly remove the previous Pytorch
      ```bash
      pip3 uninstall torch
      ```
      * then install pytorch with GPU support: 
      ```bash
      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
      ```
   + if you want to train and use th model on CPU:
      * just visit model_run in vars folder and go to setup_model and comment the following line
      ```python
      torch.device("cuda" if torch.cuda.is_available() else "cpu")
      ```
      * after that visit train_run function in model_run and change the accelerator to "cpu"
      ```python
      self.trainer = pl.Trainer(max_epochs=self._epoch, accelerator='cpu', callbacks=EarlyStop,default_root_dir="./")
      ```
now that everything is ready we need to set up the dataset.
## 3. Dataset
The dataset consists of chess pieces images collected from various sources. It contains 6 classes: King, Queen, Rook, Bishop, Knight, and Pawn. The dataset is split into training and testing sets with a ratio of 60:20:20. this dataset is a combination of  [Chessman Dataset](https://www.kaggle.com/datasets/niteshfre/chessman-image-dataset) & [Chess Pieces Detection Images Dataset](https://www.kaggle.com/datasets/anshulmehtakaggl/chess-pieces-detection-images-dataset?rvi=1). these to datasets are merged and preproccessed, including resizing and renaming and creating a CSV file including labels and addresses, into [ChessDB](https://drive.google.com/drive/folders/1ItkRGV0xCaBI1rjer3ykI71FGjX5bei1?usp=drive_link) to download each of these data sets just click on the name of the datasets. if you are not usinf [ChessDB](https://drive.google.com/drive/folders/1ItkRGV0xCaBI1rjer3ykI71FGjX5bei1?usp=drive_link) , it is encouraged to run the preprocessing functions from [utilities.py](./vars/utilities.py) file.

