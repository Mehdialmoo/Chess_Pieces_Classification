# ***Chess Pieces Classification***
![](./Data/PIC/OIG.jpg)
## **Abstract**

**keywords:** 
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
      |      |___test_loading_data.py
      |      |___test_utilities.py
      |      |___test_utilities.py
      |___Data
      |___README.md
      |___ENVIRONMENT.yml
      |___LICENCE.md
      |___RubberDuck.md
                     
```
* some files have been hidden beacuse of security reasons  the codes related to such files are commented.

## **1. model files**
this project consists of four main files that all are mainly inside [vars](./vars/) folder:
### 1.1. [loading_data](./vars/loading_data.py) : 
This file is responsible for data augmentation using pytorch transformers and splitting datasets. This python file contains a custom PyTorch Dataset and LightningDataModule class for loading and processing a chess dataset. The ChessDB class inherits from both Dataset and LightningDataModule to load the data and provide methods for splitting the data into training, validation, and test sets.The class takes in a directory path to the chess dataset, a PyTorch transform to apply to the data, and a batch size for training.

The class has several methods:

* The __init__ dunder method initializes the class variables, including the dataset and dataset labels.
* The __len__ dunder method returns the length of the dataset.
* The __getitem__ dunder method returns a single data sample and its corresponding label at a given index.
* The data_loader method loads the raw data and labels from the dataset directory.
* The db_split method splits the dataset into training, validation, and test sets based on user-specified ratios. It also creates a bar plot of the number of samples in each dataset split.
* The train_dataloader, valid_dataloader, and test_dataloader methods return the respective PyTorch DataLoader objects for the training, validation, and test sets.

Note: The code assumes that the chess dataset is organized into subdirectories for each class, and each subdirectory contains the images for that class. """

### 1.2. [model](./vars/model.py) :
This file defines a convolutional neural network (CNN) using the PyTorch library and the LightningModule framework from PyTorch Lightning that is The network is designed for image classification task, firstly it defines the ConvolutionalNetwork class, which inherits from LightningModule.
* the __init__ method, define the layers of the CNN:
   - self.conv1: A 2D convolutional layer with 3 input channels (for RGB images), 6 output channels, a kernel size of 3x3, and stride of 1.
   - self.conv2: A 2D convolutional layer with 6 input channels, 16 output channels, a kernel size of 3x3, and stride of 1.
   - self.fc1: A fully connected (linear) layer with 16 * 54 * 54 input neurons (flattened feature map size) and 120 output neurons.
   - self.fc2: A fully connected (linear) layer with 120 input neurons and 84 output neurons.
   - self.fc3: A fully connected (linear) layer with 84 input neurons and 20 output neurons.
   - self.fc4: A fully connected (linear) layer with 20 input neurons and len(labels) output neurons, where labels is an input argument representing the number of classes in the classification task.
* In the forward method, define the forward pass of the network:
Apply ReLU activation to the output of self.conv1.
Apply 2x2 max pooling to the output of self.conv1.
Apply ReLU activation to the output of self.conv2.
Apply 2x2 max pooling to the output of self.conv2.
Flatten the feature map.
Apply ReLU activation to the output of self.fc1, self.fc2, and self.fc3.
Apply log softmax activation to the output of self.fc4 to get the class probabilities.
* In the configure_optimizers method, define the optimizer for training:
Use Adam optimizer with a learning rate of 0.001.
* In the training_step method, define the training step:
Compute the model's output (class probabilities) using the forward method.
Calculate the cross-entropy loss.
Calculate the accuracy by comparing the predicted class with the ground truth.
Log the training loss and accuracy.
* In the validation_step method, define the validation step:
Compute the model's output (class probabilities) using the forward method.
Calculate the cross-entropy loss.
Calculate the accuracy by comparing the predicted class with the ground truth.
Log the validation loss and accuracy.
* In the test_step method, define the test step:
Compute the model's output (class probabilities) using the forward method.
Calculate the cross-entropy loss.
Calculate the accuracy by comparing the predicted class with the ground truth.
Log the test loss and accuracy.

Note: This file does not include data loading but uses the [loading_data](./vars/loading_data.py) to do so. for training loop, it uses Pytorch_lightning fit method to train the model that are included in [model_run](./vars/model_run.py) file , other needed functions as like saving and loading log checkpoints and  validating and testing are implemented using PyTorch Lightning in [model_run](./vars/model_run.py) file, but first lets discuss why PyTorch Lightning and CNN are use for this project.

### 1.2.1 why CNN Model:
A Convolutional Neural Network (CNN) is a type of artificial neural network that is commonly used for image processing, classification, and recognition tasks[(Sharma and Phonsa 2021)](). CNNs are designed to automatically and adaptively learn spatial hierarchies of features from images, which makes them particularly well-suited for tasks such as object detection, image segmentation, and image classification.[(Sharma and Phonsa 2021)]()

In the context of chess pieces, a CNN can be used to process images of chess pieces and classify them into different categories, such as pawn, rook, knight, bishop, queen, and king. This can be useful in a variety of applications, such as automated chess analysis, chess notation conversion, and chess game analysis.

The reason why CNNs are good for processing chess piece images is because they can effectively learn and extract features that are relevant for chess piece classification [(Quintana et al. 2020)](). For example, a CNN can learn to recognize the distinct shapes and patterns of different chess pieces, such as the cross-shaped pattern of a rook or the curved shape of a knight. Additionally, CNNs can learn to be invariant to variations in image scale, orientation, and lighting conditions, which can help improve the robustness and accuracy of chess piece classification[(Yamashita et al. 2018)]().

Overall, CNNs provide a powerful and flexible framework for processing and classifying chess piece images, and can help enable a wide range of chess-related applications and analyses.

### 1.2.2 why Pytorch_lightning
In the context of chess piece classification, a CNN can be used to process images of chess pieces and classify them into different categories, such as pawn, rook, knight, bishop, queen, and king. PyTorch Lightning can simplify the training process by automating tasks such as logging of training metrics, saving and loading checkpoints, and progress tracking. This enables you to focus on building the CNN model and improving its accuracy[(Falcon 2020)]().

 
* multiple features: It offers lightweight features such as the option to easily manage training, testing , loading previous checkpoints based on "latest train", "besd accuracy" and etc. on a distributed system, using automatic mixed precision for better speed, and accuracy.[(Maurya et al. 2023)]()

* Automatic Integration of Training Logging: PyTorch Lightning simplifies the training process by automating tasks such as logging of training metrics, saving and loading checkpoints, and progress tracking. This enables you to focus on building the deep learning model[(Maurya et al. 2023)]().

* advance Model logging loading: PyTorch Lightning offers an integrated trainer that allows you to easily save your models by weights and hyperparameters metrics this enalbles the user to load the model fully or only load the weights and continue with the training or testing.
alsother are option to load the latest log or the best log from accuracy aspect. This simplifies the process of debugging your deep learning model[(Maurya et al. 2023)]().

* Seamless Extension of Models: PyTorch Lightning makes it easy to extend the capabilities of your model by offering pre-built models or easier  and components. This enables you to easily plug-and-play various model architectures, layers, and callbacks into your model[(Maurya et al. 2023)]()

* Integration with Existing Code: PyTorch Lightning can be easily integrated with existing PyTorch code. This ensures that you can utilize the features and functionalities provided by PyTorch Lightning while still maintaining compatibility with your existing codebase[(Maurya et al. 2023)]()

In conclusion, using PyTorch Lightning for chess piece classification provides several advantages such as easy integration with existing PyTorch code, automatic training, validating, testing and evaluating, plus saving and loading logs, and simplified model debugging. This ultimately enables you to focus on building and improving your deep learning model without having to worry about the intricacies of the underlying framework.
   
### 1.3. [model_run](./vars/model_run.py) :

This file defines a class model_run that provides a convenient interface for non-developers to train and test a convolutional neural network (CNN) on a dataset. The class initializes with various attributes required for training the model, such as the data directory, transformer, batch size, ratio for splitting the data, number of epochs, and learning rate.

The class has several methods:

* setup_data method This uses the loading_data module to load and split the data into training, validation, and testing sets.
* setup_model method initializes a new instance of the ConvolutionalNetwork class and checks if a GPU is available for training.
* load_model: This method loads the weights of a previously trained model from a checkpoint.
* load_full: This method loads the entire training state, including the optimizer and scheduler, from a checkpoint.
* train_run: This method trains the model on the training set using the PyTorch Lightning Trainer class.
* validation_run: This method validates the current state of the model against the validation set.
* test_run: This method tests the trained model against the testing set.
* evaluation: This method evaluates the performance of the model using metrics defined during initialization.

Note: The class also includes an instance of the EarlyStopping callback from PyTorch Lightning, which stops training if the validation loss does not improve for a certain number of epochs.

Note: The code includes commented-out lines for using Weights & Biases (WandB) for logging training progress. If you want to use WandB, you can uncomment those lines and configure your WandB account accordingly.
### 1.3. [utilities](./vars/utilities.py) :
this file comtains methods that can be used to pre-process a dataset of images for machine learning purposes. Here's a brief description of what each function does:

* create_CSV function creates a CSV file that contains the image paths and their corresponding labels.
* pre_process function resizes all the images in the dataset to a fixed size of 224x224 pixels (works like transformers but perminant) and it takes The dir_list parameter that is a list of directories, where each directory contains images belonging to a particular class.
* plot_bar function generates a bar chart that shows the number of images in each class.

* plot_img function generates a grid of images that shows samples from each class. The dir_list parameter is a list of directories, where each directory contains images belonging to a particular class. The labels parameter is a list of strings that contain the names of the classes.

   Note: The image_no parameter is an optional parameter that specifies the number of images to be displayed for each class.

Overall, these functions can be used to pre-process the chess image dataset and prepare it for loading and training a machine learning model.

another important part of this project is unit testing files that are including in [test](./test/) folder. other files include [data](./Data/) folder that have been disscussed in the following sectons


other parts of this  
# Setup
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

# Refrences:
- Falcon, W., 2020. From PyTorch to PyTorch Lightning : a Gentle Introduction [online]. Medium. Available from: https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09.Maurya, A., Mocholí, C., 

- Pablo, J., Lienen, M. and Shenoy, N., 2023. PyTorch Lightning 2.0 [online]. Lightning AI. Available from: https://lightning.ai/releases/2.0.0/ [Accessed 24 Jan 2024].
- Quintana, D., Andrea Calderón García and Prieto-Matías, M., 2020. LiveChess2FEN: a Framework for Classifying Chess Pieces based on CNNs. ArXiv (Cornell University).
- Sharma, A. and Phonsa, G., 2021. Image Classification Using CNN. SSRN Electronic Journal.
- Yamashita, R., Nishio, M., Do, R. K. G. and Togashi, K., 2018. Convolutional Neural networks: an Overview and Application in Radiology. Insights into Imaging [online], 9 (4), 611–629. Available from: https://insightsimaging.springeropen.com/articles/10.1007/s13244-018-0639-9.