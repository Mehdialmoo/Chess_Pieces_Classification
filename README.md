# ***Chess Pieces Classification***
![](./Data/PIC/OIG.jpg)
## **Abstract**
This repository contains a simple Convolutional Neural Network (CNN) implemented in PyTorch and PyTorch Lightning for classifying chess pieces. The model is trained on a custom dataset of chess pieces images, and the project includes files for data loading, model definition, training, and evaluation. The goal of this project is to accurately classify chess pieces into six categories: King, Queen, Rook, Bishop, Knight, and Pawn. The project also includes unit testing files and instructions for setting up the environment and dataset.

**keywords: CNN(Convolutional Neural Network), PyTorch, PyTorch Lightning, Chess Dataset, Classififcation** 
## **Introduction**
This repository is a **PyTorch** and **PyTorch Lightning** implementation of a **Convolutional Neural Network (CNN)** that classifies chess pieces into six categories: King, Queen, Rook, Bishop, Knight, and Pawn. The model is trained on a custom dataset of chess piece images, and the project includes files for data loading, model definition, training, and evaluation. The repository is organized into four main files: `loading_data.py`, `model.py`, `model_run.py`, and `utilities.py`, which are responsible for data loading, model definition, model training, and utility functions, respectively. The project also includes unit testing files and instructions for setting up the environment and dataset. The use of CNNs and PyTorch Lightning is discussed, highlighting their benefits for image classification tasks. 
This project aims to classify the chess pieces (King, Queen, Bishop, Knight, Rook).
this project Files are as follows:
```
project
      |___vars
      |      |___loading_data.py
      |      |___model.py
      |      |___model_run.py
      |      |___utilities.py
      |      |___wandb_logger
      |
      |___test
      |      |___test_loading_data.py
      |      |___test_utilities.py
      |      |___test_model_run.py
      |
      |___Data
      |___README.md
      |___ENVIRONMENT.yml
      |___LICENCE.md
      |___RubberDuck.md
                     
```
- *Note: some files have been hidden beacuse of security reasons  the codes related to such files are commented.*
- *Note:`__init__.py` files are only for caliing modules and they are not part of the main architecture.*

## **1. model files**
this project consists of four main files that all are mainly inside [vars](./vars/) folder:
### 1.1. [loading_data](./vars/loading_data.py) : 
This file is responsible for data augmentation using pytorch transformers and splitting datasets. This python file contains a custom PyTorch Dataset and LightningDataModule class for loading and processing a chess dataset. The ChessDB class inherits from both Dataset and LightningDataModule to load the data and provide methods for splitting the data into training, validation, and test sets.The class takes in a directory path to the chess dataset, a PyTorch transform to apply to the data, and a batch size for training.

The class has several methods:

* The `__init__` dunder method initializes the class variables, including the dataset and dataset labels.
* The `__len__` dunder method returns the length of the dataset.
* The `__getitem__` dunder method returns a single data sample and its corresponding label at a given index.
* The data_loader method loads the raw data and labels from the dataset directory.
* The db_split method splits the dataset into training, validation, and test sets based on user-specified ratios. It also creates a bar plot of the number of samples in each dataset split.
* The train_dataloader, valid_dataloader, and test_dataloader methods return the respective PyTorch DataLoader objects for the training, validation, and test sets.

Note: The code assumes that the chess dataset is organized into subdirectories for each class, and each subdirectory contains the images for that class. """

### 1.2. [model](./vars/model.py) :
This file defines a convolutional neural network (CNN) using the PyTorch library and the LightningModule framework from PyTorch Lightning that is The network is designed for image classification task, firstly it defines the ConvolutionalNetwork class, which inherits from LightningModule.
* the `__init__` dunder method, define the layers of the CNN:
   - self.conv1: A 2D convolutional layer with 3 input channels (for RGB images), 6 output channels, a kernel size of 3x3, and stride of 1.
   - self.conv2: A 2D convolutional layer with 6 input channels, 16 output channels, a kernel size of 3x3, and stride of 1.
   - self.fc1: A fully connected (linear) layer with 16 * 54 * 54 input neurons (flattened feature map size) and 120 output neurons.
   - self.fc2: A fully connected (linear) layer with 120 input neurons and 84 output neurons.
   - self.fc3: A fully connected (linear) layer with 84 input neurons and 20 output neurons.
   - self.fc4: A fully connected (linear) layer with 20 input neurons and len(labels) output neurons, where labels is an input argument representing the number of classes in the classification task.
* In the forward method, define the forward pass of the network:
   - Apply ReLU activation to the output of self.conv1.
   - Apply 2x2 max pooling to the output of self.conv1.
   - Apply ReLU activation to the output of self.conv2.
   - Apply 2x2 max pooling to the output of self.conv2.
   - Flatten the feature map.
   - Apply ReLU activation to the output of self.fc1, self.fc2, and self.fc3.
   - Apply log softmax activation to the output of self.fc4 to get the class probabilities.
* In the configure_optimizers method, define the optimizer for training:
defualt value is set to learning rate of 0.001.
* In the training_step method:
   - It computes the model's output (class probabilities) using the forward method.
   - It calculates the cross-entropy loss.
   - It calculates the accuracy by comparing the predicted class with the ground truth.
   - Log the training loss and accuracy.
* In the validation_step method, and the test_step method is the same with training_step but it dosent have epoches to to be done over and over and it dosent return and loss for the gradient dsent to calculate weights.


Note: This file does not include data loading but uses the [loading_data](./vars/loading_data.py) to do so. for training loop, it uses Pytorch_lightning fit method to train the model that are included in [model_run](./vars/model_run.py) file , other needed functions as like saving and loading log checkpoints and  validating and testing are implemented using PyTorch Lightning in [model_run](./vars/model_run.py) file, but first lets discuss why PyTorch Lightning and CNN are use for this project.

### 1.2.1 why CNN Model:
A Convolutional Neural Network (CNN) is a type of artificial neural network that is commonly used for image processing, classification, and recognition tasks[(Sharma and Phonsa 2021)](#sharma-a-and-phonsa-g-2021-image-classification-using-cnn-ssrn-electronic-journal). CNNs are designed to automatically and adaptively learn spatial hierarchies of features from images, which makes them particularly well-suited for tasks such as object detection, image segmentation, and image classification.[(Sharma and Phonsa 2021)](#sharma-a-and-phonsa-g-2021-image-classification-using-cnn-ssrn-electronic-journal)

In the context of chess pieces, a CNN can be used to process images of chess pieces and classify them into different categories, such as pawn, rook, knight, bishop, queen, and king. This can be useful in a variety of applications, such as automated chess analysis, chess notation conversion, and chess game analysis.

The reason why CNNs are good for processing chess piece images is because they can effectively learn and extract features that are relevant for chess piece classification [(Quintana et al. 2020)]First, the code imports the WandbLogger class from the pytorch_lightning.loggers module and the wandb module itself.

The logger function starts by calling the wandb.login function with an empty string as its argument. This function is used to log in to a WandB account using an API key, which is inserted in place of the empty string. This allows the function to access the user's WandB account and use it for logging.

Next, the function creates an instance of the WandbLogger class, passing in the argument log_model="all". This argument tells the logger to log the model architecture and weights, as well as any other information that is logged by default.

Finally, the function returns the WandbLogger instance, which can be used to log information during training.(#quintana-d-andrea-calderón-garcía-and-prieto-matías-m-2020-livechess2fen-a-framework-for-classifying-chess-pieces-based-on-cnns-arxiv-cornell-university). For example, a CNN can learn to recognize the distinct shapes and patterns of different chess pieces, such as the cross-shaped pattern of a rook or the curved shape of a knight. Additionally, CNNs can learn to be invariant to variations in image scale, orientation, and lighting conditions, which can help improve the robustness and accuracy of chess piece classification[(Yamashita et al. 2018)](#yamashita-r-nishio-m-do-r-k-g-and-togashi-k-2018-convolutional-neural-networks-an-overview-and-application-in-radiology-insights-into-imaging-online-9-4-611–629-available-from-httpsinsightsimagingspringeropencomarticles101007s13244-018-0639-9).

Overall, CNNs provide a powerful and flexible framework for processing and classifying chess piece images, and can help enable a wide range of chess-related applications and analyses.

### 1.2.2 why Pytorch_lightning
In the context of chess piece classification, a CNN can be used to process images of chess pieces and classify them into different categories, such as pawn, rook, knight, bishop, queen, and king. PyTorch Lightning can simplify the training process by automating tasks such as logging of training metrics, saving and loading checkpoints, and progress tracking. This enables you to focus on building the CNN model and improving its accuracy[(Falcon 2020)](#falcon-w-2020-from-pytorch-to-pytorch-lightning--a-gentle-introduction-online-medium-available-from-httpstowardsdatasciencecomfrom-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09).

 
* multiple features: It offers lightweight featFirst, the code imports the WandbLogger class from the pytorch_lightning.loggers module and the wandb module itself.

The logger function starts by calling the wandb.login function with an empty string as its argument. This function is used to log in to a WandB account using an API key, which is inserted in place of the empty string. This allows the function to access the user's WandB account and use it for logging.

Next, the function creates an instance of the WandbLogger class, passing in the argument log_model="all". This argument tells the logger to log the model architecture and weights, as well as any other information that is logged by default.

Finally, the function returns the WandbLogger instance, which can be used to log information during training.ures such as the option to easily manage training, testing , loading previous checkpoints based on "latest train", "besd accuracy" and etc. on a distributed system, using automatic mixed precision for better speed, and accuracy.[(Maurya et al. 2023)](#maurya-a-mocholí-c-pablo-j-lienen-m-and-shenoy-n-2023-pytorch-lightning-20-online-lightning-ai-available-from-httpslightningaireleases200-accessed-24-jan-2024)

* Automatic Integration of Training Logging: PyTorch Lightning simplifies the training process by automating tasks such as logging of training metrics, saving and loading checkpoints, and progress tracking. This enables you to focus on building the deep learning model[(Maurya et al. 2023)](#maurya-a-mocholí-c-pablo-j-lienen-m-and-shenoy-n-2023-pytorch-lightning-20-online-lightning-ai-available-from-httpslightningaireleases200-accessed-24-jan-2024).

* advance Model logging loading: PyTorch Lightning offers an integrated trainer that allows you to easily save your models by weights and hyperparameters metrics this enalbles the user to load the model fully or only load the weights and continue with the training or testing.
alsother are option to load the latest log or the best log from accuracy aspect. This simplifies the process of debugging your deep learning model[(Maurya et al. 2023)](#maurya-a-mocholí-c-pablo-j-lienen-m-and-shenoy-n-2023-pytorch-lightning-20-online-lightning-ai-available-from-httpslightningaireleases200-accessed-24-jan-2024).

* Seamless Extension of Models: PyTorch Lightning makes it easy to extend the capabilities of your model by offering pre-built models or easier  and components. This enables you to easily plug-and-play various model architectures, layers, and callbacks into your model[(Maurya et al. 2023)]First, the code imports the WandbLogger class from the pytorch_lightning.loggers module and the wandb module itself.

The logger function starts by calling the wandb.login function with an empty string as its argument. This function is used to log in to a WandB account using an API key, which is inserted in place of the empty string. This allows the function to access the user's WandB account and use it for logging.

Next, the function creates an instance of the WandbLogger class, passing in the argument log_model="all". This argument tells the logger to log the model architecture and weights, as well as any other information that is logged by default.

Finally, the function returns the WandbLogger instance, which can be used to log information during training.(#maurya-a-mocholí-c-pablo-j-lienen-m-and-shenoy-n-2023-pytorch-lightning-20-online-lightning-ai-available-from-httpslightningaireleases200-accessed-24-jan-2024)

* Integration with Existing Code: PyTorch Lightning can be easily integrated with existing PyTorch code. This ensures that you can utilize the features and functionalities provided by PyTorch Lightning while still maintaining compatibility with your existing codebase[(Maurya et al. 2023)](#maurya-a-mocholí-c-pablo-j-lienen-m-and-shenoy-n-2023-pytorch-lightning-20-online-lightning-ai-available-from-httpslightningaireleases200-accessed-24-jan-2024)

In conclusion, using PyTorch Lightning for chess piece classification provides several advantages such as easy integration witFirst, the code imports the WandbLogger class from the pytorch_lightning.loggers module and the wandb module itself.

The logger function starts by calling the wandb.login function with an empty string as its argument. This function is used to log in to a WandB account using an API key, which is inserted in place of the empty string. This allows the function to access the user's WandB account and use it for logging.

Next, the function creates an instance of the WandbLogger class, passing in the argument log_model="all". This argument tells the logger to log the model architecture and weights, as well as any other information that is logged by default.

Finally, the function returns the WandbLogger instance, which can be used to log information during training.h existing PyTorch code, automatic training, validating, testing and evaluating, plus saving and loading logs, and simplified model debugging. This ultimately enables you to focus on building and improving your deep learning model without having to worry about the intricacies of the underlying framework.
   
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
### 1.4. [utilities](./vars/utilities.py) :
this file comtains methods that can be used to pre-process a dataset of images for machine learning purposes. Here's a brief description of what each function does:

* create_CSV function creates a CSV file that contains the image paths and their corresponding labels.
* pre_process function resizes all the images in the dataset to a fixed size of 224x224 pixels (works like transformers but perminant) and it takes The dir_list parameter that is a list of directories, where each directory contains images belonging to a particular class.
* plot_bar function generates a bar chart that shows the number of images in each class.

* plot_img function generates a grid of images that shows samples from each class. The dir_list parameter is a list of directories, where each directory contains images belonging to a particular class. The labels parameter is a list of strings that contain the names of the classes.

   Note: The image_no parameter is an optional parameter that specifies the number of images to be displayed for each class.

Overall, these functions can be used to pre-process the chess image dataset and prepare it for loading and training a machine learning model.

another important part of this project is unit testing files that are including in [test](./test/) folder. other files include [data](./Data/) folder that have been disscussed in the following sections.

### 1.5. [wandb_logger](./vars/wandb_logger.py)

This file defines a function called logger() that sets up and returns a logging object using the WandBLogger class from the PyTorch Lightning library. WandBLogger is a wrapper around the Weights & Biases (WandB) platform, which is a tool for tracking and visualizing machine learning experiments. to be able to the visualise these graphs please visit https://wandb.ai/ and signup then visit https://wandb.ai/authorize to copy your special key and paste it into [wandb_logger](./vars/wandb_logger.py) file. to see the demonstration of the code on graphs visit   

Here's a breakdown of what the code does:

The first line imports the WandbLogger class from the pytorch_lightning.loggers module.
The second line imports the wandb module, which provides the functionality for interacting with the Weights & Biases platform.
The logger() function is defined. It starts by calling the wandb.login() function with an empty string as its argument. This function is used to log in to the Weights & Biases platform using an API key. In this case, the key is not provided as an argument, so the function will prompt the user to enter their key manually.
alternatively you can provide your key in the argument itself as shown in the code, in that case, you don't need to manually enter the key.

The WandbLogger class is then initialized with the log_model="all" argument. This argument tells the logger to log the model architecture, weights, and gradients.
The logger object is returned by the function.
This logger object can then be used in a PyTorch Lightning training loop to log metrics, model artifacts, and other information to Weights & Biases, allowing you to track the progress of your training and compare different experiments.

## 2. Testing
A built-in module called unittest is available in Python for unit testing. It provides tools for writing and executing unit tests, as well as generating reports on the results.
to implement such here is an simple example:
```python
import unittest

def muli(a, b):
    return a*b

class TestMulti(unittest.TestCase):
    def test_multi(self):
        self.assertEqual(multi(1, 2), 2)
        self.assertEqual(multi(-1, 1), -1)
        self.assertEqual(multi(0, 0), 0)

if __name__ == '__main__':
    unittest.main()
```
In this example, we define a function multi that takes two arguments and returns their multiplication. We then create a class TestMulti that inherits from unittest.TestCase. Inside this class, we define a method test_multi that tests the multi function by comparing its output to the expected result. Finally, we call unittest.main() to run the tests. also these unit testing are provided for this project as follows:
* ### [test_loading_data](./test/test_loading_data.py)

* ### [test_model_run](./test/test_model_run.py)

* ### [test_utilities](./test/test_utilities.py)

* *Note: testing files may not for from terminal please run them from inside VS:code or PyCharm `issue of calling modules` from other folders * 
## 3. Dataset
The dataset consists of chess pieces images collected from various sources. It contains 6 classes: King, Queen, Rook, Bishop, Knight, and Pawn. The dataset is split into training and testing sets with a ratio of 60:20:20. this dataset is a combination of  [Chessman Dataset](https://www.kaggle.com/datasets/niteshfre/chessman-image-dataset/download?datasetVersionNumber=1) & [Chess Pieces Detection Images Dataset](https://www.kaggle.com/datasets/anshulmehtakaggl/chess-pieces-detection-images-dataset/download?datasetVersionNumber=31). these to datasets are merged and preproccessed, including resizing and renaming and creating a CSV file including labels and addresses, into [ChessClassDataSet](https://www.kaggle.com/datasets/mehdialmousavie/chessclassdataset/data) to download each of these data sets just click on the name of the datasets. if you are not usinf [ChessClassDataSet](https://www.kaggle.com/datasets/mehdialmousavie/chessclassdataset/data) , it is encouraged to run the preprocessing functions from [utilities.py](./vars/utilities.py) file.

## 4.Result
the results of training such model are mesured by  accuracy and  loss. finally will disscuss the evaluation of the system.
after training the model on the traning set for 50 epoches and using Early stopper the results have been as follows:

The output shows the results of training a convolutional neural network on a chess dataset. The dataset is split into training, validation, and test sets with 667, 222, and 222 samples, respectively (60:20:20).

During training, the model was evaluated every 50 batches, and the validation accuracy and loss were printed. After training for 11 epochs, the model was restored from the checkpoint with the best validation accuracy. The best validation accuracy achieved was 0.8435, and the corresponding validation loss was 0.5347.
![](./Data/PIC/train%20acc.png)
![](./Data/PIC/train%20loss.png)

The model was then tested on the validation set, and the accuracy and loss were printed. The test accuracy was 0.52, and the test loss was 1.54.
![](./Data/PIC/val%20acc.png)
![](./Data/PIC/val%20loss.png)

The model was then tested on the validation set, and the accuracy and loss were printed. The test accuracy was 0.57, and the test loss was 1.49.
![](./Data/PIC/test%20acc.png)
![](./Data/PIC/test%20loss.png)

Finally, a confusion matrix was printed, showing the number of true positives, false positives, true negatives, and false negatives for each class. The precision, recall, and F1-score for each class were also calculated. The overall accuracy was 0.5631, and the macro-average F1-score was 0.5609.

The output also shows several warnings and suggestions for improving the training process, such as increasing the number of workers for the dataloader and turning off shuffling for the validation and test dataloaders.

## Setup
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
## Conclusion 
This project demonstrated the effectiveness of using Convolutional Neural Networks (CNNs) and PyTorch Lightning for image classification tasks, specifically for classifying chess pieces. The model is trained on a custom dataset of chess pieces images, and the project includes files for data loading, model definition, training, and evaluation. The goal of this project was to accurately classify chess pieces into six categories: King, Queen, Rook, Bishop, Knight, and Pawn. The results of training the model on the dataset show promising accuracy of nearly 85 percent on the traning set and loss values, but on the other hand on the validation and testing dataset, the model is able to classify chess pieces with a reasonable level of accuracy nearly 60 percent. Overall, this project presents a simple yet effective approach for classifying chess pieces using CNNs and PyTorch Lightning.

# Refrences:
* ### Falcon, W., 2020. From PyTorch to PyTorch Lightning : a Gentle Introduction [online]. Medium. Available from: https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09.
* ### Maurya, A., Mocholí, C., Pablo, J., Lienen, M. and Shenoy, N., 2023. PyTorch Lightning 2.0 [online]. Lightning AI. Available from: https://lightning.ai/releases/2.0.0/ [Accessed 24 Jan 2024].
* ### Quintana, D., Andrea Calderón García and Prieto-Matías, M., 2020. LiveChess2FEN: a Framework for Classifying Chess Pieces based on CNNs. ArXiv (Cornell University).
* ### Sharma, A. and Phonsa, G., 2021. Image Classification Using CNN. SSRN Electronic Journal.
* ### Yamashita, R., Nishio, M., Do, R. K. G. and Togashi, K., 2018. Convolutional Neural networks: an Overview and Application in Radiology. Insights into Imaging [online], 9 (4), 611–629. Available from: https://insightsimaging.springeropen.com/articles/10.1007/s13244-018-0639-9.