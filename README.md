# Convolutional-Neural-Networks
In this assignment we were tasked with creating a Convolutional Neural Networks (CNNs). 
A step-by-step CNNs tutorial you can find [here (DeepLearning.ai)](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF).

The assignment is to be carried out using Python V3.6 (or higher) and TensorFlow 2.0 (or higher).

In this assignment, we were tasked with creating a One-shot Siamese Neural Network, using TensorFlow 2.0, based on the work presented by Gregory Koch, Richard Zemel, and Ruslan Salakhutdinov.
As specified, we used the “Labeled Faces in the Wild” dataset with over 5,700 different people. Some people have a single image, while others have dozens. We used, as requested, the improved dataset where the faces were aligned automatically using specialized software.

## Authors
* **Tomer Shahar** - [Tomer Shahar](https://github.com/Tomer-Shahar)
* **Nevo Itzhak** - [Nevo Itzhak](https://github.com/nevoit)

## Purposes of The Assignment 
Enabling students to experiment with building a convolutional neural net and using it on a real-world dataset  and problem.
In addition to practical knowledge in the “how to” of building the network, an additional goal is the integration of useful logging tools for gaining better insight into the training process. Finally, the students are expected to read, understand and (loosely) implement a scientific paper.

In this assignment, you will use convolutional neural networks (CNNs) to carry out the task of facial recognition. As shown in class, CNNs are the current state-of-the-art approach for analyzing image-based datasets. More specifically, you will implement a one-shot classification solution. Wikipedia defines one-shot learning as follows: 
“… an object categorization problem, found mostly in computer vision. Whereas most machine learning based object categorization algorithms require training on hundreds or thousands of samples/images and very large datasets, one-shot learning aims to learn information about object categories from one, or only a few, training samples/images.”

Your work will be based on the paper [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf).
Your goal, like that of the paper, is to successfully execute a one-shot learning task for previously unseen objects. Given two facial images of previously unseen persons, your architecture will have to successfully determine whether they are the same person. While we encourage you to use the architecture described in this paper as a starting point, you are more than welcome to explore other possibilities.

## Instructions
-	Use the following dataset - [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/index.html)

(a) Download the dataset. Note: there are several versions of this dataset, use the version [found here](https://talhassner.github.io/home/projects/lfwa/index.html) (it’s called LFW-a, and is also used in the DeepFace paper).

(b)	Use the following train and test sets to train your model: [Train](http://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt) \ [Test](http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt). [Remember - you will use your test set to perform one-shot learning. This division is set up so that no subject from test set is included in the train set]. Please note it is often a recommended to use a validation set when training your model. Make your own decision whether to use one and what percentage of (training) samples to allocate.

(c) In your report, include an analysis of the dataset (size, number of examples – in total and per class – for the train and test sets, etc). Also provide the full experimental setup you used – batch sizes, the various parameters of your architecture, stopping criteria and any other relevant information. A good rule of thumb: if asked to recreate your work, a person should be able to do so based on the information you provide in your report.

- Implement a Siamese network architecture while using the above-mentioned paper as a reference.

(a) Provide a complete description of your architecture: number of layers, dimensions, filters etc. Make sure to mention parameters such as learning rates, optimization and regularization methods, and the use (if exists) of batchnorm.

(b) Explain the reasoning behind the choices made in answer to the previous section. If your choices were the result of trial and error, please state the fact and describe the changes made throughout your experiments. Choosing certain parameter combination because they appeared in a previously published paper is a perfectly valid reason. 

- In addition to the details requested above, your report needs to include an analysis of your architecture’s performance. Please include the following information:

(a) Convergence times, final loss and accuracy on the test set and holdout set

(b) Graphs describing the loss on the training set throughout the training process

(c) Performance when experimenting with the various parameters

(d) Please include examples of accurate and misclassifications and try to determine why your model was not successful.

(e) Any other information you consider relevant or found useful while training the model

Please note the that report needs to reflect your decision-making process throughout the assignment. Please include all relevant information.

- Please note that your work will not be evaluated solely on performance, but also on additional elements such as code correctness and documentation, a complete and clear documentation of your experimental process, analysis of your results and breadth and originality (where applicable).

![Figure 1 - Siamese network for facial recognition](https://github.com/nevoit/Convolutional-Neural-Networks/blob/master/figures/figure%201%20explanation.png "Figure 1 - Siamese network for facial recognition")

## Dataset Analysis
- Size: 5,749  people
- Number of examples –
    - Total:13,233 images. Some people have a single image and some have dozens.
    - Class 1 (identical): 1,100 pairs (in the training file) and 500 pairs (in the testing file)
    - Class 0 (non-identical): 1,100 pairs (in the training file) and 500 pairs (in the testing file)
    - Validation set: 20 percent of the training set - 440 pairs. (leaving 1760 pairs for the actual training data)

## Getting Started
Option 1: Through Google Colab (see a [copy here](https://colab.research.google.com/drive/1fwTLg6HTIAaly3b2kuZ4kMpHHmtf_gHU?usp=sharing) - BGU users only):

1. Dataset -
- Run the Colab script
- 
- Mount your drive and follow the link. Paste the code into the input box:
- Create the subfolder  “Content/My Drive/datasets/lfw2”
- Inside this directory create two subdirectories: lfw2 which will contain all the folders with the images from the dataset and splits which will contain the files test.txt and train.txt as supplied in the assignment.
- Run all the following code blocks in order to compile the classes and run the experiments.

Option 2: Through an IDE:
1. Dataset -
- Create a folder called ‘lfwa’ in the project main folder.
- Create a sub-folder called ‘lfw2’ inside ‘lfwa’
- Create a sub-folder called ‘lfw2’ inside ‘lfwa\\lfwa2’ and put all the image folders inside this folder. For example: ‘lfwa\\lfwa2\\lfwa2\\Aaron_Eckhart’:
2. Training and Testing set - Create a sub-folder called ‘splits’ inside ‘lfwa\\lfwa2’ and put ‘train.txt’ and ‘test.txt’ files. Each file describes all the pairs:
3. Installation -
- The project has been tested on Windows 10 with Python 3.7.1 and TensorFlow 2.0.0.
- Install Pillow, Sklearn, and TQDM libraries (included in setup requirements).

## Code Design:
Our code consists of three scripts:
1. Data_loader.py - contains the DataLoader class that loads image data, manipulates it, and writes it into a specified path in a certain format.
2. Siamese_network.py - contains the SiameseNetwork class that is our implementation of the network described in the paper. It includes many functions including one that builds the CNN used in the network and a function for finding the optimal hyperparameters.
3. Experiments.py - The script that is actually running. It calls the train and predict methods from SiameseNetwork.



