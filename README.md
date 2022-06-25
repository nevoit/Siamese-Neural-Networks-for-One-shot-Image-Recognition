# One-shot Siamese Neural Network
In this assignment we were tasked with creating a Convolutional Neural Networks (CNNs). 
A step-by-step CNNs tutorial you can find [here (DeepLearning.ai)](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF).

The assignment is to be carried out using Python V3.6 (or higher) and TensorFlow 2.0 (or higher).

In this assignment, we were tasked with creating a One-shot Siamese Neural Network, using TensorFlow 2.0, based on the work presented by Gregory Koch, Richard Zemel, and Ruslan Salakhutdinov.
As specified, we used the “Labeled Faces in the Wild” dataset with over 5,700 different people. Some people have a single image, while others have dozens. We used, as requested, the improved dataset where the faces were aligned automatically using specialized software.

# Table of Contents
1. [Authors](#Authors)
2. [Purposes of The Assignment](#Purposes-of-The-Assignment)
3. [Instructions](#Instructions)
4. [Dataset Analysis](#Dataset-Analysis)
5. [Getting Started](#Getting-Started)
6. [Code Design](#Code-Design)
7. [Architecture](#Architecture)
8. [Initialization](#Initialization)
9. [Stopping criteria:](#Stopping-criteria)
10. [Network Hyper-Parameters Tuning:](#Network-Hyper-Parameters-Tuning)
11. [Full Experimental Setup](#Full-Experimental-Setup)
12. [Experimental Results](#Experimental-Results)

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

![Figure 1 - Siamese network for facial recognition](https://github.com/nevoit/Siamese-Neural-Networks-for-One-shot-Image-Recognition/blob/master/figures/figure%201%20explanation.png?raw=true "Figure 1 - Siamese network for facial recognition")
Figure 1 - Siamese network for facial recognition

## Dataset Analysis
- Size: 5,749  people
- Number of examples –
    - Total:13,233 images. Some people have a single image and some have dozens.
    - Class 1 (identical): 1,100 pairs (in the training file) and 500 pairs (in the testing file)
    - Class 0 (non-identical): 1,100 pairs (in the training file) and 500 pairs (in the testing file)
    - Validation set: 20 percent of the training set - 440 pairs. (leaving 1760 pairs for the actual training data)

## Getting Started
Option 1: Through Google Colab (see a [copy here](https://colab.research.google.com/drive/1fwTLg6HTIAaly3b2kuZ4kMpHHmtf_gHU?usp=sharing) - please click on "Star" and "Follow" before asking for permission) or on Kaggle (see a [copy here](https://www.kaggle.com/code/nevoit/siamese-neural-networks-one-shot-image-recognition) - please upvote and follow):
 
1. Dataset -
- Run the Colab script
- Mount your drive and follow the link. Paste the code into the input box:

![colab](https://github.com/nevoit/Siamese-Neural-Networks-for-One-shot-Image-Recognition/blob/master/figures/colab.png?raw=true "colab")

- Create the subfolder  “Content/My Drive/datasets/lfw2”
- Inside this directory create two subdirectories: lfw2 which will contain all the folders with the images from the dataset and splits which will contain the files test.txt and train.txt as supplied in the assignment.
- Run all the following code blocks in order to compile the classes and run the experiments.

Option 2: Through an IDE:
1. Dataset -
- Create a folder called ‘lfwa’ in the project main folder.
- Create a sub-folder called ‘lfw2’ inside ‘lfwa’
- Create a sub-folder called ‘lfw2’ inside ‘lfwa\\lfwa2’ and put all the image folders inside this folder. For example: ‘lfwa\\lfwa2\\lfwa2\\Aaron_Eckhart’:

![ide](https://github.com/nevoit/Siamese-Neural-Networks-for-One-shot-Image-Recognition/blob/master/figures/IDE.PNG?raw=true "ide")

2. Training and Testing set - Create a sub-folder called ‘splits’ inside ‘lfwa\\lfwa2’ and put ‘train.txt’ and ‘test.txt’ files. Each file describes all the pairs:

![dataset](https://github.com/nevoit/Siamese-Neural-Networks-for-One-shot-Image-Recognition/blob/master/figures/dataset.PNG?raw=true "dataset")

3. Installation -
- The project has been tested on Windows 10 with Python 3.7.1 and TensorFlow 2.0.0.
- Install Pillow, Sklearn, and TQDM libraries (included in setup requirements).

## Code Design
Our code consists of three scripts:
1. Data_loader.py - contains the DataLoader class that loads image data, manipulates it, and writes it into a specified path in a certain format.
2. Siamese_network.py - contains the SiameseNetwork class that is our implementation of the network described in the paper. It includes many functions including one that builds the CNN used in the network and a function for finding the optimal hyperparameters.
3. Experiments.py - The script that is actually running. It calls the train and predict methods from SiameseNetwork.

## Architecture
We mostly followed the architecture specified in the paper - The network is two Convolutional Neural Networks that are joined towards the end creating a Siamese network. However, our network is slightly smaller.
Number of layers: Each CNN, before being conjoined, has 5 layers (4 conventional and 1 fully connected layer). Then there is a distance layer, combining both CNNs, with a single output.
Dimensions: For the CNN part:

| Layer | Input Size | Filters | Kernel | Maxpooling | Activation Function|
|---| --- | --- | --- |  ---  | --- |
|  1 | 105x105. Reshaped from 250x250 to adhere to the paper. | 64 | 10x10 | Yes, Stride of 2 |ReLU
| 2 | 64 filters of 10x10 | 128 | 7x7 | Yes, Stride of 2 | ReLU |
| 3 | 128 filters of 7x7 | 128 | 4x4 | Yes, Stride of 2 | ReLU |
| 4 | 128 filters of 4x4 | 256 | 4x4 | No | ReLU |
| 5 | 4096x1 Fully connected feature layer with drop out rate of 0.4 (Fraction of the input units to drop) | - | - |No | Sigmoid|
- There are two identical CNNs as described in the table.
- All CNN layers, except the last one (the fully connected layer), are defined with a fixed stride of 1 (as in the paper), padding value of ‘valid’ (with no zero paddings, the kernel is restricted to traverse only within the image), L2 as kernel regularizer with regularization factor of 2e-4 and perform batch normalization.
- For the last one (the fully connected layer), we used L2 as a kernel regularizer with a regularization factor of 2e-3.
- Please note Any and all parameters that were not mentioned used the default Tensorflow 2.0.0 Keras values.
- After the last layer, we add a layer that connects both sides thus creating the Siamese network. The activation function of this layer is a Sigmoid function which is handy since we have 2 classes (Same vs Not the same person).
- Total params: 38,954,049
- Trainable params: 38,952,897
- Non-trainable params: 1,152

## Initialization
- Weight initialization for all edges was done as described in the paper: A normal distribution with a mean of 0 and a standard deviation of 0.01.
- Bias initialization was also done as it was in the paper, with a mean of 0.5 and a standard deviation of 0.01. However, the first layer has no bias. The paper doesn’t mention if they did this or not, but we found in this paper that occasionally, having no bias for the initial layer might be beneficial. This occurs when the layer is sufficiently large and the data is distributed fairly uniformly, which probably occurs in our case because the training set is predefined. Indeed, in our experiments adding a bias usually reduced the accuracy. Our final model doesn’t have a bias for the first layer.
- Note: the authors used a slightly different bias initialization for the fully connected layers.  Since there are so many edges, they sampled from a larger distribution. In our experiments, the same bias sampling as the rest of the network worked well so we used the same distribution.
- These are fairly typical methods of initializing weights and seemed to work well for the authors so we saw no reason to not imitate them (excluding the fully connected layer).

## Stopping Criteria
We used the EarlyStopping function monitoring on the validation loss with a minimum delta of 0.1 (Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.) and patience 5 (Number of epochs with no improvement after which training will be stopped.). The direction is automatically inferred from the name of the monitored quantity (‘auto’).

## Network Hyper-Parameters Tuning
NOTE: Here we explain the reasons behind the choices of the parameters.
After implementing our Siamese Network, we had to optimize the different parameters used. Many of them, such as layer size, were chosen based on the work in the paper and we decided against trying to find a better combination since the search space would be massive and we don’t know enough to narrow it down.
- Learning Rate: We tried many  different values, ranging from 0.1 to 0.00001. After running numerous experiments, we found 0.00005 to work the best.
- Optimizer: The paper didn’t specify so we used the robust and popular ADAM optimizer.
- Epochs: We tried epochs of 5, 10, 20 and 50. We found 10 to work the best.
- Batch size: We tried multiplications of 16 such as 16, 32 and 64. Our final model has a batch size of 32.

## Full Experimental Setup
Validation Set: Empirically, we learned that using a validation set is better than not if there isn’t enough data. We used the fairly standard 80/20 ratio between training and validation which worked well.
- Batch sizes - 32
- Epochs - 10
- Stopping criteria - 5 epochs without improvement.
- Learning rate: 0.0005
- Min delta for improvement: 0.1

##  Experimental Results
After implementing our Siamese Network, we ran it with many different settings as described above and chose the optimal settings. These are the results:
a.	Convergence times, final loss and accuracy on the test set and holdout set:
- Final Loss on Testing set - 3.106
- Accuracy on Holdout set - 0.734 (73.4%)
- Final Loss on Testing set - 3.111
- Accuracy on Testing set - 0.731 (73.2%)
- Convergence time - 30 seconds
- Prediction time - less than 1 second

b. Graphs describing the loss on the training set throughout the training process:

![loss-epoch](https://github.com/nevoit/Siamese-Neural-Networks-for-One-shot-Image-Recognition/blob/master/figures/loss-epoch.PNG?raw=true "loss-epoch")

Fig.2: Reduction in the loss function for each epoch. The validation set predicted the loss on the test set well.

![loss-acc](https://github.com/nevoit/Siamese-Neural-Networks-for-One-shot-Image-Recognition/blob/master/figures/acc-epoch.PNG?raw=true "loss-acc")

Fig.3: Accuracy of the training and validation sets for each epoch. For the validation set the accuracy plateaued after 2 epochs, but as you can see in fig.1 the loss continued to reduce explaining the increase in  accuracy for the test set for 2 more epochs.

c. Performance when experimenting with various parameters:
We used the best parameters and changed some of them to test their effect, seed 0, learning rate 0.00005, batch_size 32, epochs 10 patience of 5, and minimum delta of 0.1.

We tested the following learning rates: 0.000005, 0.00001, 0.00003 0.00005, 0.00007, 0.0001 and 0.001.

![lr-acc](https://github.com/nevoit/Siamese-Neural-Networks-for-One-shot-Image-Recognition/blob/master/figures/acc_lr.PNG?raw=true "lr-acc")

Fig.4: Here we can see that the learning rate around 0.0005 had similar results, but if it was too large or too small the results dropped drastically.

We tested the following epochs: 1, 2, 3, 5, 10, 15, 20, 30 epochs.

![ep-acc](https://github.com/nevoit/Siamese-Neural-Networks-for-One-shot-Image-Recognition/blob/master/figures/acc_epoch.PNG?raw=true "ep-acc")

Fig.5:# of epochs didn’t change the accuracy much past 2 epochs.

We tested the following batch sizes: 8, 16, 32, 48, 64.

![bs-acc](https://github.com/nevoit/Siamese-Neural-Networks-for-One-shot-Image-Recognition/blob/master/figures/acc_bs.PNG?raw=true "bs-acc")

Fig.6: Curiously, batch size distributes normally around 32 for the test set and is wildly different for the validation set.

d. Please include examples of accurate classifications and misclassifications and try to determine why your model was not successful.

Best correct classification:
Same person (prob: 0.9855): Unsurprising, as the images really are very similar.

![same](https://github.com/nevoit/Siamese-Neural-Networks-for-One-shot-Image-Recognition/blob/master/figures/gordon_campbell.PNG?raw=true "same")

Different people (prob: 0.0000379): It is quite clear that it’s two different people. Nothing too interesting here - The colors and facial expressions are very different.

![babe](https://github.com/nevoit/Siamese-Neural-Networks-for-One-shot-Image-Recognition/blob/master/figures/babe_ruth_joshua_perper.PNG?raw=true "babe")

Worst Misclassification:
Same person (prob:0.0587): Even though both are the same person, the images are radically different - In the left image, Candice is wearing sunglasses, has bright hair and is looking the other way. On the right, she has dark hair, no sunglasses and has her teeth showing. We theorize that most people would classify this wrong as well.

![Candice_Bergen](https://github.com/nevoit/Siamese-Neural-Networks-for-One-shot-Image-Recognition/blob/master/figures/candice_bergen.PNG?raw=true "Candice_Bergen")

Different people (prob: 0.9464): This is quite surprising since it’s quite apparent that this is not the same person, but the network had such high confidence that they are. Perhaps the coat resembles the hair lapping around her head.

![lisa](https://github.com/nevoit/Siamese-Neural-Networks-for-One-shot-Image-Recognition/blob/master/figures/lisa_murkowski_svetlana_belousova.PNG?raw=true "lisa")

e. Any other information you consider relevant or found useful while training the model
- We used K.clear_session() in order to make sure we are in a new session in each combination in the experiment (We imported consider K as tensorflow.keras.backend).
- We initialized the seeds using these lines:

`os.environ['PYTHONHASHSEED'] = str(self.seed)`

`random.seed(self.seed)`

`np.random.seed(self.seed)`

`tf.random.set_seed(self.seed)`
