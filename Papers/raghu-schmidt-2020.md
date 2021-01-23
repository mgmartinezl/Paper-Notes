## A Survey of Deep Learning for Scientific Discovery

[Link to the paper](https://arxiv.org/abs/2003.11755)

Authors: *Maithra Raghu, Eric Schmidt*

Affiliation: *Cornell Univeristy and Schmidt Futures report*

Year: 2020

This paper presents an overview on fundamental deep learning (DL) concepts, the development and interpretability of DL models trained on less data than tradititonal applications, and tutorials and codes for reference.

### The problems tackled by DL applications
* *Prediction problems* to map inputs to predicted outputs.
* *Predictions' interpretability*, when having accurate predictions is not enough, and an understanding on how those predictions were generated is required.
* *Transform complex input data*, e.g. massive visual data that needs to be processed efficiently.

### A typical deep learning workflow
Note the presence of iterative steps in the following pipeline. DL cannot be thought as a natural sequence with defined beginning and end.

![DL Typical Workflow](dl_workflow.PNG)

### Most relevant DL libraries and resources

**Main software options**

* Pytorch --> Lightning (high level API)
* Tensorflow --> Keras (high level API)
  
**Where to find pre-trained models**

* [PyTorch](https://pytorch:org/docs/stable/torchvision/models:html)
* [TensorFlow](https://github:com/tensorflow/models)
* [Hugging Face](https://github:com/huggingface), famous for its Transformer models.
* https://github:com/rasbt/deeplearning-models
* https://github:com/hysts/pytorch_image_classification
* https://github:com/openai/baselines
* https://modelzoo:co/
* https://github:com/rusty1s/pytorch_geometric

**Visualization, Analysis and Compute Resources**

* [Tensorboard:](https://www:tensorflow:org/tensorboard) to visualize metrics such as loss and accuracy while the model is training.
* [Google Colab:](https://colab:research:google:com/notebooks/welcome:ipynb) interactive model development, analysis, and also provides some free computation resources.

### Standard Neural Network Models and Tasks

The authors here highlight the role of *supervised learning* as *"the most basic yet most critical method for training deep neural networks"*.

**1. Multilayer Perceptrons**  
The authors state those are the most basic DL models, as they are more expressive than logistic/linear regression models. I would say here, that MLP are more expressive than linear models in general, as they allow to tackle classification problems for which decision boundaries are nonlinear. The authors also highlight that typical linear models such would be a good first step to try. However, I would say it's almost mandatory to try them, along with a simple baseline model that allows us to benchmark if more complex solutions are actually adding more value vs relatively simple ones.

**2. Convolutional Neural Networks**  
By far, the most common family of NNs, highly used in computer vision applications. The most common use cases for CNNs are summarized in the following image:

![CNNs Use Cases](CNNs_applications.PNG)

**2.1. Image Classification**  
Most common architectures for image classification include: 
* *VGG*, a simple stack of convolutional layers followed by a fully connected layer.
* *ResNets* which are a family of convolutional networks of different sizes and depths and *skip connections*. Basic idea behind ResNets is to backpropagate through the identity function to preserve the gradient.

    ![ResNet](ResNet.PNG)

* *DenseNets*, where unlike standard neural networks, every layer in a "block" is connected to every other layer. This ensures a maximum flow of information between the layers, and the concatenation of the channel feature dimension results in highly compact models.
  
    ![DenseNet](DenseNet.PNG)

* More recently, *ResNeXt* and *EfficientNets*.

**2.2. Object Detection**  
In the authors words, while *"image classification can be thought of as a global summary of the image*, object detection focuses on the lower details of tha image that allow capturing and identifying specific details (e.g. objects) in them. DL models for object detection are composed by a *backbone* (an image classification model) and a *region proposal* component (to draw the bounding boxes around detected objects). Examples of the most common pre-trained models for such purpose are:

* Faster R-CNN.
* YOLOv3.
* EfficientDets.
* Mask R-CNN.

**2.3. Semantic Segmentation and Instance Segmentation**  
These applications intent to exploit the lowest levels of detail contained in the processed images, this is, categorize every single pixel in them. Example taken from the paper: suppose we are given an image of a street, with a road, different vehicles, pedestrians, etc. We would like to determine if a pixel is part of any pedestrian, part of any vehicle or part of the road. In instance segmentation, these pixels would be further subdivided into those belonging to pedestrian one, pedestrian two or pedestrian three.

**3. Super-Resolution**  
Super resolution is a technique for transforming low resolution images to high resolution images. This problem is considered to be undetermined because there multiple possible mappings of a high-resolution output that can be assigned to a single low-resolution image (fewer equations than unknowns). Some of the available architectures to solve this problem include:

* SRCNN
* Residual Dense Networks
* Predictive Filter Flow, which has also looked at image denoising and deblurring.
  
**4. Image Registration**  
Image registration considers the problem of aligning two input images to each other. For instance, when the two input images might be from different imaging modalities (e.g. a 3D scan and a
2D image).

**5. Pose Estimation**  
Pose estimation, and most popularly human pose estimation, studies the problem of predicting the pose of a human in a given image. DL models are trained to identify the location of the main joints, the keypoints (e.g. knees, elbows, head) of the person in the image. Widely used in neuroscience, these models help in the prediction of animal behavior. 

![PE](PoseEstimation.PNG)



**6. Neural Networks for Sequence Data**  
When data has sequential structure (e.g. words in a sentence, aminoacid sequences in a protein), DL models need to be able to capture that special attribute. Sequential models are an extensive research field predominantly influenced by advances in Natural Language Processing (NLP). In particular, *machine translation* and *question answering* have been popular tasks. Within the machine translation tasks, we can find:

**6.1. Language Modelling (Next Token Prediction)**
Language modelling is a self-supervised learning method because the models used to learn from such tasks do not need to consider additional labels to the ones already set by the inputs (e.g. a sentence). In NLP, neural networks are fed with sequences of words so that the next one coming can be predicted.

**6.2. Sequence to Sequence Tasks**  
While next token prediction focuses on placing the correct entry that follows a sequence, seq-to-seq tasks transform the complete input sequence into another sequence. This is the typical application of translation machines. Sequence to sequence tasks typically rely on neural network models that have an encoder-decoder structure, with the encoder neural network
taking in the input sequence and learning to extract the important features, which is then used by the decoder neural network to produce the target output:

![Seq-to-Seq](Seq-to-Seq.PNG)

**6.3. Question Answering**










**Applications not covered by the authors of this paper: video prediction, action recognition, and style transfer**.
