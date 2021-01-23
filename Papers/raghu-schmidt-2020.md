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


