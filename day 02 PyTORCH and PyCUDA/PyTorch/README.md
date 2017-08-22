<p align="center"><img width="40%" src="../images/pt.jpg" /></p>

--------------------------------------------------------------------------------

### The slides were created using:

`
%%bash
jupyter nbconvert \
    --to=slides \
    --reveal-prefix=https://cdnjs.cloudflare.com/ajax/libs/reveal.js/3.2.0/ \
    --output=py05.html \
    './05 PyTorch Automatic differentiation.ipynb'
`

# Deep Learning Bootcamp November 2017, GPU Computing for Data Scientists: PyTorch

Web: https://www.meetup.com/Tel-Aviv-Deep-Learning-Bootcamp/events/241762893/

https://www.meetup.com/Tel-Aviv-Deep-Learning-Bootcamp/events/242418339/


Notebooks: <a href="https://github.com/QuantScientist/Data-Science-PyCUDA-GPU"> On GitHub</a>

PyTorch is an optimized tensor library for Deep Learning, and is a recent newcomer to the growing list of GPU programming frameworks available in Python. Like other frameworks it offers efficient tensor representations and is agnostic to the underlying hardware. However, unlike other frameworks it allows you to create “define-by-run” neural networks resulting in dynamic computation graphs, where every single iteration can be different—opening up a whole new world of possibilities. Central to all neural networks in PyTorch is the Autograd package, which performs Algorithmic Differentiation on the defined model and generates the required gradients at each iteration.

***Keywords: GPU Processing, Algorithmic Differentiation, Deep Learning, Linear algebra.***


## Table of Contents

#### Jupyter Notebooks
- 01 
- 02 
- 03 
- 04 
- 05 
- 06
- 07
- 08

<br/>

## How to get started

### GPU
```bash
$ git clone https://github.com/QuantScientist/Data-Science-PyCUDA-GPU/tree/master/docker
```

### CPU
```
!pip install http://download.pytorch.org/whl/cu75/torch-0.2.0.post1-cp27-cp27mu-manylinux1_x86_64.whl
!pip install torchvision 

```
<br/>


# Workshop Agenda:

#### Module 1 Getting Started  

- What is Pytorch

- Install and Run Pytorch

- Allocating CPU Tensors using PyTorch

- Allocating GPU Tensors using PyTorch 



#### Module 2 Basic Pytorch Operations

- Tensor Operation

- Numpy Bridge

- Variable

- Gradients and Autograd



#### Module 3 Data Pre-processing

- Install and Run Torchvision

- Datasets

- Data Transformation


#### Module 4 Linear/Logistic Regression with Pytorch

- Loss Function 

- Optimizer

- Training

#### Module 5 Neural Network (NN) with Pytorch

- What is Neural Network 

- Activation Functions

- Deep Neural Network with Pytorch


#### Module 7 Convolutional Neural Network (CNN) with Pytorch

- What is CNN?

- CNN Architecture

- Convolution 

- Pooling and Stride

- Dropout


## Author
Shlomo Kashani/ [@QuantScientist](https://github.com/QuantScientist)


# A very comprehensice list of PyTorch links:

* [Deep Learning with PyTorch: A 60-minute Blitz](https://github.com/pytorch/tutorials/blob/master/Deep%20Learning%20with%20PyTorch.ipynb)
* [Deep Learning For NLP in PyTorch](https://github.com/rguthrie3/DeepLearningForNLPInPytorch)
* [Practical PyTorch](https://github.com/spro/practical-pytorch)

https://github.com/ritchieng/the-incredible-pytorch/blob/master/README.md

https://github.com/Cadene/pretrained-models.pytorch

https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208

http://europython2017.pogrady.com/#20


https://blog.paperspace.com/adversarial-autoencoders-with-pytorch/

https://deepsense.io/deep-learning-for-satellite-imagery-via-image-segmentation/

https://github.com/jcjohnson/pytorch-examples
