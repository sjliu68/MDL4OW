# MDL4OW

Source code and annotations for :

Shengjie Liu, Qian Shi, and Liangpei Zhang. Few-shot Hyperspectral Image Classification With Unknown Classes Using Multitask Deep Learning. IEEE TGRS, 2020. [doi:10.1109/TGRS.2020.3018879](https://doi.org/10.1109/TGRS.2020.3018879)

Contact: sjliu.me AT gmail.com

Code and annotations are released here, or check [https://github.com/sjliu68/MDL4OW](https://github.com/sjliu68/MDL4OW)

## Overview
Ordinary            |  What we do
:-------------------------:|:-------------------------:
![](https://sjliu.me/images/mdl4ow1.png)  |  ![](https://sjliu.me/images/mdl4ow2.png)

#### Normal: closed classification
Below is a normal/closed classification. If you are familiar with hyperspectral data, you will notice some of the materials are not represented in the training samples. For example, for the upper image (Salinas Valley), the road and the houses between farmlands cannot be classified into any of the known classes. But still, a deep learning model has to assign one of the labels, because it is never taught to identify an unknown instance.

#### What we do: open classification

What we do here is, by using multitask deep learning, enpowering the deep learning model with the ability to identify the unknown: those masked with black color. 

For the upper image (Salinas Valley), the roads and houses between farmlands are successfully identified.

For the lower image (University of Pavia Campus), helicopters and trucks are successfully identified. 


## Key packages
    tensorflow-gpu==1.9
    keras==2.1.6
    libmr
    
Tested on Python 3.6, Windows 10

Recommend Anaconda, Spyder
    
## How to use
#### Quick usage
    python demo_salinas.py

#### Arguments
    --nos: number of training samples per class. 20 for few-shot, 200 for many-shot
    --key: dataname: 'salinas', 'paviaU', 'indian'
    --gt: gtfile path
    --closs: classification loss weight, default=50 (0.5)
    --patience: it loss not decrease for {patience} epoches, stop training
    --output: save path for output files (model, predict probabilities, predict labels, reconstruction loss)
    --showmap: save classification map

