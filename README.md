# MDL4OW

Shengjie Liu, Qian Shi, and Liangpei Zhang. Few-shot Hyperspectral Image Classification With Unknown Classes Using Multitask Deep Learning. IEEE TGRS, 2020. [doi:10.1109/TGRS.2020.3018879](https://doi.org/10.1109/TGRS.2020.3018879)

Contact: sjliu.me AT gmail.com

This paper is published in IEEE TGRS: 

Code and annotations are released here, or check [https://github.com/sjliu68/MDL4OW](https://github.com/sjliu68/MDL4OW)

### Overview


### Key packages
    tensorflow-gpu==1.9
    keras==2.1.6
    libmr
    
Tested on Python 3.6, Windows 10

Recommend Anaconda, Spyder
    
### How to use
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

