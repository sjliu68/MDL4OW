# -*- coding: utf-8 -*-
"""
Last update: 2020-09-05
@author: Shengjie Liu, sjliu.me@gmail.com
"""

import time
import numpy as np
import argparse
from keras.callbacks import EarlyStopping
import rscls
from keras.utils import to_categorical
from keras.optimizers import Adadelta
import network_openset as nw
from keras import losses
import os
import utils_openset as u

from sklearn.metrics.pairwise import paired_distances as dist
import libmr

## number of training samples per class
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--nos', type=int, default = 20) # number of training samples per class
parser.add_argument('--key', type=str, default = 'salinas') # data name
parser.add_argument('--gt', type=str, default = 'data/salinas_raw_gt.npy') # only known training samples included
parser.add_argument('--closs', type=int, default = 50) # classification loss weight, 50->0.5
parser.add_argument('--patience', type=int, default = 50) # earlystopping
parser.add_argument('--output', type=str, default = 'output/') # save path for output files
parser.add_argument('--showmap', type=int, default = 1) # show classification map, change to 0 if run multiple times
args = parser.parse_args()

#%% network and basic configuration
# set EVT tail number
# numofevm for the class-wise method: MDL4OW/C
# numofevm_all for the global method: MDL4OW
numofevm_all = int(args.nos*4*0.5)
numofevm = int(args.nos*4*0.05)
if numofevm<3:
    numofevm=3
if numofevm_all<20:
    numofevm_all=20
    
patch = 9  # Stick to patch=9
vbs = 0  # if vbs==0, training in silent mode; vbs==1, print training process
bsz1 = 20  # batch size
ensemble = 1  # Stick to ensemble=1

if args.nos>=200:
    args.patience=5 

# if loss not decrease for {args.patience} epoches, stop training
early_stopping = EarlyStopping(monitor='loss', patience=args.patience, verbose=2)
loss1 = 'categorical_crossentropy'

#%%
key2 = args.gt.split('/')[-1].split('_')[1]

imfile = 'data/'+args.key+'_im.npy'
spath = args.output+args.key+'_'+key2+'_'+str(args.nos)+'_closs'+str(args.closs)+'/'

if not os.path.exists(spath):
    os.makedirs(spath)

gt = np.load(args.gt)
novellabel = gt.max()+1

#%%
seedx = [0,1,2,3,4,5,6,7,8,9,10,11,12]
seedi = 0 

#%%
def s255(im, perc=0.5):
    maxx = np.percentile(im,100-perc)
    minn = np.percentile(im,perc)
    im[im>maxx] = maxx
    im[im<minn] = minn
    im_new = np.fix((im-minn)/(maxx-minn)*255).astype(np.uint8)
    return im_new
        
#%%
unknown=0
for seedi in range(0,1):
    print('Random seed:',seedx[seedi])
    
    # load image and GT
    im = np.load(imfile)
    gt1 = np.load(args.gt)
    
    clss = np.unique(gt1)[1:]
    gt1[gt1==novellabel] = 0
    
    cls1 = gt1.max()
    im1x,im1y,im1z = im.shape
    im = np.float32(im)
    im = im/im.max()
    
    c1 = rscls.rscls(im,gt1,cls=cls1)
    c1.padding(patch)  
    
    # load train samples
    np.random.seed(seedx[seedi])
    x1_train,y1_train = c1.train_sample(args.nos)  # load train samples
    x1_train,y1_train = rscls.make_sample(x1_train,y1_train)  # augmentation
    y1_train = to_categorical(y1_train,cls1)  # to one-hot labels
    
    if patch==9:
        model1,model2 = nw.resnet99_avg_recon(im1z,patch,cls1,l=1)
    elif patch==5:
        model1,model2 = nw.wcrn_recon(im1z,cls1)
    else:
        print('ERROR: patch size unknown, no network defined !')
    
    if vbs:
        model1.summary() # print network structure

    if True: # begin training
        # first train the model with lr=1.0
        time2 = int(time.time())
        model1.compile(loss=[loss1,losses.mean_absolute_error],optimizer=Adadelta(lr=1.0),metrics=['accuracy'],loss_weights=[args.closs/100.0,1-args.closs/100.0])
        model1.fit(x1_train,[y1_train,x1_train],batch_size=bsz1,epochs=170,verbose=vbs,shuffle=True,callbacks=[early_stopping])
            
        # then train the model with lr=0.1
        model1.compile(loss=[loss1,losses.mean_absolute_error],optimizer=Adadelta(lr=0.1),metrics=['accuracy'],loss_weights=[args.closs/100.0,1-args.closs/100.0])
        model1.fit(x1_train,[y1_train,x1_train],batch_size=bsz1,epochs=30,verbose=vbs,shuffle=True,callbacks=[early_stopping])
        time3 = int(time.time()) # training time
        print('training time:',time3-time2)
        
        # predict part, predicting image row-by-row
        pre_all = []
        preloss = []
        for i in range(ensemble):
            pre_rows_1 = []
            for j in range(im1x):
                sam_row = c1.all_sample_row(j)
                pre_row1,_ = model1.predict(sam_row)
                _ = dist(_.reshape(im1y,-1),sam_row.reshape(im1y,-1))
                preloss.append(_)
                pre_rows_1.append(pre_row1)
            pre_all.append(np.array(pre_rows_1))
        
        
        #%% predict finished, post processing
        # reconstruction loss, predicted
        preloss = np.array(preloss) 
        preloss = np.float64(preloss.reshape(-1))
        np.save(spath+args.key+'_predictloss'+'_'+str(seedi),preloss)
        
        # predicted probabilities
        pre = pre_all[0] 
        np.save(spath+args.key+'_predict'+'_'+str(seedi),pre)
        
        # save model
        model1.save(spath+args.key+'_model'+'_'+str(seedi)) 
        
        # closed classification (baseline)
        pre0 = np.argmax(pre,axis=-1) + 1
        np.save(spath+args.key+'_close'+'_'+str(seedi),pre0) 
        
        # get training reconstruction loss
        _,trainloss = model1.predict(x1_train)
        trainloss = dist(trainloss.reshape(trainloss.shape[0],-1),
                         x1_train.reshape(x1_train.shape[0],-1))
        np.save(spath+args.key+'_trainloss'+'_'+str(seedi),trainloss) #2
        
        #%% MDL4OW, global fashion
        # get unknown mask
        mr = libmr.MR()
        mr.fit_high(trainloss,numofevm_all) # tmp3, loss of training samples
        wscore = mr.w_score_vector(preloss)
        mask = wscore>0.5 # default threshold=0.5, no need to change this
        wmax = preloss[wscore<0.5].max() # max w-score
        mask = mask.reshape(im1x,im1y)
        
        # apply unknown mask, global
        pre = pre_all[0]
        pre1 = np.argmax(pre,axis=-1) + 1
        pre1[mask==1] = 0
        np.save(spath+args.key+'_pre_global'+'_'+str(seedi),pre1)
        
        #%% MDL4OW/C
        mrs = {} # save libmr model
        wscores = {}
        y2_train = np.argmax(y1_train,axis=-1)+1
        np.save(spath+args.key+'_trainlabel'+'_'+str(seedi),y2_train) #4
        
        realmask = np.zeros([im1x,im1y],np.uint8)
        for cls2 in clss:
            idx = y2_train==cls2
            tmp4 = trainloss[idx]
            mrs[cls2] = libmr.MR()
            mrs[cls2].fit_high(tmp4,numofevm)
            wscore = mrs[cls2].w_score_vector(preloss)
            mask = wscore>0.5
            wmax = preloss[wscore<0.5].max() # max w-score
            wscores[cls2] = wmax
            mask = mask.reshape(im1x,im1y)
            realmask[np.logical_and(mask,pre1==cls2)]=1
            
        # apply mask, class-wise
        pre = pre_all[0]
        pre2 = np.argmax(pre,axis=-1) + 1
        pre2[realmask==1] = 0
        np.save(spath+args.key+'_pre_classwise'+'_'+str(seedi),pre2) #5
        
        time4 = int(time.time()) # training time
        print('predict time:',time4-time3)
        
        #%%
        if args.showmap:
            u.save_cmap_salinas16(pre0,0,spath+args.key+'_close_'+str(seedi))
            u.save_cmap_salinas16(pre1,0,spath+args.key+'_mdl4ow_'+str(seedi))
            u.save_cmap_salinas16(pre2,0,spath+args.key+'_mdl4ow_classwise_'+str(seedi))
            pass
        
            














