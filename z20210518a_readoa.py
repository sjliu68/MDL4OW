# -*- coding: utf-8 -*-
"""
Created on Wed May 20 04:06:55 2020

@author: sj
"""

import numpy as np
import rscls
import glob
import copy
import libmr
import matplotlib.pyplot as plt

#%%
def read(fp=None,mode=0,key='salinas',seed=-1,opendic=1,cls1=-1,num=20):
    if mode==0: # closed classification
        pre = np.load(fp+key+'_close_'+str(seed)+'.npy')
    elif mode==1: # MDL4OW
        pre = np.load(fp+key+'_pre_o1_'+str(seed)+'.npy')
    elif mode==2: # MDL4OW/C
        pre = np.load(fp+key+'_pre_o2_'+str(seed)+'.npy')
    elif mode==3: # closed classification, same as mode==1, except input is probablity: predict image, imx*imy*c
        pre = np.load(r'G:\open-set-standard\keras\saved\hresnet_200\paviaU_'+str(seed)+'.npy') 
        pre = np.argmax(pre,axis=-1)+1
    elif mode==4:  # softmax-threshold
        pre = np.load(fp+key+'_pre_'+str(seed)+'.npy')
        pre1 = np.argmax(pre,axis=-1)+1
        mask = pre.max(axis=-1)
        pre1[mask<opendic] = cls1
        pre = pre1
    elif mode==5:  # openmax
        pre = np.load(fp+key+'_close_'+str(seed)+'.npy')
        im1x,im1y = pre.shape
        tmp3 = np.load(fp+key+'_trainloss_'+str(seedi)+'.npy') #2
        evm = np.load(fp+key+'_evm_'+str(seedi)+'.npy')
        numofevm_all = int(num*4*0.5)
        numofevm = int(num*4*0.05)
        if numofevm<3:
            numofevm=3
        if numofevm_all<20:
            numofevm_all=20
        # all in 
        mr = libmr.MR()
        mr.fit_high(tmp3,numofevm_all) # tmp3, loss of training samples
        wscore = mr.w_score_vector(evm)
        mask = wscore>1-opendic
        mask = mask.reshape(im1x,im1y)
        pre[mask] = cls1
    return pre

def calarea(pre,gt,inclass): # calculate mapping error
    area = []
    da = []
    gts = []
    for i in inclass:
        a = np.sum(pre==i)
        b = np.sum(gt==i)
        area.append(a)
        gts.append(b)
        da.append(np.abs(a-b))
    area = np.array(area)
    da = np.array(da)
    da = np.sum(da)/np.sum(gts)
    return area,da

def F_measure(preds, labels, openset=True, unknown=-1): # F1
    if openset:
        true_pos = 0.
        false_pos = 0.
        false_neg = 0.
        for i in range(len(labels)):
            true_pos += 1 if preds[i] == labels[i] and labels[i] != unknown else 0
            false_pos += 1 if preds[i] != labels[i] and labels[i] != unknown else 0
            false_neg += 1 if preds[i] != labels[i] and labels[i] == unknown else 0

        precision = true_pos / (true_pos + false_pos + 1e-12)
        recall = true_pos / (true_pos + false_neg + 1e-12)
        return 2 * ((precision * recall) / (precision + recall + 1e-12))

#%%
avg = []
cl = 50 # threshold
if True:
    fp0 = r'G:\open_hsi_code\r0710a/'
    key = 'salinas'
    data = '_raw_'
    num = str(20)
    closs = '_closs'+str(cl)
    fp = fp0+key+data+num+closs+'/'
    gt2file = glob.glob('data/'+key+'*gt*')[0]
    
    
    #%%
    oas = []
    f1s = []
    errs = []

    #%%
    mode = 1 #### change here!!!!!!!!!!!!!!!!
    opendic = 0.5
    seedi = 0
    for seedi in range(10):
#    for seedi in [0,2,3,5,7,8,9]:
        gt1file = glob.glob('data/'+key+data+'*')[0]
        gt1 = np.load(gt1file)
        inclass = np.unique(gt1)
        unknown = gt1.max()+1
        gt2 = np.load(gt2file)
        gt1[np.logical_and(gt1==0,gt2!=0)] = unknown
        
        # OA
        gt = copy.deepcopy(gt1)
        pre = read(fp,mode,key,seedi,opendic=opendic,cls1=unknown,num=int(num))
        cfm = rscls.gtcfm(pre,gt,unknown)
        oas.append(cfm[-1,0])
        
        # F1
        gt = gt.reshape(-1)
        pre = pre.reshape(-1)
        pre = pre[gt!=0]
        gt = gt[gt!=0]
        f1s.append(F_measure(pre,gt,openset=True,unknown=unknown))
        
        # area
        _area,_da = calarea(pre,gt,inclass)
        errs.append(_da)
        
    #%%
    a1 = {}
    b1 = {}
    a1['oa'] = []
    a1['f1'] = []
    a1['a'] = []
    b1['oa'] = []
    b1['f1'] = []
    b1['a'] = []
        
        
    #%% oa
    x = np.array(oas)*100
    print('oa,',x.mean(),x.std())
    a1['oa'].append(x.mean())
    b1['oa'].append(x.std())    
    
    # f1
    x = np.array(f1s)*100
    print('f1,',x.mean(),x.std())
    a1['f1'].append(x.mean())
    b1['f1'].append(x.std()) 
    
    # errs
    x = np.array(errs)*100
    print('error,',x.mean(),x.std())
    a1['a'].append(x.mean())
    b1['a'].append(x.std())    
        
    
    #%%
    a0 = []
    a0std = []
    for key in a1.keys():
        a0.append(a1[key][0])
        a0std.append(b1[key][0])
        
    a0 = np.array(a0).reshape(1,-1)
    a0std = np.array(a0std).reshape(1,-1)
    avg.append(a0)
avg = np.array(avg)
avg = avg.reshape(avg.shape[0],3)
avg2 = avg[:,1:]

    