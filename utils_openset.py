# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:56:22 2020

@author: sj
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rscls

colors = ['#000000','#CACACA', '#02FF00', '#00FFFF', '#088505', '#FF00FE', '#AA562E', '#8C0085', '#FD0000', '#FFFF00']
cmap = ListedColormap(colors)

#0000FF, #228B22, #7BFC00, #FF0000, #724A12, #C0C0C0, #00FFFF, #FF8000, #FFFF00

def save_cmap_hk(img, cmap, fname):
    colors = ['#000000','#008000','#808080','#FFF700','#0290DE','#EDC9Af','#F3F2E7']
    cmap = ListedColormap(colors)
   
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(img, cmap=cmap, vmin=0, vmax=6)
    plt.savefig(fname, dpi = height)
    plt.close()

def save_cmap_pc(img, cmap, fname):
    colors = ['#000000','#0000FF','#228B22','#7BFC00', '#FF0000', '#724A12', '#C0C0C0',
              '#00FFFF', '#FF8000', '#FFFF00']
    cmap = ListedColormap(colors)
   
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(img, cmap=cmap, vmin=0, vmax=9)
    plt.savefig(fname, dpi = height)
    plt.close()
    
def save_cmap_salinas16(img,cmap,fname):
    colors = ['#000000','#DCB809','#03009A','#FE0000','#FF349B','#FF66FF',
              '#0000FD','#EC8101','#00FF00','#838300','#990099','#00F7F1',
              '#009999','#009900','#8A5E2D','#67FECB','#F6EF00']
    cmap = ListedColormap(colors)
   
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(img, cmap=cmap, vmin=0, vmax=16)
    plt.savefig(fname, dpi = height)
    plt.close()
    
def save_cmap_indian16(img,cmap,fname):
    colors = ['#000000','#FFFC86','#0037F3','#FF5D00','#00FB84','#FF3AFC',
              '#4A32FF','#00ADFF','#00FA00','#AEAD51','#A2549E','#54B0FF',
              '#375B70','#65BD3C','#8F462C','#6CFCAB','#FFFC00']
    cmap = ListedColormap(colors)
   
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(img, cmap=cmap, vmin=0, vmax=16)
    plt.savefig(fname, dpi = height)
    plt.close()

def save_cmap_pu9(img, cmap, fname):
    colors = ['#000000','#CACACA','#02FF00','#00FFFF','#088505','#FF00FE','#AA562E','#8C0085','#FD0000', '#FFFF00']
    cmap = ListedColormap(colors)
   
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(img, cmap=cmap, vmin=0, vmax=9)
    plt.savefig(fname, dpi = height)
    plt.close()
    
def save_im(img,fname):
   
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(img)
    plt.savefig(fname, dpi = height)
    plt.close()
    
#save_im(rscls.strimg255(im[:,:,[50,34,20]],5),'indian_im')
#plt.imshow(rscls.strimg255(im[:,:,[50,34,20]],5))
    
#save_cmap(pre1,cmap,'a')
    

    


