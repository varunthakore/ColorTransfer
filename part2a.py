#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIP assignment 1

Part 2: Transferring Color to Greyscale Images

"""
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import generic_filter



path1 = '/Users/thakore/Desktop/My Documents/Digital Image Analysis/Assignments/A1/Final Submission/1_src.png'
path2 = '/Users/thakore/Desktop/My Documents/Digital Image Analysis/Assignments/A1/Final Submission/1_tar.png'
#read target and source image
sou = cv2.imread(path1)
sou = cv2.cvtColor(sou, cv2.COLOR_BGR2Lab)
sou=np.array(sou)
source=sou.copy() #To plot in the output


tar = cv2.imread(path2,0)
target = cv2.imread(path2,0) #To plot in the output
tar=np.array(tar)


#luminance remapping
mu_sou=np.mean(sou[:,:,0])
mu_tar=np.mean(tar)
sig_sou=np.std(sou[:,:,0])
sig_tar=np.std(tar)
sou[:,:,0]= (sig_tar/sig_sou)*(sou[:,:,0]-mu_sou)+mu_tar


#Calculate SD
sou_filt = generic_filter(sou, np.std, size=5)
tar_filt = generic_filter(tar, np.std, size=5)


#print(sou.shape, sou_filt.shape)

#Jittered Sampling
samp_pix=[]
x,y,_ =sou.shape
x_step=math.floor(x/16)
y_step=math.floor(y/16)
for i in  range(16):
    for j in range(16):
        x_samp=np.random.choice(list(range(x_step*i,(x_step*(i+1))+1)))
        y_samp=np.random.choice(list(range(y_step*j,(y_step*(j+1))+1)))
        samp_pix.append((x_samp,y_samp,sou_filt[x_samp,y_samp,0]*0.5+sou[x_samp,y_samp,0]*0.5))
             
#Match pixel values
m,n =tar.shape
newtar=np.zeros((m, n, 3))
newtar[:,:,0]=tar
for i in range(m):
    for j in range(n):
        wavg=tar[i,j]*0.5+ tar_filt[i,j]*0.5
        minimum=1e10
        for k in range(len(samp_pix)):
            diff=abs(wavg-samp_pix[k][2])
            if diff <= minimum:
                    minimum = diff
                    newtar[i,j,1]=sou[samp_pix[k][0],samp_pix[k][1],1]
                    newtar[i,j,2]=sou[samp_pix[k][0],samp_pix[k][1],2]



    
#plot images
source = cv2.cvtColor(source, cv2.COLOR_Lab2RGB)
newtar = newtar.astype(np.uint8)
newtar = cv2.cvtColor(newtar, cv2.COLOR_Lab2RGB)
fig = plt.figure()
ax2 = fig.add_subplot(1,3,1)
ax2.imshow(source)
ax2 = fig.add_subplot(1,3,2)
ax2.imshow(target)
ax2 = fig.add_subplot(1,3,3)
ax2.imshow(newtar)


# write_img = cv2.cvtColor(newtar, cv2.COLOR_RGB2BGR)
# cv2.imwrite("Global_result.png", newtar) 
    