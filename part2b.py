#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Color Transfer with Swatches
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import generic_filter
import math

def getSD(image, filter_size): #returns std deviation in 5*5 neighbour
    #print(image.shape)
    image_std = generic_filter(image, np.std, size=filter_size)
    return image_std

def l2_dist(img1, img2): #l2 distance of m*m neighbour
    m,m,_ = img1.shape
    dist = np.sum((img1[:,:,0]-img2[:,:,0])**2)
    
    return math.sqrt(dist)


#read color image
img = cv2.imread('/Users/thakore/Desktop/My Documents/Digital Image Analysis/Assignments/A1/Final Submission/1_src.png')
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_img)
print(rgb_img.shape)

#convert into Lab
color_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

#read grayscale image
gray_img = cv2.imread('/Users/thakore/Desktop/My Documents/Digital Image Analysis/Assignments/A1/Final Submission/1_tar.png',0)
plt.imshow(gray_img, cmap = 'gray')
print(gray_img.shape)



#target image
m,n = gray_img.shape
newtar=np.zeros((m, n, 3))
newtar[:,:,0]=gray_img




#take swatches as input-
n = int(input("Enter number of swatches- "))

print("\nSwatch input format-\n")
print(" color swatch1, gray swatch1, color swatch2, gray swatch2, color swatch3........")
print(" x1 y1 x2 y2, x1 y1 x2 y2, x1 y1 x2 y2, x1 y1...")

string = input()
list1 = list(string.split(','))
    
src_swatches = []
tar_swatches = []
for i in range(0,n*2,2):
    # take source image swatch as input
    (a,b,c,d) = tuple(map(int, list1[i].split()))

    
    src_swatches.append(((a,b), (c,d)))
  
    # take target swatch as input

    (a,b,c,d) = tuple(map(int, list1[i+1].split()))
    tar_swatches.append(((a,b), (c,d)))



#transfer color between swatches
for sw in range(n):
    
    (a,b) = src_swatches[sw][0]
    (c,d) = src_swatches[sw][1]
    color_swatch = color_img[a:c, b:d]
  
   
    (a,b) = tar_swatches[sw][0]
    (c,d) = tar_swatches[sw][1]
    gray_swatch = gray_img[a:c, b:d]
    #Luminance Remapping-
    
    #find mean and std deviation of luminance in color swatch
    color_mu= np.mean(color_swatch[:,:,0])
    color_sigma = np.std(color_swatch[:,:,0])
    
    #find mean and std deviation of luminance in grayscale swatch
    gray_mu = np.mean(gray_swatch)
    gray_sigma= np.std(gray_swatch)
    #luminance remapping between corrosponding swatches     
    color_swatch[:,:,0] = (gray_sigma/color_sigma)*(color_swatch[:,:,0]-color_mu)+gray_mu

    #compute std deviation for the color swatch neighbour
    sigma1 = getSD(color_swatch[:,:,0], 5)
    #compute std deviation for the gray swatch neighbour
    sigma2 = getSD(gray_swatch, 5)
    #Jittered Sampling in a swatch
    samp_pix=[]
    x,y,_ = color_swatch.shape
    x_step=math.floor(x/7)
    y_step=math.floor(y/7)
    for i in  range(7):
        for j in range(7):
            
            x_samp=np.random.choice(list(range(x_step*i,(x_step*(i+1)))))
            y_samp=np.random.choice(list(range(y_step*j,(y_step*(j+1)))))
            samp_pix.append((x_samp,y_samp,color_swatch[x_samp,y_samp,0]*0.5+sigma1[x_samp,y_samp]*0.5))
                 
    #Match pixel values
    m,n = gray_img.shape
    #print(gray_swatch.shape, sigma2.shape)
    for i in range(c-a):
        for j in range(d-b):
            wavg=gray_swatch[i,j]*0.5+ sigma2[i,j]*0.5
            minimum=1e10
            for k in range(len(samp_pix)):
                diff=abs(wavg-samp_pix[k][2])
                if diff <= minimum:
                        minimum = diff
                        newtar[a+i,b+j,1]=color_swatch[samp_pix[k][0],samp_pix[k][1],1]
                        newtar[a+i,b+j,2]=color_swatch[samp_pix[k][0],samp_pix[k][1],2]
      





#Precompute Jittered Samples for each of the newtar image swatch
newtar_samp_pix=[]
for k in range(len(tar_swatches)):
    #compute samp pix for swatch k in target swatches
    samp_pix=[]
    
    (a,b) = tar_swatches[k][0]
    (c,d) = tar_swatches[k][1]
    color_swatch = newtar[a:c,b:d]
    x,y,_ = color_swatch.shape
    x_step=math.floor(x/7)
    y_step=math.floor(y/7)
    for ii in  range(7):
        for jj in range(7):    
            x_samp=np.random.choice(list(range(x_step*ii,(x_step*(ii+1)))))
            y_samp=np.random.choice(list(range(y_step*jj,(y_step*(jj+1)))))
            samp_pix.append((a+x_samp,b+y_samp,color_swatch[x_samp,y_samp,0]))
    
    newtar_samp_pix.append(samp_pix)
    
    


# take a n*n neighbour
# find l2 distance in a n*n neighbour in each swatch
# transfer color from selected swatch to this n*n box
m,n,_ = newtar.shape
#print(m,n)grayscale image


for i in range(0,m,7):
    for j in range(0,n,7): 
        #get 7*7 neighbour in target image
        x,y  = 7,7
        if m-i < x:
            x = m-i
        if n-j < y:
            y = n-j
        newtar_nbr = newtar[i:i+x, j:j+y,:]
        
        #Texture synthesis
        #select swatch with minimum lumiance difference in 5*5 neighbour
        minimum=1e10
        matched_swatch = 0        
        for k in range(len(tar_swatches)):
            
            (a,b) = tar_swatches[k][0]
            (c,d) = tar_swatches[k][1]
            
            swatch_nbr = newtar[a:a+x, b:b+y, :]
            #find l2 distance
            dist_ = l2_dist(newtar_nbr, swatch_nbr)
            
            if dist_ < minimum:
                minimum = dist_
                matched_swatch = k
                
        #transfer color to 7*7 neighbour
        
        (a,b) = tar_swatches[matched_swatch][0]
        (c,d) = tar_swatches[matched_swatch][1]
        color_swatch = newtar[a:c,b:d] #selected swatch
        
        samp_pix = newtar_samp_pix[matched_swatch]  #samples of selected swatch
        
        #Match pixel values
        for ii in range(min(7,m-i)):
            for jj in range(min(7,n-j)):
                if j+jj>=n:
                    break
                wavg=newtar[i+ii,j+jj,0]
                minimum=1e10
                for k in range(len(samp_pix)):
                    diff=abs(wavg-samp_pix[k][2])
                    if diff <= minimum:
                            minimum = diff
                            newtar[i+ii,jj+j,1]=newtar[samp_pix[k][0],samp_pix[k][1],1]
                            newtar[i+ii,jj+j,2]=newtar[samp_pix[k][0],samp_pix[k][1],2]
           
  
newtar = newtar.astype(np.uint8)
newtar = cv2.cvtColor(newtar, cv2.COLOR_Lab2RGB)
#print(m,n)
plt.imshow(newtar)
# newtar = cv2.cvtColor(newtar, cv2.COLOR_RGB2BGR())
# cv2.imwrite("Swatch_result.png", newtar) 
