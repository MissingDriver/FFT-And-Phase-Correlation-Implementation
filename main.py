import sys
import numpy as np
import cv2
import pyximport
import time 
import transforms #this is my custom transforms library, look in cython_stuff/transforms.pyx for the details.
import math 
import ctypes
pyximport.install()

np.set_printoptions(threshold=sys.maxsize)

def getIndex(pixel, regions):
        for region in regions:
            if pixel in region:
                return regions.index(region)
        return -1

def getCenter(image): #Finds circular blobs in the image and returns the center of the largest one
    regions = []
    up = -1
    for index, pixel in np.ndenumerate(image):
        left = -1
        if pixel == 0:
            if(index[1]-1) >= 0:
                left = getIndex((index[0],index[1]-1), regions)
            if(index[0]-1) >= 0:
                up = getIndex((index[0]-1,index[1]), regions)
            if(left > -1) and (up > -1) and (left != up):
                regions[left] += regions[up]
                regions[left].append(index)
                del(regions[up])
            elif left > -1:
                regions[left].append(index)
            elif up > -1:
                regions[up].append(index)
            else:
                regions.append([index])
    
    maxReg = regions[0]
    for region in regions:
        if len(region) > len(maxReg):    
            maxReg = region           
    center = (0,0)
    for point in maxReg:
        center = (center[0] + point[0] , center[1] + point[1])
    center = (center[0]+.5, center[1]+.5)
    center = (int(center[0]/len(maxReg)), int(center[1]/len(maxReg)))

    return center

def getNumOfZero(num): #gets the number of zeros needed for zero padding
    ans = math.log(num,2)
    
    if ans > int(ans):
        ans = int(ans)+1
    ans = (2 ** ans) - num
    return int(ans)

def padImage(image): #returns a zero padded version of the image so that its # of pixels is a power of two which is needed for fft
    ypad = getNumOfZero(image.shape[0])
    xpad = getNumOfZero(image.shape[1])
    return np.pad(image, ((0,int(ypad)),(0,int(xpad))), 'constant')

#Im currently using tracker as the main loop. IDK how our structure is gona be in the end
def tracker(scaler = 1): #scaler works best with powers of 2
    originalBall = cv2.imread("ball-gif/1.png",0) 
    center = getCenter(originalBall) #finds the initial location of the object you want to track

    originalDft = transforms.forward_transform(padImage(cv2.resize(originalBall, (0,0), fx = (1/scaler), fy = (1/scaler)).astype(np.complex))) #computes the dft and scales the image down to desierd size

    for i in range(1, 8):
        
        changeBall = cv2.imread("ball-gif/"+ str(i+1) +".png",0)
        start = time.time()
        
        changeDft = transforms.forward_transform(padImage(cv2.resize(changeBall, (0,0), fx = (1/scaler), fy = (1/scaler)).astype(np.complex)))
        
        locDft = changeDft * originalDft / abs(changeDft * originalDft) 

        location = transforms.inverse_transform(locDft).real
        end = time.time()
        temp = np.where(location == np.amax(location))
        
        cent = ((temp[0]*scaler)-center[0], (temp[1]*scaler)-center[1])

        for y in range(0,6):
            for x in range(0,6):
                changeBall[(cent[0]+y, cent[1])] = 255
                changeBall[(cent[0]-y, cent[1])] = 255
                changeBall[(cent[0], cent[1]+x)] = 255
                changeBall[(cent[0], cent[1]-x)] = 255

        print("Center: ",temp[1], ",", temp[0], "\nTime: ", end-start)

        cv2.imshow('Location', changeBall)
        
        cv2.waitKey(1)

    cv2.destroyAllWindows()
tracker(4)