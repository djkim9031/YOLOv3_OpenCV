#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 19:10:11 2020

@author: djkim9031
"""


import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture('P1033651.mp4')
whT = 320
#confThresh = 0.3
#nmsThresh = 0.5 #The lower it is, the stricter its standard is, hence less bboxes

classFile = 'coco_classes.txt'
classNames = []
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)
#print(len(classNames))
    
#modelConfig = 'yolov3.cfg'
#modelWeights = 'yolov3.weights'
modelConfig='yolov3-tiny.cfg'
modelWeights='yolov3-tiny.weights'
#net = cv2.dnn.readNet(modelWeights,modelConfig)
net = cv2.dnn.readNetFromDarknet(modelConfig,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
layerNames = net.getLayerNames()
#print(layerNames)
#print(net.getUnconnectedOutLayers()) #Output layer index
outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

def findObjects(outputs, img):
    height,width,_= img.shape
    bbox = []
    confs = []
    class_ids = []
    
    for output in outputs:
        for detect in output:
            scores = detect[5:] #The first 5 values are x,y,w,h,confidence - the remainders are scores for 80 classes
            class_id = np.argmax(scores) #Index for maximum value in the array
            conf = scores[class_id]
            if conf > 0.3:
                cx,cy = int(detect[0]*width),int(detect[1]*height)
                w,h = int(detect[2]*width), int(detect[3]*height)
                x,y = int(cx-w/2),int(cy-h/2)
                bbox.append([x,y,w,h])
                confs.append(float(conf))
                class_ids.append(class_id)
                
    #print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, 0.5, 0.4) #Returning indices of bbox to keep
    #print(len(indices))
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(img,f'{classNames[class_ids[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        




while True:
    _, frame = cap.read()
    frame = imutils.resize(frame,width=1200)
    
    blob = cv2.dnn.blobFromImage(frame,scalefactor=1/255,size=(whT,whT),mean=(0,0,0),swapRB=True,crop=False)
    net.setInput(blob)
    
    
    #print(outputNames)
    
    outputs = net.forward(outputNames)
    #print(outputs[0].shape)
    #print(outputs[1].shape)
    #print(outputs[2].shape)
    findObjects(outputs,frame)


    
    cv2.imshow('Image',frame)
    cv2.waitKey(1)