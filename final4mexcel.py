# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 22:11:07 2024

@author: nilah
"""

import matplotlib.pyplot as plt
import pandas as pd

sheets= ["newLARa_imu", "newMobiact", "newMS", "newSisfall", "motionminer"]
#sheets= ["newLARa_imu half correct", "newMobiact half correct", "newMS half correct", "newSisfall half correct", "motionminer half correct"]
title= {"newLARa_imu":'LARa_imu',  "newMobiact":'Mobiact', "newMS":'Motionsense', "newSisfall":'Sisfall', "motionminer":'LARa_mm'}

def draw(*args):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] 
    
    title = "LSTM" #####<<<<<<<<<-------------
    
    plt.figure(figsize=(5, 5))  # MAKE SURE TO ADJUST THIS!!!

    for i, (x, y, error, label) in enumerate(args):
        print('Args i')
        print(i)
        color = colors[i] 
        print(x)

        plt.plot(x, y, color=color, label=label) #linewidth=2, 
        plt.fill_between(x, y-error, y+error, color=color, alpha=0.2) 

    plt.xticks(rotation=45, ha='right')
    #if title == "LSTM":
    #    plt.xlabel('Augmentations')
    if title == "CNN":
        plt.ylabel('Accuracy')
    plt.title(title) 
    if title == "CNN-Transformer":
        plt.legend(loc='right', framealpha=0.4, fontsize="10")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams.update({'font.size': 13})
    plt.tight_layout()
    plt.ylim((10,100))
    plt.show()
    

if __name__=="__main__":
    path_to_excel = "C:/Users/nilah/OneDrive/Desktop/dataaugmentation.xlsx"
    network= 'lstm' #'cnn' 'lstm' 'cnntrans' #####<<<<<<<<<-------------
    
    df = pd.read_excel(path_to_excel,sheet_name="newLARa_imu") #sheets[1]

    if network == 'cnn':            
        x = df['network'].iloc[1:15]
        #print(x)
        y = df['acc'].iloc[1:15].astype(float)
        e = df['accstd'].iloc[1:15].astype(float)
    elif network == 'lstm':
        x = df['network'].iloc[18:32]
        #print(x)
        y = df['acc'].iloc[18:32].astype(float)
        e = df['accstd'].iloc[18:32].astype(float)
    elif network == 'cnntrans':
        x = df['network'].iloc[35:49]
        #print(x)
        y = df['acc'].iloc[35:49].astype(float)
        e = df['accstd'].iloc[35:49].astype(float)
        
    df = pd.read_excel(path_to_excel,sheet_name="newMobiact") #sheets[1]

    if network == 'cnn':            
        x1 = df['network'].iloc[1:15]
        #print(x)
        y1 = df['acc'].iloc[1:15].astype(float)
        e1 = df['accstd'].iloc[1:15].astype(float)
    elif network == 'lstm':
        x1 = df['network'].iloc[18:32]
        #print(x)
        y1 = df['acc'].iloc[18:32].astype(float)
        e1 = df['accstd'].iloc[18:32].astype(float)
    elif network == 'cnntrans':
        x1 = df['network'].iloc[35:49]
        #print(x)
        y1 = df['acc'].iloc[35:49].astype(float)
        e1 = df['accstd'].iloc[35:49].astype(float)
        
    df = pd.read_excel(path_to_excel,sheet_name="newMS") #sheets[1]

    if network == 'cnn':            
        x2 = df['network'].iloc[1:15]
        #print(x)
        y2 = df['acc'].iloc[1:15].astype(float)
        e2 = df['accstd'].iloc[1:15].astype(float)
    elif network == 'lstm':
        x2 = df['network'].iloc[18:32]
        #print(x)
        y2 = df['acc'].iloc[18:32].astype(float)
        e2 = df['accstd'].iloc[18:32].astype(float)
    elif network == 'cnntrans':
        x2 = df['network'].iloc[35:49]
        #print(x)
        y2 = df['acc'].iloc[35:49].astype(float)
        e2 = df['accstd'].iloc[35:49].astype(float)
        
    df = pd.read_excel(path_to_excel,sheet_name="newSisfall") #sheets[1]

    if network == 'cnn':            
        x3 = df['network'].iloc[1:15]
        #print(x)
        y3 = df['acc'].iloc[1:15].astype(float)
        e3 = df['accstd'].iloc[1:15].astype(float)
    elif network == 'lstm':
        x3 = df['network'].iloc[18:32]
        #print(x)
        y3 = df['acc'].iloc[18:32].astype(float)
        e3 = df['accstd'].iloc[18:32].astype(float)
    elif network == 'cnntrans':
        x3 = df['network'].iloc[35:49]
        #print(x)
        y3 = df['acc'].iloc[35:49].astype(float)
        e3 = df['accstd'].iloc[35:49].astype(float)
        
    df = pd.read_excel(path_to_excel,sheet_name="motionminer") #sheets[1]

    if network == 'cnn':            
        x4 = df['network'].iloc[1:15]
        #print(x)
        y4 = df['acc'].iloc[1:15].astype(float)
        e4 = df['accstd'].iloc[1:15].astype(float)
    elif network == 'lstm':
        x4 = df['network'].iloc[18:32]
        #print(x)
        y4 = df['acc'].iloc[18:32].astype(float)
        e4 = df['accstd'].iloc[18:32].astype(float)
    elif network == 'cnntrans':
        x4 = df['network'].iloc[35:49]
        #print(x)
        y4 = df['acc'].iloc[35:49].astype(float)
        e4 = df['accstd'].iloc[35:49].astype(float)
        
    print('LARa_imu')
    print(x)
    print(y)
    print(e)
    print('Mobiact')
    print(x1)
    print(y1)
    print(e1)
    print('Motionsense')
    print(x2)
    print(y2)
    print(e2)
    print('Sisfall')
    print(x3)
    print(y3)
    print(e3)
    print('LARa_mm')
    print(x4)
    print(y4)
    print(e4)
    draw((x,y,e, "LARa_imu"), (x,y1,e1, "Mobiact"), (x,y2,e2, "Motionsense"), (x,y3,e3, "Sisfall"), (x,y4,e4, "LARa_mm"))