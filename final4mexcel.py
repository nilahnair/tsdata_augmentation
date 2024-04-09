# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 22:11:07 2024

@author: nilah
"""

import matplotlib.pyplot as plt
import pandas as pd

sheets= ["newLARa_imu", "newMobiact", "newMS", "newSisfall"]
title= {"newLARa_imu":'LARa_imu',  "newMobiact":'Mobiact', "newMS":'Motionsense', "newSisfall":'Sisfall'}


def draw(*args):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] 
    title = "LSTM"
    
    plt.figure(figsize=(5, 5))  # MAKE SURE TO ADJUST THIS!!!

    for i, (x, y, error, label) in enumerate(args):
        print('Args i')
        print(i)
        color = colors[i] 
        print(x)

        plt.plot(x, y, color=color, label=label) #linewidth=2, 
        plt.fill_between(x, y-error, y+error, color=color, alpha=0.2) 

    plt.xticks(rotation=45, ha='right')
    plt.xlabel('augmentation')
    plt.ylabel('accuracy')
    plt.title(title) 
    plt.legend(loc='lower right')
    plt.rcParams["font.family"] = "Times New Roman"
    plt.tight_layout() 
    plt.show()
    

if __name__=="__main__":
    path_to_excel = "C:/Users/nilah/OneDrive/Desktop/dataaugmentation.xlsx"
    network= 'lstm' #'cnn' 'lstm' 'cnntrans'
    
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
    draw((x,y,e, "LARa_imu"), (x,y1,e1, "Mobiact"), (x,y2,e2, "Motionsense"), (x,y3,e3, "Sisfall"))