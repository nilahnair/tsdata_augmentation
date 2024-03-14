# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:51:09 2024

@author: nilah
"""
import os
import csv
from ast import literal_eval
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm, rcParams

folder_base= 'C:/Users/nilah/OneDrive/Desktop/dataaugmentation.xlsx'

sheets= {0:'newLARa_imu', 1:'newMobiact', 2:'newMS', 3:'newSisfall'}
title= {0:'LARa_imu', 1:'Mobiact', 2:'MS', 3:'Sisfall'}
hfont = {'fontname':'Helvetica'}

'''
x = np.array([1, 2, 3, 4, 5])
y = np.power(x, 2) # Effectively y = x**2
e = np.array([1.5, 2.6, 3.7, 4.6, 5.5])

plt.errorbar(x, y, e, linestyle='None', marker='^')

plt.show()

def plotaug(file):
'''

def actaugrel():
    '''relation between the augmentation and the activities'''
    df= pd.read_excel(folder_base,sheet_name=sheets[1])#,header=None,na_values=['NA'],usecols="A:I",skiprows=range(105),nrows=9)
    x=df['network'].iloc[32:40]
    x=df['precision'].iloc[32:40]
    for i in range(len(x)):
        k=x.iloc[i]
        #print(k.dtype)
        lists=k.split()
        for j in range(len(lists)):
            print(lists[j].isnumeric())
    
    
    return


def auggraph():
    '''
    code for generating graphs for all the augmentation techniques

    '''
    
    
    df= pd.read_excel(folder_base,sheet_name=sheets[1])#,header=None,na_values=['NA'],usecols="A:I",skiprows=range(105),nrows=9)
    
    #df.info()
    #df.set_index(["network", "learning", "batchsize", "epoch", "acc", "accstd", "wf1", "wf1std", "precision"], inplace = True,
                           # append = True, drop = False)
                           #17,25 for lstm
    x=df['network'].iloc[32:40]
   
    y=df['acc'].iloc[17:25]
  
    e=df['accstd'].iloc[17:25]
 
    y1=df['wf1'].iloc[17:25]

    e1=df['wf1std'].iloc[17:25]
  
    yr=df['acc'].iloc[32:40]

    er=df['accstd'].iloc[32:40]

    yr1=df['wf1'].iloc[32:40]
 
    er1=df['wf1std'].iloc[32:40]
    
    plt.errorbar(x, y, e, color='r', linestyle='None', marker='o', linewidth=2, capsize=3, label='lstm')
    #plt.errorbar(x,y1,e1,linestyle='None', marker='o', linewidth=2, capsize=3)
    plt.errorbar(x, yr, er, color='b', linestyle='None', marker='*', linewidth=2, capsize=3, label='cnntrans')
    #plt.errorbar(x,yr1,er1,linestyle='None', marker='*', linewidth=2, capsize=3)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('augentation')
    plt.ylabel('accuracy')
    plt.title(title[1])
    plt.legend(loc='lower right')
    plt.rcParams["font.family"] = "Times New Roman"
    plt.show()
    
    return
    
    

if __name__ == "__main__":
    auggraph()
    #actaugrel()