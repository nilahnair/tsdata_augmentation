# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 14:40:49 2024

@author: nilah
"""
import os
import csv
import ast
import numpy as np
import matplotlib.pyplot as plt

folder_base= 'C:/Users/nilah/OneDrive/Desktop/Documents/augment_test/'
#file = 'jitter_lstm_sisfall.csv'

def is_empty_csv(reader):
    for i, _ in enumerate(reader):
        if i:  # found the second row
            return False
    return True

def aug_plot(file):
    file_name = folder_base + file
    print(file_name)
    toplot=[]
    with open(file_name, 'r') as strlist:
        csvreader= csv.reader(strlist)
        header=next(csvreader)
        for i in range(len(header)):
            toplot.append(ast.literal_eval(header[i]))
        
    # print(len(toplot))
    final= np.array(toplot)
    #print(final.shape)
    #print(final)
    x=[]
    acc=[]
    f1=[]
    for i in range(final.shape[0]):
        x.append(final[i][0])
        acc.append(final[i][1])
        f1.append(final[i][2])
        
    #print(x)
    #print(acc)
    #print(f1)
     
    plt.plot(x,acc, 'r', label='acc')
    plt.plot(x, f1, 'b', label='wf1')
    plt.title(file.split('.csv')[0])
    plt.xlabel('aug parameter')
    #plt.ylabel('acc & f1')
    plt.legend()
    plt.savefig(folder_base + file.split('.csv')[0]+'.png')
    plt.show()
    plt.close()  
    
    return

def groupby(groupname):
    if groupname == 'networks':
        
    elif groupname ==

def main():
    for root,dirs,files in os.walk(folder_base):
        for file in files:
            if file.endswith(".csv"):
                aug_plot(file)
        
    return
    
    

if __name__ == "__main__":
    main()