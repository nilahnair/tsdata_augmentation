# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:34:56 2024

@author: nilah
"""
import numpy as np
import matplotlib.pyplot as plt

def check_sampling(x,title):
    
    print(x)
    t = np.arange(0, x.shape[1], 1)
    #print(t)
    fig, ax = plt.subplots()
    for signal in range(x.shape[2]):
        ax.plot(t, x[0,:,signal])
    plt.title(title)
    plt.show()
    
    return

def resampling_random(x):
    '''
    https://github.com/diheal/resampling/blob/main/Augment.py

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    x_selected : TYPE
        DESCRIPTION.

    '''
    import random
    M = random.randint(1, 3)
    print(M)
    N = random.randint(0, M - 1)
    print(N)
    assert M > N, 'the value of M have to greater than N'

    timesetps = x.shape[1]
    print('shape of x {}'.format(x.shape))

    for i in range(timesetps - 1):
        x1 = x[:, i * (M + 1), :]
        #print('shape of x1 {}'.format(x1.shape))
        x2 = x[:, i * (M + 1) + 1, :]
        #print('shape of x2 {}'.format(x2.shape))
        for j in range(M):
            v = np.add(x1, np.subtract(x2, x1) * (j + 1) / (M + 1))
            #print('shape of v {}'.format(v.shape))
            x = np.insert(x, i * (M + 1) + j + 1, v, axis=1)
            print('shape of x {}'.format(x.shape))
    title='plot2'
    check_sampling(x,title)
    length_inserted = x.shape[1]
    num = x.shape[0]
    start = random.randint(0, length_inserted - timesetps * (N + 1))
    index_selected = np.arange(start, start + timesetps * (N + 1), N + 1)
    x_selected=x[0,index_selected,:][np.newaxis,]
    for k in range(1,num):
        start = random.randint(0, length_inserted - timesetps * (N + 1))
        index_selected = np.arange(start, start + timesetps * (N + 1), N + 1)
        x_selected = np.concatenate((x_selected,x[k,index_selected,:][np.newaxis,]),axis=0)
    title='plot3'
    check_sampling(x_selected,title)
    return 

if __name__ == '__main__':
    x=np.random.rand(1,100,1)
    title='plot1'
    check_sampling(x,title)
    resampling_random(x)
    