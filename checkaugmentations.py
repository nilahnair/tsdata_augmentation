import numpy as np
import matplotlib.pyplot as plt

def check_vertical_flip():
    
    x=range(1,101)
    print(x)
    y1=np.random.rand(1,100,9)
    y2=y1
    #print(y1)
    #y2=np.flip(y1,1)
    b_idx = np.arange(y1.shape[1])
    print(b_idx)
    np.random.shuffle(b_idx)
    print(b_idx)
    crop_num = int(y1.shape[1]*0.5)
    print(crop_num)
    y1[:,b_idx[:crop_num],:] = 0
    #y2=((y1-0.5)*-1)+0.5
    #print(y2)
    
    plt.plot(x, y1[0,:,0], label ='y1')
    plt.plot(x, y2[0,:,0], '-.', label ='y2')

    plt.xlabel("X-axis data")
    plt.ylabel("Y-axis data")
    plt.legend()
    plt.title('multiple plots')
    plt.show()

    return


if __name__=='__main__':
    check_vertical_flip()