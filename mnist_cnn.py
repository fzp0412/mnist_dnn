import time
import numpy as np
from scipy import misc
from scipy import signal
import mnist_mnn as mnn
import skimage.measure

'''
global value
'''
batch_size = 128
epochs = 1
filter1_rate =0.00049
filter2_rate =0.00025
hide1_rate   =0.00010
filter1_size = 32
filter2_size = 64
class_num = 10
hide1_num = int(filter2_size*((28-2*2)/2)**2) #9126
hide2_num = 128

'''
180 degree rotation function
'''
def rot_fun(x):
    return x

'''
use convolve2d to achieve convolve3d funcition
'''
def cov3d_fun(x,fil):
    z = np.zeros((x.shape[0],fil.shape[0],(x.shape[1]-fil.shape[1]+1),(x.shape[2]-fil.shape[2]+1)))
    for i in range(fil.shape[0]):
        for j in range(x.shape[0]):
            z[j][i] = signal.convolve2d(x[j],rot_fun(fil[i]),'valid')
    return z

'''
use convolve2d to achieve convolve4d funcition
'''
def cov4d_fun(x,fil):
    z = np.zeros((x.shape[0],fil.shape[0],(x.shape[2]-fil.shape[2]+1),(x.shape[3]-fil.shape[3]+1)))
    for i in range(fil.shape[0]):
        for j in range(x.shape[0]):
            for k in range(x.shape[1]):
                z[j][i] += signal.convolve2d(x[j][k],rot_fun(fil[i][k]),'valid')
    return z
'''
max pooling function
'''
def max_pool_fun(x):
    z= np.zeros((x.shape[0],x.shape[1],int(x.shape[2]/2),int(x.shape[3]/2)))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i][j]= skimage.measure.block_reduce(x[i][j], (2,2), np.max)
    return z
'''
flatten function 
'''
def flatten_fun(x):
    x=x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
    return x

'''
output layer function
input x.shape(num,784)
input w1.shape(784,hide1_num)
input w2.shape(hide1_num,hide2_num)
input w3.shape(hide2_num,class_num)
output z.shape(num,class_num)
'''
def output_layer(x,filter1,filter2,w1,w2):
    z1 = cov3d_fun(x,filter1)
    x2 = z1
    z2 = cov4d_fun(x2,filter2)
    x3 = z2
    z3 = max_pool_fun(x3)
    x4 = flatten_fun(z3)
    z4 = np.dot(x4,w1)
    x5 = mnn.relu_fun(z4)
    z5 = np.dot(x5,w2)
    return x2,x3,x4,x5,z1,z2,z3,z4,z5

'''
last recognition function
input x.shape(num,784)
input w1.shape(784,hide1_num)
input w2.shape(hide1_num,hide2_num)
input w3.shape(hide2_num,class_num)
output z.shape(num,class_num)
output sum_exp_z.shape(num,1)
output y.shape(num,class_num)
'''
def recognition_fun(x,filter1,filter2,w1,w2):
    x2,x3,x4,x5,z1,z2,z3,z4,z5 = output_layer(x,filter1,filter2,w1,w2)
    exp_z = np.exp(z5)
    exp_z =exp_z.T
    y = exp_z/(exp_z.sum(axis =0))
    y=y.T
    return (x2,x3,x4,x5,z1,z2,z3,z4,z5,y)


(x_train, y_train), (x_test, y_test) = mnn.load_data()
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
filter1 = np.random.randn(filter1_size,3,3)
filter2 = np.random.randn(filter2_size,filter1_size,3,3)
filter1 = filter1/100 
filter2 = filter2/100 
w1 = np.random.randn(hide1_num,hide2_num)
w2 = np.random.randn(hide2_num,class_num)
w1 = w1/100
w2 = w2/100

def training_fun(data,label,filter1,filter2,w1,w2):
    inner_size = int(data.shape[0]/batch_size)
    for i in range(epochs):
        for j in range(inner_size):
            batch_data  = data[j*batch_size:(j+1)*batch_size]
            batch_label = label[j*batch_size:(j+1)*batch_size]
            recognition_fun(batch_data,filter1,filter2,w1,w2)
            print(j)

s_time = int(time.time())
training_fun(x_train,y_train,filter1,filter2,w1,w2)
e_time = int(time.time())
print("%02d:%02d:%02d" %((e_time-s_time)/3600,(e_time-s_time)%3600/60,(e_time-s_time)%60))

