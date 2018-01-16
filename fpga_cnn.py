import time
import numpy as np
from scipy import misc
from scipy import signal
import mnist_mnn as mnn
import skimage.measure
import parameter

'''
global value
'''
filter1_size = 4
filter2_size = 8
class_num = 10
hide1_num = int(filter2_size*((28-2*2)/3)**2) #512
hide2_num = 64

'''
2d 180 degree rotation function
'''
def rot_fun(x):  
    z = x.reshape((x.shape[0]*x.shape[1]))[::-1].reshape((x.shape[0],x.shape[1]))
    return z

'''
use convolve2d to achieve forword convolve3d funcition
'''
def cov3d_for_fun(x,fil,fb):
    z = np.zeros((x.shape[0],fil.shape[0],(x.shape[1]-fil.shape[1]+1),(x.shape[2]-fil.shape[2]+1)))
    for i in range(fil.shape[0]):
        for j in range(x.shape[0]):
            z[j][i] = signal.convolve2d(x[j],rot_fun(fil[i]),'valid')
            z[j][i] = z[j][i]+fb[i]
    return z

'''
use convolve2d to achieve forword convolve4d funcition
'''
def cov4d_for_fun(x,fil,fb):
    z = np.zeros((x.shape[0],fil.shape[0],(x.shape[2]-fil.shape[2]+1),(x.shape[3]-fil.shape[3]+1)))
    for i in range(fil.shape[0]):
        for j in range(x.shape[0]):
            for k in range(x.shape[1]):
                z[j][i] += signal.convolve2d(x[j][k],rot_fun(fil[i][k]),'valid')
            z[j][i]=z[j][i]+fb[i]
    return z

'''
max pooling function
'''
def max_pool_fun(x):
    z= skimage.measure.block_reduce(x, (1,1,3,3), np.max)
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
input filter1.shape(filter1_size,3,3)
input filter1.shape(filter2_size,filter1_size,3,3)
input w1.shape(hide1_num,hide2_num)
input w2.shape(hide2_num,class_num)
output z.shape(num,class_num)
'''
def output_layer(x,filter1,fb1,filter2,fb2,w1,b1,w2,b2):
    z1 = cov3d_for_fun(x,filter1,fb1)
    x2 = mnn.relu_fun(z1)
    z2 = cov4d_for_fun(x2,filter2,fb2)
    x3 = mnn.relu_fun(z2)
    z3 = max_pool_fun(x3)
    x4 = flatten_fun(z3)
    z4 = np.dot(x4,w1)+b1
    x5 = mnn.relu_fun(z4)
    z5 = np.dot(x5,w2)+b2
    return x2,x3,x4,x5,z1,z2,z3,z4,z5

'''
test function
'''
def test_fun(x_test,y_test,filter1,fb1,filter2,fb2,w1,b1,w2,b2):
    tax2,tax3,tax4,tax5,taz1,taz2,taz3,taz4,taz5 = output_layer(x_test,filter1,fb1,filter2,fb2,w1,b1,w2,b2)
    acc = mnn.test_fun(x_test,y_test,taz5)
    print(acc)

'''
get paramater
'''
def get_para(params):
    new_paras =[]
    for param in params:
        if param :
            new_paras.append(param)
    arrs = np.array(new_paras)
    filter1 = np.zeros((filter1_size,1,3,3))
    for i in range (filter1_size):
        for j in range(1):
            filter1[i,j] = arrs[0][0][:,:,j,i]
    filter1 = filter1.reshape(filter1_size,3,3) 
    fb1 = arrs[0][1]
    filter2 = np.zeros((filter2_size,filter1_size,3,3))
    for i in range (filter2_size):
        for j in range(filter1_size):
            filter2[i,j] = arrs[1][0][:,:,j,i]
    fb2 = arrs[1][1]
    w1 = arrs[2][0] 
    b1 = arrs[2][1] 
    w2 = arrs[3][0] 
    b2 = arrs[3][1]
    return filter1,fb1,filter2,fb2,w1,b1,w2,b2

'''
main funciton
'''
def run():
    s_time = int(time.time())
    (x_train, y_train), (x_test, y_test) = mnn.load_data()
    x_test  = x_test.astype('float32')
    x_test = x_test/255;
    filter1 = np.random.randn(filter1_size,3,3)
    filter2 = np.random.randn(filter2_size,filter1_size,3,3)
    fb1 =np.random.randn(filter1_size) 
    fb2 =np.random.randn(filter1_size) 
    w1 = np.random.randn(hide1_num,hide2_num)
    b1 = np.random.randn(hide2_num)
    w2 = np.random.randn(hide2_num,class_num)
    b2 = np.random.randn(class_num)
    print(x_test.shape)
    params = parameter.parameter
    filter1,fb1,filter2,fb2,w1,b1,w2,b2 = get_para(params)
    test_fun(x_test,y_test,filter1,fb1,filter2,fb2,w1,b1,w2,b2)
    e_time = int(time.time())
    print("%02d:%02d:%02d" %((e_time-s_time)/3600,(e_time-s_time)%3600/60,(e_time-s_time)%60))


if __name__ =='__main__':
    run()

