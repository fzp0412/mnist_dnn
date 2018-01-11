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
epochs = 10
filter1_rate =0.1
filter2_rate =0.08
hide1_rate   =0.05
hide2_rate   =0.02
filter1_size = 32
filter2_size = 64
class_num = 10
hide1_num = int(filter2_size*((28-2*2)/2)**2) #9216
hide2_num = 128

'''
2d 180 degree rotation function
'''
def rot_fun(x):  
    z = x.reshape((x.shape[0]*x.shape[1]))[::-1].reshape((x.shape[0],x.shape[1]))
    return z

'''
flatten layer delta
'''
def fla_delta(delta,w,shape):
    delta_new = np.dot(delta,w.T)
    delta_new = delta_new.reshape(shape) 
    return delta_new

'''
mean pooling layer delta
Kronecker product
c = np.kron(a,b)
'''
def mean_pool_delta_fun(delta,n,z):
    mean = np.ones((n,n))
    mean = mean/n/n
    delta_new = np.kron(delta,mean)
    delta_new = delta_new*mnn.relu_grad_fun(z)
    return delta_new

'''
max pooling layer delta
Kronecker product
c = np.kron(a,b)
'''
def max_pool_delta_fun(delta,n,z,max_sit):
    one = np.ones((n,n))
    delta_new = np.kron(delta,one)
    delta_new = delta_new*max_sit*mnn.relu_grad_fun(z)
    return delta_new
'''
use convolve2d to achieve forword convolve3d funcition
'''
def cov3d_for_fun(x,fil):
    z = np.zeros((x.shape[0],fil.shape[0],(x.shape[1]-fil.shape[1]+1),(x.shape[2]-fil.shape[2]+1)))
    for i in range(fil.shape[0]):
        for j in range(x.shape[0]):
            z[j][i] = signal.convolve2d(x[j],rot_fun(fil[i]),'valid')
    return z

'''
use convolve2d to achieve forword convolve4d funcition
'''
def cov4d_for_fun(x,fil):
    z = np.zeros((x.shape[0],fil.shape[0],(x.shape[2]-fil.shape[2]+1),(x.shape[3]-fil.shape[3]+1)))
    for i in range(fil.shape[0]):
        for j in range(x.shape[0]):
            for k in range(x.shape[1]):
                z[j][i] += signal.convolve2d(x[j][k],rot_fun(fil[i][k]),'valid')
    return z

'''
use convolve2d to achieve back convolve3d funcition
'''
def cov3d_back_fun(x,delta,shape):
    z = np.zeros((shape))
    for i in range(shape[0]):
        for k in range(x.shape[0]):
            z[i] += signal.convolve2d(x[k],rot_fun(delta[k][i]),'valid')
    return z

'''
use convolve2d to achieve back convolve4d funcition
'''
def cov4d_back_fun(x,delta,shape):
    z = np.zeros((shape))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(x.shape[0]):
                z[i][j] += signal.convolve2d(x[k][j],rot_fun(delta[k][i]),'valid')
    return z

'''
use convolve2d to achieve back delta of convolve
'''
def con_delta_fun(delta,fil,z):
    delta_new = np.zeros((z.shape)) 
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            for k in range(fil.shape[0]):
                delta_new[i][j] += signal.convolve2d(delta[i][k],fil[k][j],'full')
    delta_new = delta_new*mnn.relu_grad_fun(z)
    return delta_new

'''
max pooling function
'''
def max_pool_fun(x):
    z= skimage.measure.block_reduce(x, (1,1,2,2), np.max)
    z1 = np.ones((x.shape[0],x.shape[1],2,2))
    z_sit = np.kron(z,z1)
    x -= z_sit
    x[x==0]=1
    x[x<0]=0
    return z,x

'''
mean pooling function
'''
def mean_pool_fun(x):
    z= skimage.measure.block_reduce(x, (1,1,2,2), np.mean)
    return z

'''
flatten function 
'''
def flatten_fun(x):
    x=x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
    return x

'''
Dropout function
input x and dropout value
output after dropout
'''
def dropout_fun(x,value):
    z = np.random.random(x.shape)
    z[z>value]=1
    z[z<=value]=0
    x =x*z
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
def output_layer(x,filter1,filter2,w1,w2):
    z1 = cov3d_for_fun(x,filter1)
    x2 = mnn.relu_fun(z1)
    z2 = cov4d_for_fun(x2,filter2)
    x3 = mnn.relu_fun(z2)
    z3 = mean_pool_fun(x3)
    x4 = flatten_fun(z3)
    z4 = np.dot(x4,w1)
    x5 = mnn.relu_fun(z4)
    z5 = np.dot(x5,w2)
    return x2,x3,x4,x5,z1,z2,z3,z4,z5

'''
last recognition function
input x.shape(num,784)
input filter1.shape(filter1_size,3,3)
input filter1.shape(filter2_size,filter1_size,3,3)
input w1.shape(hide1_num,hide2_num)
input w2.shape(hide2_num,class_num)
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

'''
loss function and accurate function
'''
def loss_and_acc_fun(data,label,y_data,filter1,filter2,w1,w2,batch_size,loss_bool=0):
    inner_size = int(data.shape[0]/batch_size)
    loss = 0
    acc  = 0 
    for i in range(inner_size):
        batch_data  = data[i*batch_size:(i+1)*batch_size]
        batch_y     = y_data[i*batch_size:(i+1)*batch_size]
        x2,x3,x4,x5,z1,z2,z3,z4,z5,y = recognition_fun(batch_data,filter1,filter2,w1,w2)
        acc  += mnn.test_fun(batch_data,batch_y,z5)
        if loss_bool ==1:
            batch_label = label[i*batch_size:(i+1)*batch_size]
            loss += mnn.loss_fun(y,batch_label)
    loss = loss/inner_size
    acc  = acc/inner_size
    return acc ,loss

'''
training function
'''
def training_fun(data,label,y_data,filter1,filter2,w1,w2):
    inner_size = int(data.shape[0]/batch_size)
    for i in range(epochs):
        for j in range(inner_size):
            batch_data  = data[j*batch_size:(j+1)*batch_size]
            batch_label = label[j*batch_size:(j+1)*batch_size]
            x2,x3,x4,x5,z1,z2,z3,z4,z5,y = recognition_fun(batch_data,filter1,filter2,w1,w2)
            w2delta5 = mnn.delta3_fun(z5,y,batch_label)
            w1delta4 = mnn.delta_fun(w2delta5,w2,x5)
            delta3 = fla_delta(w1delta4,w1,z3.shape)
            f2delta2 = mean_pool_delta_fun(delta3,2,x3)
            f1delta1 = con_delta_fun(f2delta2,filter2,x2)
            dew2 = mnn.w_grad_fun(x5,w2delta5)
            dew1 = mnn.w_grad_fun(x4,w1delta4)
            def2 = cov4d_back_fun(x2,f2delta2,filter2.shape)
            def1 = cov3d_back_fun(batch_data,f1delta1,filter1.shape)
            w2 = w2 - hide2_rate/(((i*inner_size*2+1))**0.5)/batch_size*dew2
            w1 = w1 - hide1_rate/(((i*inner_size*2+1))**0.5)/batch_size*dew1
            filter2 = filter2 - filter2_rate/(((i*inner_size*2+1))**0.5)/batch_size*def2
            filter1 = filter1 - filter1_rate/(((i*inner_size*2+1))**0.5)/batch_size*def1
            print("epochs = %0d,batch cycle = %0d" %(i,j))
        acc,loss = loss_and_acc_fun(data,label,y_data,filter1,filter2,w1,w2,batch_size,1)
        print(i+1,loss,acc)
    return filter1,filter2,w1,w2

'''
main funciton
'''
def run():
    s_time = int(time.time())
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
    label = np.zeros((y_train.shape[0],class_num))
    for i in range(y_train.shape[0]):
        label[i,y_train[i]]=1
    
    #x_data = x_train[0:batch_size*200]
    #l_data = label[0:batch_size*200]
    #y_data = y_train[0:batch_size*200]
    x_data = x_train
    l_data = label
    y_data = y_train
    filter1,filter2,w1,w2 = training_fun(x_data,l_data,y_data,filter1,filter2,w1,w2)
    acc,loss = loss_and_acc_fun(x_test,label,y_test,filter1,filter2,w1,w2,100,0)
    print(acc)
    e_time = int(time.time())
    print("%02d:%02d:%02d" %((e_time-s_time)/3600,(e_time-s_time)%3600/60,(e_time-s_time)%60))


if __name__ =='__main__':
    run()
