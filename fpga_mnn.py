import numpy as np
from scipy import misc
import mnn_parameter

'''
three lay NN
two hide lay
hide 1 num = 128
hide 2 num = 64
hide 1 activate function relu
hide 2 activate function relu
relu = 0 when input <0
relu = input when input >=0
output lay activate function softmax
softmax = exp(zi)/sum(exp(zi))

loss funciton Cross Entropy
L = -sum(label*ln(y)+(1-label)*ln(1-y)
'''

'''
global value
'''
class_num = 10
hide1_num = 128
hide2_num = 64
'''
use numpy load mnist data from mnist.npz
'''
def load_data(path='mnist.npz'):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test) 

'''
test funciton input data and label and w
output Correct rate
'''
def test_fun(data,label,z):
    sum_num = label.shape[0]
    right_num = 0
    i=0
    for i in range (sum_num) :
        last_num = np.argmax(z[i])
        if last_num == label[i]:
            right_num +=1 
    return right_num/sum_num



'''
relu funciton
input X when Xij<0 output 0
    else output X
'''
def relu_fun(x):
    x[x<0]=0
    return x
    
    
'''
output layer function
input x.shape(num,784)
input w1.shape(784,hide1_num)
input w2.shape(hide1_num,hide2_num)
input w3.shape(hide2_num,class_num)
output z.shape(num,class_num)
'''

def output_layer(x,w1,w2,w3):
    z1 = np.dot(x,w1)
    x2 = relu_fun(z1)
    z2 = np.dot(x2,w2)
    x3 = relu_fun(z2)
    z3  = np.dot(x3,w3)
    return x2,x3,z1,z2,z3

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
def recognition_fun(x,w1,w2,w3):
    x2,x3,z1,z2,z3 = output_layer(x,w1,w2,w3)
    exp_z = np.exp(z3)
    exp_z =exp_z.T
    y = exp_z/(exp_z.sum(axis =0))
    y=y.T
    return (x2,x3,z1,z2,z3,y)

'''
loss function 
loss = -label*log(y)+(1-label)*log(1-y)
'''
def loss_fun(y,label):
    one = np.ones((label.shape))
    loss = -(label*np.log(y)+(one-label)*np.log(one-y))
    loss = loss.sum()/label.shape[0]
    return loss

'''
get paramater
'''
def get_para(params):
    new_paras =[]
    for param in params:
        if param :
            new_paras.append(param)
    arrs = np.array(new_paras)
    w1 = arrs[0][0] 
    w2 = arrs[1][0] 
    w3 = arrs[2][0]
    return w1,w2,w3 


'''
main function
'''
def run():
    (x_train, y_train), (x_test, y_test) = load_data()
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_test  /= 255
    w1 =np.random.randn(784,hide1_num)
    w2 =np.random.randn(hide1_num,hide2_num)
    w3 =np.random.randn(hide2_num,class_num)
    params = mnn_parameter.parameter
    w1,w2,w3 =get_para(params)  
    tax2,tax3,taz1,taz2,taz3,tay = recognition_fun(x_test,w1,w2,w3)
    tacc = test_fun(x_test,y_test,taz3)
    print(tacc)
if __name__ =='__main__':
    run()


