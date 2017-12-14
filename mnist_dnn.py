import numpy as np
from scipy import misc

'''
global value
'''
batch_size = 128
epochs = 20
rate =0.001
num_classes = 10

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
calculate two vector inner product 
'''
def exp_fun(w,x):
    value = np.exp(np.dot(w,x))
    return value
'''
input initial w1 w2 w3 ...., output w1 grad,w2 grad w3 grad ....
w.shape = (10,28*28)
x.shape = (28*28,)
return grad_w.shape = (10,28*28)
only one photo
all photo must sum all gradient
'''
def grad_fun(w,x):
    grad_w = np.zeros(w.shape)
    exp_value = []
    sum_exp =0
    for i in range(w.shape[0]):
        exp_value.append(exp_fun(w[i],x))
        sum_exp+=exp_value[i]
    for i in range(w.shape[0]):
        grad_w[i] =-(sum_exp -exp_value[i])/sum_exp * x
    return grad_w

'''
all photo gradient
'''
def all_grad_fun(w,data):
    sum_grad = np.zeros(w.shape)
    for x in data :
        sum_grad += grad_fun(w,x)
    return sum_grad

'''
calculate Multi-class Classification Cross Entropy
only one photo
all loss function must sum all value
'''
def loss_fun(w,x,label):
    exp_value = []
    sum_exp =0
    for i in range(w.shape[0]):
        exp_value.append(exp_fun(w[i],x))
        sum_exp+=exp_value[i]
    loss_value = - np.log(exp_value[label]/sum_exp)
    return loss_value

'''
all loss function
'''
def all_loss_fun(w,data,label):
    sum_loss = 0
    i = 0
    for x in data:
        sum_loss += loss_fun(w,x,label[i])
        i+=1
    return sum_loss

'''
training function input x data
'''
def training_fun(data,label,w):
    loss_value = all_loss_fun(w,data,label)
    print(loss_value)
    for i in range(epochs):
        for j in range(int(data.shape[0]/batch_size)):
            batch_data  = data[j*batch_size:(j+1)*batch_size]
            batch_label = label[j*batch_size:(j+1)*batch_size]
            grad_w = all_grad_fun(w,batch_data)
            #print(grad_w)
            w_new = w - rate/((j+1)**0.5)* grad_w
        loss_value = all_loss_fun(w_new,data,label)
        print(loss_value)
    return w_new

(x_train, y_train), (x_test, y_test) = load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train /= 255
x_test  /= 255

print(x_train.shape)
print(y_train.shape)
w =np.random.randn(num_classes,784)
w_new = training_fun(x_train,y_train,w)

