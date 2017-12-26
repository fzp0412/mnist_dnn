import numpy as np
from scipy import misc

'''
three lay NN
two hide lay
hide 1 num = 512
hide 2 num = 512
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
batch_size = 128
epochs = 1
rate1 =0.05
rate2 =0.05
rate3 =0.05
num_classes = 10
hide1_num = 512
hide2_num = 512
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
def test_fun(data,label,w):
    sum_num = label.shape[0]
    right_num = 0
    i=0
    for value in data :
        y = np.dot(w,value)
        last_num = np.argmax(y)
        if last_num == label[i]:
            right_num +=1 
        i+=1
    print(right_num/sum_num)
'''
calculate  delta 3
L = -sum(label*ln(y)+(1-label)*ln(1-y)

delta3 = (pi(L)/pi(y))*(pi(y)/pi(z))
pi(L)/pi(y) = (label - y)/y*(1-y)
pi(y)/pi(z) = exp(zi)*(sum(exp(zi))-exp(zi))/(sum(exp(zi))^2)

input w1.shape(784,512)
input w2.shape(512,512)
input w3.shape(512,10)
input x.shape(num,784)
input label.shape(num,10)
z.shape(num,10)
sum_exp_z.shape(num,10)
y.shape(num,10)
grad_ly.shape(num,10)
'''
def delta3_fun(z3,y,label):
    num = y.shape[0]
    exp_z = np.exp(z3)
    sum_exp_z = exp_z.sum(axis=1)
    grad_yz = np.ones(z3.shape)
    for i in range (z3.shape[1]):
        grad_yz[:,i] = exp_z[:,i]*(sum_exp_z - exp_z[:,i])/sum_exp_z/sum_exp_z
    grad_ly = (label-y)/y*(np.ones((num,num_classes))-y)
    return grad_yz * grad_yz


'''
calculate  delta 2
'''
def delta2_fun():
    return 0

'''
calculate  delta 1
'''
def delta1_fun():
    return 0

'''
gradient delta 3
'''
def delta3_grad_fun(x3,delta3):
    return 0


'''
relu funciton
input X when Xij<0 output 0
    else output X
'''
def relu_fun(x):
    x[x<0]=0
    return x
    
'''
relu grad_funciton
input X when Xij<0 output 0
    else output 1
'''
def relu_grad_fun(x):
    x[x>0]=1
    return x
    
'''
output layer function
input x.shape(num,784)
input w1.shape(784,512)
input w2.shape(512,512)
input w3.shape(512,10)
output z.shape(num,10)
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
input w1.shape(784,512)
input w2.shape(512,512)
input w3.shape(512,10)
output z.shape(num,10)
output sum_exp_z.shape(num,1)
output y.shape(num,10)
'''
def recognition_fun(x,w1,w2,w3):
    x2,x3,z1,z2,z3 = output_layer(x,w1,w2,w3)
    exp_z = np.exp(z3)
    sum_exp_z =exp_z.sum(axis=1)
    y = np.ones(z3.shape)
    for i in range (z3.shape[1]):
        y[:,i] = exp_z[:,i]/sum_exp_z
    return (x2,x3,z1,z2,z3,y)


def training_fun(data,label,w1,w2,w3):
    inner_size = int(data.shape[0]/batch_size)
    for i in range(epochs):
        for j in range(inner_size):
            batch_data  = data[j*batch_size:(j+1)*batch_size]
            batch_label = label[j*batch_size:(j+1)*batch_size]
            x2,x3,z1,z2,z3,y = recognition_fun(batch_data,w1,w2,w3)
            delta3 = delta3_fun(z3,y,batch_label)
            print(delta3.shape)
    
(x_train, y_train), (x_test, y_test) = load_data()
x_train = x_train.reshape(60000, 784)
y_train = y_train.reshape(60000, 1)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train /= 255
x_test  /= 255
w1 =np.random.randn(784,512)
w2 =np.random.randn(512,512)
w3 =np.random.randn(512,10)
w1 = w1/10
w2 = w2/10
w3 = w3/10
#label = np.zeros(y_train.shape[0],num_classes)
label = np.zeros((y_train.shape[0],num_classes))
for i in range(y_train.shape[0]):
    label[i,y_train[i]]=1
print(label.shape)

training_fun(x_train,label,w1,w2,w3)
