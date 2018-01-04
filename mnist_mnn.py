import numpy as np
from scipy import misc

'''
three lay NN
two hide lay
hide 1 num = 512
hide 2 num = 256
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
epochs = 20
rate1 =0.06
rate2 =0.035
rate3 =0.015
class_num = 10
hide1_num = 512
hide2_num = 256
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
calculate  delta 3
L = -sum(label*ln(y)+(1-label)*ln(1-y)

delta3 = (pi(L)/pi(y))*(pi(y)/pi(z))
pi(L)/pi(y) = -(label - y)/y*(1-y)
pi(y)/pi(z) = exp(zi)*(sum(exp(zi))-exp(zi))/(sum(exp(zi))^2)

input w1.shape(784,hide1_num)
input w2.shape(hide1_num,hide2_num)
input w3.shape(hide2_num,class_num)
input x.shape(num,784)
input label.shape(num,class_num)
z.shape(num,class_num)
sum_exp_z.shape(num,1)
y.shape(num,class_num)
grad_ly.shape(num,class_num)
'''
def delta3_fun(z3,y,label):
    num = y.shape[0]
    exp_z = np.exp(z3)
    exp_z = exp_z.T
    grad_yz = exp_z*(exp_z.sum(axis =0)-exp_z)/exp_z.sum(axis =0)/exp_z.sum(axis =0)
    sum_exp_z = exp_z.sum(axis=1)
    grad_yz =grad_yz.T 
    grad_ly = -(label-y)/y*(np.ones((num,class_num))-y)
    return grad_ly * grad_yz

'''
calculate delta not last layer
calculate delta(n)
input delta(n+1),w(n+1),z(n)
output delta(n)
'''
def delta_fun(delta,w,z):
    delta_new = (np.dot(delta,w.T)*relu_grad_fun(z))
    return delta_new

'''
gradient dew(n)
input w(n),delta(n)
output dew(n)
'''
def w_grad_fun(x,delta):
    dew =np.dot(x.T,delta) 
    return dew

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
triaining function
'''
def training_fun(data,label,w1,w2,w3,y_train):
    inner_size = int(data.shape[0]/batch_size)
    for i in range(epochs):
        for j in range(inner_size):
            batch_data  = data[j*batch_size:(j+1)*batch_size]
            batch_label = label[j*batch_size:(j+1)*batch_size]
            x2,x3,z1,z2,z3,y = recognition_fun(batch_data,w1,w2,w3)
            delta3 = delta3_fun(z3,y,batch_label)
            dew3 = w_grad_fun(x3,delta3)
            delta2 = delta_fun(delta3,w3,x3)
            dew2 = w_grad_fun(x2,delta2)
            delta1 = delta_fun(delta2,w2,x2)
            dew1 = w_grad_fun(batch_data,delta1)
            w3 = w3 - rate3/(((i*inner_size*2+1))**0.5)/batch_size*dew3
            w2 = w2 - rate2/(((i*inner_size*2+1))**0.5)/batch_size*dew2
            w1 = w1 - rate1/(((i*inner_size*2+1))**0.5)/batch_size*dew1
        ax2,ax3,az1,az2,az3,ay = recognition_fun(data,w1,w2,w3)
        loss = loss_fun(ay,label)
        acc = test_fun(data,y_train,az3)
        print(i+1,loss,acc)
    tax2,tax3,taz1,taz2,taz3,tay = recognition_fun(x_test,w1,w2,w3)
    tacc = test_fun(x_test,y_test,taz3)
    print(i+1,tacc)

'''
main function
'''
def run():
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train.reshape(60000, 784)
    y_train = y_train.reshape(60000, 1)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255
    w1 =np.random.randn(784,hide1_num)
    w2 =np.random.randn(hide1_num,hide2_num)
    w3 =np.random.randn(hide2_num,class_num)
    w1 = w1/8
    w2 = w2/8
    w3 = w3/8
    label = np.zeros((y_train.shape[0],class_num))
    for i in range(y_train.shape[0]):
        label[i,y_train[i]]=1
    print(label.shape)
    training_fun(x_train,label,w1,w2,w3,y_train)

if __name__ =='__main__':
    run()

