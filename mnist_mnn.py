import numpy as np
from scipy import misc

'''
global value
'''
batch_size = 128
epochs = 20
rate =0.05
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


(x_train, y_train), (x_test, y_test) = load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train /= 255
x_test  /= 255
w =np.random.randn(784,num_classes)
w = w/10
print (w.shape)
