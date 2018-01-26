import time
import numpy as np
from scipy import misc
from scipy import signal
import mnist_mnn as mnn
import skimage.measure
import parameter

def floattobin(value, temp=0):
    str_value = hex(np.float16(value).view('H'))[2:].zfill(4)
    if temp==0:
        str_value += ";\n"
    else:
        str_value += "\n" 
    return str_value

def data2string(paras):
    string ='logic [%03d:0][%03d:0] x_data;\n'%(paras.shape[0]-1,paras.shape[1]-1)
    for i in range(paras.shape[0]):
        for j in range(paras.shape[1]):
            string += "assign x_data[%03d][%03d] = 16'h" %(i,j)
            string += floattobin(paras[i][j])
    return string


def data2hex(paras):
    for i in range(paras.shape[0]):
        string =""
        for j in range(paras.shape[1]):
            string += floattobin(paras[i][j],1)
        file_path ="mem%d.txt" %i
        with open(file_path,'w') as write_object:
            write_object.write(string)  


(x_train, y_train), (x_test, y_test) = mnn.load_data()
x_test  = x_test.astype('float16')
x_test = x_test/255
x_test = x_test.reshape(10000, 784)

print(x_test.shape)
print(y_test.shape)
data2hex(x_test)
i=0
string ='logic[9999:0][3:0] y_data;\n'
for y in y_test:
   string+="assign y_data[%03d] = 4'h%0d;\n" %(i,y)
   i+=1
with open('label.sv','w') as write_object:
    write_object.write(string)  


