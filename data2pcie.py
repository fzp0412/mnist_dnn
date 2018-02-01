import numpy as np
def load_data(path='mnist.npz'):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)



def floattobin(value):
    str_value = hex(np.float16(value).view('H'))[2:].zfill(4)
    return str_value


def run():
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train.reshape(60000, 784)
    y_train = y_train.reshape(60000, 1)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255
    string =''
    #for i in range(x_test.shape[0]):
    for i in range(10):
        for j in range(x_test.shape[1]):
            string += floattobin(x_test[i][j])
    with open('x_test.txt','w') as write_object:
        write_object.write(string)  
    
if __name__ =='__main__':
    run()

