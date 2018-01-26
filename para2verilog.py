import numpy as np
import parameter

filter1_size =4
filter2_size =8
def floattobin(value):
    str_value = hex(np.float16(value).view('H'))[2:].zfill(4)
    return str_value

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

def cov_weight(fil,bias):
    print(fil.shape)
    string  = "    logic[%0d:0][%0d:0][%0d:0][15:0]filter;\n"%(fil.shape[0]-1,fil.shape[1]-1,fil.shape[2]*fil.shape[3]-1)
    string += '    logic[%0d:0][%0d:0][15:0]bias;\n'%(bias.shape[0]-1,fil.shape[1]-1)
    for i in range(fil.shape[0]):
        for j in range(fil.shape[1]):
            for k in range(fil.shape[2]):
                for m in range(fil.shape[3]):
                    string += "    assign filter[%0d][%0d][%0d] = 16'h%s;\n"%(i,j,k*fil.shape[3]+m,floattobin(fil[i][j][k][m]))
    for i in range(bias.shape[0]):
        for j in range(fil.shape[1]):
            if j==0:
                string += "    assign bias[%0d][%0d] = 16'h%s;\n" %(i,j,floattobin(bias[i]))
            else:
                string += "    assign bias[%0d][%0d] = 16'h%s;\n" %(i,j,"0000")                
    return string

def linear_weight(w,b):
    string  = '    logic[%0d:0][%0d:0][15:0]weight;\n'%(w.shape[0]-1,w.shape[1]-1)
    string += '    logic[%0d:0][15:0]bias;\n'%(b.shape[0]-1)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
             string += "    assign weight[%0d][%0d] = 16'h%s;\n" %(i,j,floattobin(w[i][j]))
    for i in range(b.shape[0]):
        string += "    assign bias[%0d] = 16'h%s;\n" % (i,floattobin(b[i]))
    return string

def par2str_linear(w,b):
    string  = '    shortreal weight[%0d:0][%0d:0];\n'%(w.shape[0]-1,w.shape[1]-1)
    string += '    shortreal bias[%0d:0];\n'%(b.shape[0]-1)
    for i in range(w.shape[0]) :
        for j in range (w.shape[1]):
            string += "    assign weight[%0d][%0d] = %f;\n"%(i,j,w[i][j])
    for i in range (b.shape[0]):
            string += "    assign bias[%0d] = %f;\n" % (i,b[i])
    return string

def write_sv():
    params = parameter.parameter
    filter1,fb1,filter2,fb2,w1,b1,w2,b2 = get_para(params)
    string = cov_weight(filter1,fb1) 
    with open('con1_param.sv','w') as write_object:
        write_object.write(string)     
    
    string = cov_weight(filter2,fb2)  
    with open('con2_param.sv','w') as write_object:
        write_object.write(string)    
    
    string = linear_weight(w1,b1) 
    with open('linear1_param.sv','w') as write_object:
        write_object.write(string)         
    
    string = linear_weight(w2,b2) 
    with open('linear2_param.sv','w') as write_object:
        write_object.write(string)

if __name__ =='__main__':
    write_sv()

    
