
'''
input initial w1 w2 w3 ...., output w1 grad,w2 grad w3 grad ....
w.share = (10,28*28)
x.share = (28*28,)
return grad_w.share = (10,28*28)
'''
def grad(w,x):
    grad_w[:] = w[:]
    exp_value = []
    sum_exp =0
    for i in range(w.share[0]):
        exp_value.append(exp_fun(w[i],x))
        sum_exp+=exp_fun[i]
    for i in range(w.share[0]):
        grad_w[i] =(sum_exp -exp_fun[i])/sum_exp * x
    return grad_w
    
''' 
calculate two vector inner product 
'''
def exp_fun(w,x):
    value = np.exp(np.dot(w,x))
    return value

'''
calculate Multi-class Classification Cross Entropy
'''
def loss_fun(w,x,label):
    exp_value = []
    sum_exp =0
    for i in range(w.share[0]):
        exp_value.append(exp_fun(w[i],x))
        sum_exp+=exp_fun[i]
    loss_value = np.log(exp_value[label]/sum_exp)
    return loss_value

