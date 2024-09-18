from math import exp, log
import random
import math
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np

class Initializer:
    def init_weights(self, n_in, n_out):
        raise NotImplementedError

    def init_bias(self, n_out):
        raise NotImplementedError

class NormalInitializer(Initializer):
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def init_weights(self, n_in, n_out):
        return [[Var(random.gauss(self.mean, self.std)) for _ in range(n_out)] for _ in range(n_in)]

    def init_bias(self, n_out):
        return [Var(0.0) for _ in range(n_out)]

class ConstantInitializer(Initializer):
    def __init__(self, weight=1.0, bias=0.0):
        self.weight = weight
        self.bias = bias

    def init_weights(self, n_in, n_out):
        return [[Var(self.weight) for _ in range(n_out)] for _ in range(n_in)]

    def init_bias(self, n_out):
        return [Var(self.bias) for _ in range(n_out)]
      
class Var:
    """
    A variable which holds a float and enables gradient computations.
    """

    def __init__(self, val: float, grad_fn=lambda: []):
        assert type(val) == float
        self.v = val
        self.grad_fn = grad_fn
        self.grad = 0.0

    def backprop(self, bp):
        self.grad += bp
        for input, grad in self.grad_fn():
            input.backprop(grad * bp)

    def backward(self):
        self.backprop(1.0)

    def __add__(self: 'Var', other: 'Var') -> 'Var':
        return Var(self.v + other.v, lambda: [(self, 1.0), (other, 1.0)])

    def __mul__(self: 'Var', other: 'Var') -> 'Var':
        return Var(self.v * other.v, lambda: [(self, other.v), (other, self.v)])

    def __pow__(self, power):
        assert type(power) in {float, int}, "power must be float or int"
        return Var(self.v ** power, lambda: [(self, power * self.v ** (power - 1))])

    def __neg__(self: 'Var') -> 'VarS':
        return Var(-1.0) * self

    def __sub__(self: 'Var', other: 'Var') -> 'Var':
        return self + (-other)

    def __truediv__(self: 'Var', other: 'Var') -> 'Var':
        return self * other ** -1

    def __repr__(self):
        return "Var(v=%.4f, grad=%.4f)" % (self.v, self.grad)

    def relu(self):
        return Var(self.v if self.v > 0.0 else 0.0, lambda: [(self, 1.0 if self.v > 0.0 else 0.0)])

    def identity(self):
        return Var(self.v, lambda: [(self, 1)])

    def tanh(self):
        v = math.tanh(self.v)
        grad = 1-v*v
        return Var(v, lambda: [(self, grad)])

    def sigmoid(self):
        v = 1 / (1 + math.exp(-self.v))
        grad = v * (1.0 - v)
        return Var(v, lambda: [(self, grad)])

def nparray_to_Var(x):
    if x.ndim==1:
        y = [[Var(float(x[i]))] for i in range(x.shape[0])] # always work with list of list
    else:
        y = [[Var(float(x[i,j])) for j in range(x.shape[1])] for i in range(x.shape[0])]
    return y
    
def Var_to_nparray(x):
    y = np.zeros((len(x),len(x[0])))
    for i in range(len(x)):
        for j in range(len(x[0])):
            y[i,j] = x[i][j].v
    return y

def squared_loss(t, y):
  
    # add check that sizes agree
    assert len(t) == len(y), "t and y must have same size"
  
    def squared_loss_single(t, y):
        Loss = Var(0.0)
        for i in range(len(t)): # sum over outputs
            Loss += (t[i]-y[i]) ** 2
        return Loss

    Loss = Var(0.0)
    for n in range(len(t)): # sum over training data
        Loss += squared_loss_single(t[n],y[n])
    return Loss

def parameters(network):
    params = []
    for layer in range(len(network)):
        params += network[layer].parameters()
    return params

def update_parameters(params, learning_rate=0.01):
    for p in params:
        p.v -= learning_rate*p.grad

def zero_gradients(params):
    for p in params:
        p.grad = 0.0

## Glorot
def DenseLayer_Glorot_tanh(n_in: int, n_out: int):
    std = (2*1/(n_in+n_out))**(1/2)  # <- replace with proper initialization
    return DenseLayer(n_in, n_out, lambda x: x.tanh(), initializer = NormalInitializer(std))

## He
def DenseLayer_He_relu(n_in: int, n_out: int):
    std = (2/n_in)**(1/2) # <- replace with proper initialization
    return DenseLayer(n_in, n_out, lambda x: x.relu(), initializer = NormalInitializer(std))
    
class DenseLayer:
    def __init__(self, n_in: int, n_out: int, act_fn, initializer = NormalInitializer(),norm=True):
        self.weights = initializer.init_weights(n_in, n_out)
        self.bias = initializer.init_bias(n_out)
        self.norm = norm
        self.norm_layer = NormLayer()
        self.act_fn = act_fn
    
    def __repr__(self):    
        return 'Weights: ' + repr(self.weights) + ' Biases: ' + repr(self.bias)

    def parameters(self) -> Sequence[Var]:
        params = []
        for r in self.weights:
            params += r
        params += self.bias
        if self.norm:
            params_norm = self.norm_layer.parameters()
            params += params_norm
        return params

    def forward(self, single_input: Sequence[Var]) -> Sequence[Var]:
        assert len(self.weights) == len(single_input), "weights and single_input must match in first dimension"
        weights = self.weights
        out = []
        
        for j in range(len(weights[0])): 
            node = self.bias[j]# <- Insert code
            for i in range(len(single_input)):
                node += single_input[i] * weights[i][j]
            out.append(node)
        if self.norm:
            out = self.norm_layer.forward(out)
        
        for i in range(len(out)):
            out[i] = self.act_fn(out[i])
        return out

def forward(input, network):
    def forward_single(x, network):
        for layer in network:
            x = layer.forward(x)
        return x
    output = [forward_single(input[n], network) for n in range(len(input))]
    return output

class NormLayer:
    def __init__(self):
        self.gamma = Var(1.0)
        self.beta = Var(0.0)
        self.eps = 1e-5 

    def parameters(self):
        return [self.gamma,self.beta]

    def forward(self, x_input):
        mean = Var(sum(x.v for x in x_input) / len(x_input))
        variance = Var(sum((x.v - mean.v) ** 2 for x in x_input) / len(x_input))
        
        normalized = [(x - mean) / (variance + Var(self.eps))**0.5 for x in x_input]
        output = [self.gamma * n + self.beta for n in normalized]
        return output
        
def fitting_test(EPOCHS = 100, LEARN_R_List = [[1000,1e-3]],train_batch_data=[],validation_data=[[],[]],test_data=[],need_print=True):
    NN = [
        DenseLayer_Glorot_tanh(1, 64),
        DenseLayer(64, 32, lambda x: x.tanh(),norm=True),
        DenseLayer(32, 8, lambda x: x.tanh(),norm=True),
        DenseLayer(8, 1, lambda x: x.identity(),norm=False)
    ]

    x_validation,y_validation = validation_data
    x_validation = nparray_to_Var(x_validation)
    y_validation = nparray_to_Var(y_validation)
    
    train_loss = []
    val_loss = []
    log_list = []

    for e in range(EPOCHS):
        
        e_end = 0
        for e_num,LEARN_R in LEARN_R_List:
            if e in range(e_end,e_num):
                break

        batch_sequence = random.sample(range(0,len(train_batch_data)), len(train_batch_data))
        batch_loss = 0
        for b_i in batch_sequence:
            batch_data = train_batch_data[b_i]
            x_train,y_train = batch_data
            batch_size = len(x_train)
            x_train = nparray_to_Var(x_train)
            y_train = nparray_to_Var(y_train)
            
            Loss = squared_loss(y_train, forward(x_train, NN))/Var(float(batch_size))
            Loss.backward()
            update_parameters(parameters(NN), LEARN_R)
            zero_gradients(parameters(NN))
            batch_loss+=Loss.v
    
        train_loss.append(batch_loss/len(batch_sequence))
    
        Loss_validation = squared_loss(y_validation, forward(x_validation, NN))/Var(float(len(x_validation)))
        val_loss.append(Loss_validation.v)
    
        if (e+1)%(EPOCHS/10)==0 or e==0:
            log = "{:4d}\t".format(e+1)+"({:5.2f}%)\t".format((e+1)/EPOCHS*100)+"Train loss: {:4.3f} \t Validation loss: {:4.3f}".format(train_loss[-1], val_loss[-1])
            log_list.append(log)
            if need_print:
                print(log)

    if need_print:
        plt.plot(range(len(train_loss)), train_loss, label="train_loss");
        plt.plot(range(len(val_loss)), val_loss, label="validation_loss");
        plt.legend()
        plt.show()

    print_info = [log_list,train_loss,val_loss]
    return Var_to_nparray(forward(nparray_to_Var(test_data), NN)),None,print_info

def muti_thread_func(result_queue,epoch_num,lr,train_batch_set,validation_data,x_test):
    test_output = fitting_test(epoch_num,lr,train_batch_set,validation_data,x_test,False)

    result_queue.put([lr,test_output])

if __name__ == '__main__':
    print("muti_func")