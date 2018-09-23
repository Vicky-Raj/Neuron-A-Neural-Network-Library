"""
@author: vignesh
"""
import numpy as np 
#Layer class for adding layers and activations
class Layer():
    def __init__(self, neurons, activation):
        self.neurons = neurons
        self.activation = activation

class Activations():
#Activations for layers
    @staticmethod
    def sigmoid(z, deriv=False):
        if deriv:
            a = Activations.sigmoid(z)
            return np.multiply(a, (1-a))
        return ( 1 / (1 + np.exp(-z)))

    @staticmethod
    def relu(z, deriv=False):
        z = np.asarray(z)
        if deriv:
            z = 1. * (z >= 0)
            return (np.asmatrix(z))
        z = z * (z > 0)
        z += 0
        return(np.asmatrix(z))
    
    @staticmethod
    def tanh(z, deriv=False):
        if deriv:
            a = Activations.tanh(z)
            return (1 - np.power(a,2))
        return ((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)))

class Losses():
#Different losses
    @staticmethod
    def sse(y,y_hat,deriv=False):
        if deriv:
            return (y_hat - y)
        difference = y - y_hat
        squared_diff = np.power(difference, 2)
        sum_1 = squared_diff.sum(axis=1)
        error = (1 / y.shape[1]) * sum_1
        return error
    
    @staticmethod
    def binary_crossentropy(y,y_hat,deriv=False):
        if deriv:
            return ((y_hat - y)/ (np.multiply(y_hat,1-y_hat)))
        temp1 = np.multiply(-y,np.log(y_hat))
        temp2 = np.multiply((1-y),np.log(1-y_hat))
        cost = np.sum(temp1 - temp2)
        return cost

class Optimizers():
#Different optimizers
    class GradientDescent():
        def __init__(self,learning_rate=0.001):
            self.learning_rate = learning_rate

        def step(self,weights,bias,dw,db):
            for i in range(len(weights)):
                weights[i] -= (self.learning_rate * dw[i])
                bias[i] -= (self.learning_rate * db[i])
            return weights, bias
    class Adam():
        def __init__(self,learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=0.00000001):
            self.learning_rate = learning_rate
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            self.epsilon = epsilon
            self.t = 1
            self.vdw = []
            self.sdw = []
            self.vdb = []
            self.sdb = []
            self.vdw_cor = []
            self.sdw_cor = []
            self.vdb_cor = []
            self.sdb_cor = []
        def step(self,weights,bias,dw,db):
            if self.t == 1:
                self.vdw = [np.asmatrix(np.zeros(w.shape)) for w in dw]
                self.vdb = [np.asmatrix(np.zeros(b.shape)) for b in db]
                self.sdw = [np.asmatrix(np.zeros(w.shape)) for w in dw]
                self.sdb = [np.asmatrix(np.zeros(b.shape)) for b in db]
                self.vdw_cor = [0 for _ in range(len(dw))]
                self.vdb_cor = [0 for _ in range(len(db))]
                self.sdw_cor = [0 for _ in range(len(dw))]
                self.sdb_cor = [0 for _ in range(len(db))]
            for i in range(len(weights)):
                self.vdw[i] = self.beta_1 * self.vdw[i] + (1-self.beta_1) * dw[i]
                self.vdb[i] = self.beta_1 * self.vdb[i] + (1-self.beta_1) * db[i]
                self.sdw[i] = self.beta_2 * self.sdw[i] + (1-self.beta_2) * np.power(dw[i],2)
                self.sdb[i] = self.beta_2 * self.sdb[i] + (1-self.beta_2) * np.power(db[i],2)
                self.vdw_cor[i] = self.vdw[i] / (1-np.power(self.beta_1,self.t))
                self.vdb_cor[i] = self.vdb[i] / (1-np.power(self.beta_1,self.t))
                self.sdw_cor[i] = self.sdw[i] / (1-np.power(self.beta_2,self.t))
                self.sdb_cor[i] = self.sdb[i] / (1-np.power(self.beta_2,self.t))
                weights[i] -= self.learning_rate * (self.vdw_cor[i] / (np.sqrt(self.sdw_cor[i])+self.epsilon))
                bias[i] -= self.learning_rate * (self.vdb_cor[i] / (np.sqrt(self.sdb_cor[i])+self.epsilon))
            self.t += 1
            return weights,bias
    class RMSprop():
        def __init__(self,learning_rate=0.001,beta=0.999,epsilon=0.00000001):
            self.learning_rate = learning_rate
            self.beta = beta
            self.epsilon = epsilon
            self.t = 1
            self.sdw = []
            self.sdb = []
            self.sdw_cor = []
            self.sdb_cor = []
        def step(self,weights,bias,dw,db):
            if self.t == 1:
                self.sdw = [np.asmatrix(np.zeros(w.shape)) for w in dw]
                self.sdb = [np.asmatrix(np.zeros(b.shape)) for b in db]
                self.sdw_cor = [0 for _ in range(len(dw))]
                self.sdb_cor = [0 for _ in range(len(db))]
            for i in range(len(weights)):
                self.sdw[i] = self.beta * self.sdw[i] + (1-self.beta) * np.power(dw[i],2)
                self.sdb[i] = self.beta * self.sdb[i] + (1-self.beta) * np.power(db[i],2)
                self.sdw_cor[i] = self.sdw[i] / (1-np.power(self.beta,self.t))
                self.sdb_cor[i] = self.sdb[i] / (1-np.power(self.beta,self.t))
                weights[i] -= self.learning_rate * (dw[i] / (np.sqrt(self.sdw_cor[i])+self.epsilon))
                bias[i] -= self.learning_rate * (db[i]/(np.sqrt(self.sdb_cor[i])+self.epsilon))
            self.t += 1
            return weights,bias

class Neuron(Activations,Losses):
    #Initializing variables
    def __init__(self, input_neurons, loss, optimizer, epochs):
        #Forwardprop lists
        self.input_neurons = input_neurons
        self.loss = loss
        self.optimizer = optimizer
        self.layers = []
        self.weights = []
        self.bias = []
        self.z = []
        self.activations = []
        self.epochs = epochs
        #Backprop lists
        self.dz = []
        self.dw = []
        self.db = []
        #Activations list
        self.act_dict = {'sigmoid':super().sigmoid, 'relu':super().relu, 'tanh':super().tanh}
        self.loss_dict = {'sse':super().sse,'binary_crossentropy':super().binary_crossentropy}
    
    def split_data(self,X,batch_size):
        self.start = 0
        self.end = batch_size
        self.splited_data = []
        while True:
            if self.end < X.shape[1]:
                self.splited_data.append(X[:,self.start:self.end])
                self.start = self.end
                self.end += batch_size
            else:
                self.splited_data.append(X[:,self.start:X.shape[1]+1])
                break
        return self.splited_data
    #Computing the error of the model using the chosen error fucntion
    def compute_error(self,y):
        self.error = self.loss_dict.get(self.loss)(y,self.activations[-1])
        print('error: {}'.format(self.error))
    #Computing the z term for applying activation
    def compute_z(self,weights, activations, bias):
        z = np.dot(weights,activations) + bias
        return z
    #back propogation to compute the gradients 
    def back_prop(self,y):
        self.avg = (1 / y.shape[1])
        self.temp_1 = self.loss_dict.get(self.loss)(y,self.activations[-1],deriv=True)
        self.temp_2 = self.act_dict.get(self.layers[-1].activation)(self.z[-1],deriv=True)
        self.dz[-1] = np.multiply(self.temp_1,self.temp_2)
        self.dz[-1] += 0
        self.dw[-1] = self.avg * np.dot(self.dz[-1],self.activations[-2].T)
        self.db[-1] = self.avg * np.sum(self.dz[-1],axis=1)
        for i in range(len(self.layers)-2,-1,-1):
            self.temp_1 = np.dot(self.weights[i+1].T,self.dz[i+1])
            self.temp_2 = self.act_dict.get(self.layers[i].activation)(self.z[i],deriv=True) 
            self.dz[i] = np.multiply(self.temp_1,self.temp_2)
            self.dz[i] += 0 
            self.dw[i] = self.avg * np.dot(self.dz[i],self.activations[i].T)
            self.db[i] = self.avg * np.sum(self.dz[i],axis=1)
        self.weights,self.bias =  self.optimizer.step(self.weights,self.bias,self.dw,self.db)

    #Forward propogating
    def forward_prop(self):
        for i in range(len(self.layers)):
            self.z[i] = self.compute_z(self.weights[i],self.activations[i],self.bias[i])
            self.activations[i+1] = self.act_dict.get(self.layers[i].activation)(self.z[i])

    #Initializing weights
    def init_weights(self):
        self.weights.append(np.random.randn(self.layers[0].neurons,self.input_neurons))
        for i in range(1, len(self.layers)):
            self.weights.append(np.random.randn(self.layers[i].neurons,self.layers[i-1].neurons))

    #Initializing bias
    def init_bias(self):
        for i in range(len(self.layers)):
            self.bias.append(np.zeros((self.layers[i].neurons, 1)))

    #Adding layers to the model
    def add(self, layer):
        self.layers.append(layer)

    #Predicting after training
    def predict(self, test_x):
        self.test_weights = self.weights
        self.test_bias = self.bias
        self.test_z = [0 for x in range(len(self.layers))]
        self.test_activations = [0 for x in range(len(self.layers)+1)]
        self.test_activations[0] = test_x.T
        for i in range(len(self.layers)):
            self.test_z[i] = self.compute_z(self.test_weights[i],self.test_activations[i],self.test_bias[i])
            self.test_activations[i+1] = self.act_dict.get(self.layers[i].activation)(self.test_z[i])
        print(self.test_activations[-1])

    #Fitting the neural network on training set
    def fit(self,X,y,batch_size,print_error=False):
        self.z = [0 for x in range(len(self.layers))]
        self.activations = [0 for x in range(len(self.layers)+1)]
        self.dz = [0 for x in range(len(self.layers))]
        self.dw = [0 for x in range(len(self.layers))]
        self.db = [0 for x in range(len(self.layers))]
        self.init_weights()
        self.init_bias()
        if not batch_size is None and batch_size < X.T.shape[1] and batch_size > 0: 
            self.X = self.split_data(X.T,batch_size)
            self.y = self.split_data(y.T,batch_size)
            for _ in range(self.epochs):
                for x_temp,y_temp in zip(self.X,self.y):
                    self.activations[0] = x_temp 
                    self.forward_prop()
                    self.back_prop(y_temp)
                if print_error:
                    self.activations[0] = X.T
                    self.forward_prop()
                    self.compute_error(y.T)
        else:
            self.activations[0] = X.T
            self.y = y.T 
            for _ in range(self.epochs):
                self.forward_prop()
                self.compute_error(self.y)
                self.back_prop(self.y)
