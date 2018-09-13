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
    def fit(self, X, y):
        self.z = [0 for x in range(len(self.layers))]
        self.activations = [0 for x in range(len(self.layers)+1)]
        self.activations[0] = X.T
        self.y = y.T
        self.dz = [0 for x in range(len(self.layers))]
        self.dw = [0 for x in range(len(self.layers))]
        self.db = [0 for x in range(len(self.layers))]
        self.init_weights()
        self.init_bias()
        for _ in range(self.epochs):
            self.forward_prop()
            self.compute_error(self.y)
            self.back_prop(self.y)
