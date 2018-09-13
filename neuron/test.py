import neuron
import numpy as np

# And gate
X = np.matrix([[1,1],[1,0],[0,1],[0,0]]) #Train set
y = np.matrix([1, 0, 0, 0]).T#Train set
test_x = np.matrix([[0,0],[0,0],[1,1],[0,1]]) #Test set
#Defining model
model = neuron.Neuron(input_neurons=2,loss='binary_crossentropy',optimizer=neuron.Optimizers.GradientDescent(0.1),epochs=5000)
model.add(neuron.Layer(5,activation='tanh'))
model.add(neuron.Layer(2,activation='tanh'))
model.add(neuron.Layer(1,activation='sigmoid'))
#Fitting the model
model.fit(X,y)
#Predicting
model.predict(test_x)