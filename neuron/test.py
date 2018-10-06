import neuron
import numpy as np

# And gate NN
X = np.matrix([[1,1],[1,0],[0,1],[0,0],[1,1],[0,1],[1,1]]) #Train set
y = np.matrix([1, 0, 0, 0, 1, 0, 1]).T#Train set
test_x = np.matrix([[0,0],[0,0],[1,1],[0,1]]) #Test set
#Adam optimizer
op = neuron.Optimizers.Adam(learning_rate=0.1)
#Defining model
model = neuron.Neuron(input_neurons=2,loss='binary_crossentropy',optimizer=op,epochs=50)
model.add(neuron.Layer(5,activation='tanh'))
model.add(neuron.Layer(2,activation='tanh'))
model.add(neuron.Layer(1,activation='sigmoid'))
#Fitting the model
model.fit(X,y,val_x=X,val_y=y,batch_size=1,print_stats=True)
#Predicting
print(model.predict(test_x))
#Saving model
model.save_parameters('filename')
#Loading pretrained model and predicting
model.load_parameters('filename or path+filename')
print(model.predict(test_x))
