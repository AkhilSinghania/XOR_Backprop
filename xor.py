
import sys
import numpy as np

def sigmoid(x):
	#Activation Function used in forward propagation
	return 1/(1+exp(-x))
	
def sigmoidDash(x):
	#Derivative of sigmoid function
	#Activation function used in back propagation
	return x*(1-x)
	
#Given Data
x = np.array(([0,0],[0,1],[1,0],[1,1])) 						#4x2 matrix

#Actual Output (The Output expected by the result of our neural network) 
y = np.array(([0],[1],[1],[1])) 							#4x1 matrix

#Command for generating the same random numbers every time
#Makes it easy for Debugging
np.random.seed(1)

#Intializing random synapse weights
W1 = np.random.randn(2,4) 								#2x4 matrix
W2 = np.random.randn(4,1) 								#4x1 vector

for i in xrange(500000):
	
	#Forward propagation
	layer1 = x 									#input layer
	layer2 = sigmoid(np.dot(layer1,W1))	 					#4x4 matrix, Hidden layer
	layer3 = sigmoid(np.dot(layer2,W2)) 						#4x1 vector, Output layer
	
	#^In Forward propgation we first multiply the
	#values of each node with weights of the synapses
	#and then use the activation function to get the
	#value for the node in next layer
	
	#Calculating Error
	Layer3_error = y - layer3 							#4x1 vector
	
	#Backward propagation
	layer3_Delta = layer3_error*sigmoidDash(layer3) 				#4x1 vector
	layer2_error = layer3_Delta.dot(W2.T) 						#4x4 matrix
	layer2_Delta = layer2_error*sigmoidDash(layer2) 				#4x4 matrix
	
	#^In Backward propgation we first use the derivative
	#(Derivative - slope of the Activation Function)
	#of activation function and then multiply the error
	#of that particular layer to get a value Delta for 
	#that particular layer. This Delta value is then 
	#multiplied with the weight of the synapses to get 
	#the error in the previous layer. This goes till the
	#second layer as there is no error in the input layer.
	
	#Performing Gradient Descent To change the weights accordingly
	W2 += layer2.T.dot(layer3_Delta) 						#4x1 vector
	W1 += layer1.T.dot(layer2_Delta) 						#2x4 matrix
	
#Printing the Output
print "Output:"
print layer3
