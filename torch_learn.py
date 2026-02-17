import numpy as np
# Author: Gabriel Alas
# PLEASE KEEP IN MIND the variable n attached to other variables in this code stand for the current layer of calculation

class Layer:
    def __init__(self, input, output):
        self.weight_n = np.random.randn(input, output) * np.sqrt(2/input) # using He Initialization which is np.random.randn(out, in) * sqrt(2/input)
        self.bias_n = np.zeros((1, output)) # this is the initialization for the bias

    def forward(self, x):
        self.input = x # just receive the input layer
        self.z_n = np.dot(x, self.weight_n) + self.bias_n # dot product/matrix product of the input/resultant activated vector/matrix by the weight matrix at the level n
        self.output = np.where(self.z_n > 0, self.z_n, self.z_n * 0.01) # ReLU activation function (this is a variant called leaky ReLU so that if a neuron results in a negative value, it still has a super small slope so in the backprop process, it still sends some "blame back if that makes sense lol) to get rid of negative values
        # ^^ this practically states if value from the prior resultant dot product is greater than 0 return the matrix value at row i column j, if less then return 0.01*matrix_val so there is some value passed back instead of none, this prevents slow learning from matrix values being "cancelled out"
        return self.output # return dot product/matrix product
    
    def backward(self, grad_before, lr):
        # lr = learning rate, grad_output = blame from previous levels
        grad_reLU = np.where(self.z_n > 0, 1, 0.01) # like self.z_n calculation in forward function, this also uses the leaky_relu ** reference function for explanation **
         # ^^ this practically states if value from the prior resultant dot product is greater than 0 return the gradient at row i column j, if less then return 0.01*gradient so there is some value passed back instead of none, this prevents slow learning from gradients being "cancelled out"
        grad_z = grad_before * grad_reLU # MULTIVARIABLE CHAIN RULE PRODUCT BEFORE THIS LAYER (BLAME of L-1 * curr_ReLU) this takes everything from the previous chain rule operations, and then multiplies it by the relu gradient which in most cases is just ReLU(z_n) with n being the number of the current layer

        grad_Wn = np.dot(self.input.T, grad_z) # JACOBIAN MATRIX UPDATE ->dot product of the input vector/matrix from previous level times the accumulated chain rule resultant

        grad_Bn = np.sum(grad_z, axis=0, keepdims=True) # BIAS UPDATE -> update bias, hint this will just be 1 times the everything prior in the chain rule!

        grad_input = np.dot(grad_z, self.weight_n.T) # SUPER IMPORTANT -> this is what will pass on to layer L-1 for the next calculation, this will become grad_before in the next or i guess previous layer

        self.weight_n -= lr * grad_Wn # same formula w_new = w_curr - lr*gradient
        self.bias_n -= lr * grad_Bn # same formula b_new = b_curr - lr * grad_b

        return grad_input

class Network:
    def __init__(self, layers): #initialize layers variable to determine how many layers (output + input + n hidden) will have 
        self.layers = layers

    def forward_calc(self, x):
        for layer in self.layers: # loop through all the layers 
            x = layer.forward(x) # call the forward function from the Layer class to therefore act reflexively upon itself
        return x 
    
    def back_prop(self, start_grad, lr):
        grad = start_grad # now we are starting at the very end backwards meaning we are going from output -> input, hence the multivariable chain rule/backpropagation
        
        for layer in reversed(self.layers): # reversed to go backwards in order, we are going -> start layer
            grad = layer.backward(grad, lr) # we call the backward_pass function from the Layer class in order to use the multivariable chain rule for backpropagation


X_train = np.random.randn(100, 5)
Y_train = np.full((100, 1), 42.0)

model = Network([
    Layer(5, 64),
    Layer(64, 64), 
    Layer(64, 1)
])

learning_rate = 0.01
runthroughs = 200


for runthrough in range(runthroughs):
    predictions = model.forward_calc(X_train) # perform the forward pass on each layer

    loss = np.mean((predictions - Y_train) ** 2) # MSE function as our loss function

    loss_grad = 2 * (predictions - Y_train) / X_train.shape[0] # NUM NEEDS TO BE DIVIDED BY MEAN so gradients dont exponentially grow, initial gradient to pass through backprop since we dont have an initial grad_before in teh function referenced way above in the Layer class

    model.back_prop(loss_grad, learning_rate) # this is the backprop loop this will update the weights and bias ** reference above for more plz **

    if runthrough % 10 == 0: # print every 10 runthroughs lol
        print(f"Runthrough {runthrough}, loss is {loss:.4f}, and the prediction rn is {np.mean(predictions):.2f}")
