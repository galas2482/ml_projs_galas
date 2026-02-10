import numpy as np

x = np.array([30,28,32,28,26]).reshape(-1, 1) # reshape makes it a column vector
W1 = np.random.randn(10, 5) * 0.01 # 
b1 = np.zeros((10, 1)) # bias all set to 0
W2 = np.random.randn(1, 10) * 0.01
b2 = np.zeros((1, 1))
actual_y = 42
learning_rate = 0.00001

for epoch in range(500):
    z1 = np.dot(W1, x) + b1 # this is if you took the 10 x 5 Weight matrix multiplied by the 5 x 1 input column vector
    a1 = np.maximum(0, z1) # ReLU activation function
    prediction = np.dot(W2, a1) + b2

    error = prediction - actual_y

    relu_grad = (z1 > 0).astype(float)
    delta_hidden = (np.dot(W2.T, error)) * relu_grad

    grad_W1 = np.dot(delta_hidden, x.T)
    grad_W2 = np.dot(error, a1.T)

    W1 -= learning_rate * grad_W1
    W2 -= learning_rate * grad_W2

    b1 -= learning_rate * delta_hidden
    b2 -= learning_rate * error 

    if epoch % 10 == 0:
       print(f"Runthrough {epoch}: Prediction: was {prediction[0][0]:.2f} | The Actual was: {actual_y:.2f} | So the model was {error[0][0]} off")
