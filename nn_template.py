import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# Returns the ReLU value of the input x
def relu(x):
    return max(0, x)

# Returns the derivative of the ReLU value of the input x
def relu_derivative(x):
    return (x>0).astype(int)

## TODO 1a: Return the sigmoid value of the input x
def sigmoid(x):
    return 1/(1+np.exp(-x))

## TODO 1b: Return the derivative of the sigmoid value of the input x
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s*(1-s)

## TODO 1c: Return the derivative of the tanh value of the input x
def tanh(x):
    return np.tanh(x)

## TODO 1d: Return the derivative of the tanh value of the input x
def tanh_derivative(x):
    return 1-np.square(tanh(x))

# Mapping from string to function
str_to_func = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative)
}

# Given a list of activation functions, the following function returns
# the corresponding list of activation functions and their derivatives
def get_activation_functions(activations):  
    activation_funcs, activation_derivatives = [], []
    for activation in activations:
        activation_func, activation_derivative = str_to_func[activation]
        activation_funcs.append(activation_func)
        activation_derivatives.append(activation_derivative)
    return activation_funcs, activation_derivatives

class NN:
    def __init__(self, input_dim, hidden_dims, activations=None):
        '''
        Parameters
        ----------
        input_dim : int
            size of the input layer.
        hidden_dims : LIST<int>
            List of positive integers where each integer corresponds to the number of neurons 
            in the hidden layers. The list excludes the number of neurons in the output layer.
            For this problem, we fix the output layer to have just 1 neuron.
        activations : LIST<string>, optional
            List of strings where each string corresponds to the activation function to be used 
            for all hidden layers. The list excludes the activation function for the output layer.
            For this problem, we fix the output layer to have the sigmoid activation function.
        ----------
        Returns : None
        ----------
        '''
        assert(len(hidden_dims) > 0)
        assert(activations == None or len(hidden_dims) == len(activations))
         
        # If activations is None, we use sigmoid activation for all layers
        if activations == None:
            self.activations = [sigmoid]*(len(hidden_dims)+1)
            self.activation_derivatives = [sigmoid_derivative]*(len(hidden_dims)+1)
        else:
            self.activations, self.activation_derivatives = get_activation_functions(activations + ['sigmoid'])

        self.weights = []
        self.biases = []

        layers = [input_dim]+hidden_dims+[1]
        
        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(layers[i+1],layers[i]))
            self.biases.append(np.random.randn(layers[i+1],1))

        self.velocity_W = [np.zeros_like(W) for W in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]

        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        self.t = 0

    def forward(self, X):
        '''
        Parameters
        ----------
        X : input data, numpy array of shape (N, D) where N is the number of examples and D 
            is the dimension of each example
        ----------
        Returns : output probabilities, numpy array of shape (N, 1) 
        ----------
        '''
        # Forward pass
        
        self.A_values = [X.T]
        self.Z_values = []
        A=X.T
        
        for W, b, activation in zip(self.weights, self.biases, self.activations):        
            Z = W@A + b
            A = activation(Z)
            
            self.Z_values.append(Z)
            self.A_values.append(A)   

        # output_probs = self.A_values[-1].reshape(-1)
        output_probs = A.T
        return output_probs

    def backward(self, X, y):
        '''
        Parameters
        ----------
        X : input data, numpy array of shape (N, D) where N is the number of examples and D 
            is the dimension of each example
        y : target labels, numpy array of shape (N, 1) where N is the number of examples
        ----------
        Returns : gradients of weights and biases
        ----------
        '''
        # Backpropagation

        N = X.shape[0]

        y_hat = self.A_values[-1]
        output_error = ((1-y)/(1-y_hat) - y/y_hat)/N
        delta = output_error * self.activation_derivatives[-1](self.Z_values[-1])
        
        self.grad_weights = []
        self.grad_biases = []

        for i in reversed(range(len(self.weights))):
            grad_w = delta @ self.A_values[i].T 
            grad_b = np.sum(delta, axis=1, keepdims=True)
            
            self.grad_weights.insert(0, grad_w)
            self.grad_biases.insert(0, grad_b.reshape(-1,1))
            
            if i>0:
                delta = np.dot(self.weights[i].T, delta) * self.activation_derivatives[i-1](self.Z_values[i-1])

        return self.grad_weights, self.grad_biases

    def step_bgd(self, weights, biases, delta_weights, delta_biases, optimizer_params, epoch):
        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.
            optimizer_params: Dictionary containing the following keys:
                learning_rate: Learning rate for the update step.
                gd_flag: 1 for Vanilla GD, 2 for GD with Exponential Decay, 3 for Momentum
                momentum: Momentum coefficient, used when gd_flag is 3.
                decay_constant: Decay constant for exponential learning rate decay, used when gd_flag is 2.
            epoch: Current epoch number
        '''
        gd_flag = optimizer_params['gd_flag']
        learning_rate = optimizer_params['learning_rate']
        momentum = optimizer_params['momentum']
        decay_constant = optimizer_params['decay_constant']

        if gd_flag == 1:
            updated_W = [W - learning_rate*dW for W, dW in zip(weights, delta_weights)]
            updated_B = [b - learning_rate*db for b, db in zip(biases, delta_biases)]
            return updated_W, updated_B

        if gd_flag == 2:
            learning_rate = learning_rate * np.exp(-decay_constant*epoch)
            updated_W = [W - learning_rate*dW for W, dW in zip(weights, delta_weights)]
            updated_B = [b - learning_rate*db for b, db in zip(biases, delta_biases)]
            return updated_W, updated_B

        if gd_flag == 3:
            self.velocity_W = [momentum*vW + (1-momentum)*dW for vW, dW in zip(self.velocity_W, delta_weights)]
            self.velocity_b = [momentum*vb + (1-momentum)*db for vb, db in zip(self.velocity_b, delta_biases)]

            updated_W = [W - learning_rate*vW for W, vW in zip(weights, self.velocity_W)]
            updated_B = [b - learning_rate*vb for b, vb in zip(biases, self.velocity_b)]

            return updated_W, updated_B


    def step_adam(self, weights, biases, delta_weights, delta_biases, optimizer_params):
        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.
            optimizer_params: Dictionary containing the following keys:
                learning_rate: Learning rate for the update step.
                beta: Exponential decay rate for the first moment estimates.
                gamma: Exponential decay rate for the second moment estimates.
                eps: A small constant for numerical stability.
        '''
        learning_rate = optimizer_params['learning_rate']
        beta = optimizer_params['beta1']
        gamma = optimizer_params['beta2']
        eps = optimizer_params['eps']       
            
        self.t += 1

        updated_W, updated_B = [], []

        for i in range(len(weights)):
            self.v_weights[i] = beta*self.v_weights[i] + (1-beta)*delta_weights[i]
            self.m_weights[i] = gamma*self.m_weights[i] + (1-gamma)*(delta_weights[i]**2)
            v_hat = self.v_weights[i] / (1-beta**self.t)
            m_hat = self.m_weights[i] / (1-gamma**self.t)
            
            updated_W.append(weights[i] - learning_rate * (v_hat / (np.sqrt(m_hat)+eps)))
            
            self.v_biases[i] = beta*self.v_biases[i] + (1-beta)*delta_biases[i]
            self.m_biases[i] = gamma*self.m_biases[i] + (1-gamma)*(delta_biases[i]**2)
            v_hat = self.v_biases[i] / (1-beta**self.t)
            m_hat = self.m_biases[i] / (1-gamma**self.t)
            
            updated_B.append(biases[i] - learning_rate * (v_hat / (np.sqrt(m_hat)+eps)))

        return updated_W, updated_B

    def train(self, X_train, y_train, X_eval, y_eval, num_epochs, batch_size, optimizer, optimizer_params):
        train_losses = []
        test_losses = []
        for epoch in range(num_epochs):
            # Divide X,y into batches
            X_batches = np.array_split(X_train, X_train.shape[0]//batch_size)
            y_batches = np.array_split(y_train, y_train.shape[0]//batch_size)
            for X, y in zip(X_batches, y_batches):
                # Forward pass
                self.forward(X)
                # Backpropagation and gradient descent weight updates
                dW, db = self.backward(X, y)
                if optimizer == "adam":
                    self.weights, self.biases = self.step_adam(
                        self.weights, self.biases, dW, db, optimizer_params)
                elif optimizer == "bgd":
                    self.weights, self.biases = self.step_bgd(
                        self.weights, self.biases, dW, db, optimizer_params, epoch)

            # Compute the training accuracy and training loss
            train_preds = self.forward(X_train)
            train_loss = np.mean(-y_train*np.log(train_preds) - (1-y_train)*np.log(1-train_preds))
            train_accuracy = np.mean((train_preds > 0.5).reshape(-1,) == y_train)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            train_losses.append(train_loss)

            # Compute the test accuracy and test loss
            test_preds = self.forward(X_eval)
            test_loss = np.mean(-y_eval*np.log(test_preds) - (1-y_eval)*np.log(1-test_preds))
            test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
            print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            test_losses.append(test_loss)

        return train_losses, test_losses

    
    # Plot the loss curve
    def plot_loss(self, train_losses, test_losses, optimizer, optimizer_params):
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if optimizer == "bgd":
            plt.savefig(f"loss_bgd_{optimizer_params['gd_flag']}.png")
        else:
            plt.savefig(f'loss_adam.png')
 

# Example usage:
if __name__ == "__main__":
    # Read from data.csv 
    csv_file_path = "data_train.csv"
    eval_file_path = "data_eval.csv"
    
    data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=0)
    data_eval = np.genfromtxt(eval_file_path, delimiter=',', skip_header=0)

    # Separate the data into X (features) and y (target) arrays
    X_train = data[:, :-1]
    y_train = data[:, -1]
    X_eval = data_eval[:, :-1]
    y_eval = data_eval[:, -1]

    # Create and train the neural network
    input_dim = X_train.shape[1]
    X_train = X_train**2
    X_eval = X_eval**2
    hidden_dims = [4,2] # the last layer has just 1 neuron for classification
    num_epochs = 30
    batch_size = 100
    activations = ['sigmoid', 'sigmoid']
    
    optimizer = "bgd"
    optimizer_params = {
        'learning_rate': 0.05,
        'gd_flag': 3,
        'momentum': 0.99,
        'decay_constant': 0.2
    }
    
    # For Adam optimizer you can use the followin
    # optimizer = "adam"
    # optimizer_params = {
    #     'learning_rate': 0.01,
    #     'beta1' : 0.9,
    #     'beta2' : 0.999,
    #     'eps' : 1e-8
    # }
     
    model = NN(input_dim, hidden_dims)
    train_losses, test_losses = model.train(X_train, y_train, X_eval, y_eval,
                                    num_epochs, batch_size, optimizer, optimizer_params) #trained on concentric circle data 
    test_preds = model.forward(X_eval)

    test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
    print(f"Final Test accuracy: {test_accuracy:.4f}")

    model.plot_loss(train_losses, test_losses, optimizer, optimizer_params)
