import numpy as np
np.random.seed(0)

class DimensionsMismatchError(Exception):
    def __init__(self, expected_dims, actual_dims):
        message = f"Expected input to be {expected_dims}, got {actual_dims}"
        super().__init__(message)

class NeuralNetwork():
    
    def __init__(self):
        self.layers = []
        self.parameters = []
        self.activations = []
        self.input_dims = []
        self.activations = []
        self.forward_cache = []
        self.backward_cache = []

    def add(self, size_of_layer, activation='relu', input_layer=False):
        if input_layer:
            self.input_dims.append(size_of_layer)
        else:
            self.layers.append({"nodes": size_of_layer, "activation": activation})
        
    def initialize_parameters(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                w_temp = np.random.randn(self.layers[0]['nodes'], self.input_dims[0])
                b_temp = np.random.randn(self.layers[0]['nodes'], 1)
                self.parameters.append({"W": w_temp, "b": b_temp})
            else:
                w_temp = np.random.randn(self.layers[i]['nodes'], self.layers[i-1]['nodes'])
                b_temp = np.random.randn(self.layers[i]['nodes'], 1)
                self.parameters.append({"W": np.squeeze(w_temp), "b": np.squeeze(b_temp)})
        
 
    def __forward_step(self, W, A_prev, b, activation_function):

        Z_temp = np.squeeze(np.dot(W, A_prev) + b)
        print(type(Z_temp))
        if activation_function is "relu":
            A_temp = np.maximum(0, Z_temp)
        self.forward_cache.append({"Z": np.squeeze(Z_temp), "A": np.squeeze(A_temp)})  

        
    def forward_prop(self, input_data):
        input_data = np.array(input_data)
        if input_data.ndim == 1:
            input_data = input_data.reshape(input_data.shape[0], 1)
        """if self.input_dims[0] != input_data.shape[0]:
            raise DimensionsMismatchError(self.input_dims[0], input_data.shape[0])"""
        for i, (parameter, layer) in enumerate(zip(self.parameters, self.layers)):
            W = parameter["W"]
            b = parameter["b"]
            activation = layer["activation"]
            if i == 0:
                A_prev = input_data
            else:
                A_prev = self.forward_cache[i-1]["A"]
            self.__forward_step(W=W, A_prev=A_prev, b=b, activation_function=activation)





        

