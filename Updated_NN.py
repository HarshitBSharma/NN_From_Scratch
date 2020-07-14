import numpy as np
np.random.seed(0)

class DimensionsMismatchError(Exception):
    def __init__(self, expected_dims, actual_dims):
        message = f"Expected input to have {expected_dims} rows, got {actual_dims} instead"
        super().__init__(message)

class NeuralNetwork():
    
    def __init__(self):
        self.layers = []
        self.parameters = {}
        self.activations = []
        self.input_dims = 0
        self.activations = []
        self.forward_cache = {}
        self.backward_cache = []
        self.loss = 0
        self.input_data = []
        self.gradients = []

    def add(self, size_of_layer, activation='relu', input_layer=False):
        if input_layer:
            self.input_dims = size_of_layer
        else:
            self.layers.append({"neurons": size_of_layer, "activation": activation})
        
    def initialize_parameters(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                w_temp = np.random.randn(self.layers[0]['neurons'], self.input_dims)
                b_temp = np.random.randn(self.layers[0]['neurons'], 1)
            else:
                w_temp = np.random.randn(self.layers[i]['neurons'], self.layers[i-1]['neurons'])
                b_temp = np.random.randn(self.layers[i]['neurons'], 1)
                #self.parameters.append({"W": w_temp, "b": b_temp})
            self.parameters["W" + str(i)] = w_temp
            self.parameters["b" + str(i)] = b_temp 
        
 
    def __forward_step(self, W, A_prev, b, activation_function):
        Z_temp = np.dot(W, A_prev) + b
        if activation_function == "relu":
            A_temp = np.array(np.maximum(0, Z_temp))
        #self.forward_cache.append({"Z": Z_temp, "A": A_temp}) 
        return Z_temp, A_temp

        
    def forward_prop(self, input_data):
        self.input_data = np.array(input_data)
        if input_data.ndim == 1:
            input_data = input_data.reshape(input_data.shape[0], 1)
        if self.input_dims != input_data.shape[0]:
            raise DimensionsMismatchError(self.input_dims, input_data.shape[0])
        n_layers = len(self.layers)
        for i in range(n_layers):
            W = self.parameters["W" + str(i)]
            b = self.parameters["b" + str(i)]
            activation_function = self.layers[i]["activation"]
            if i == 0:
                A_prev = input_data
            else:
                A_prev = self.forward_cache["A" + str(i-1)]
            
            Z_temp, A_temp = self.__forward_step(W=W, A_prev=A_prev, b=b, activation_function=activation_function)
            self.forward_cache["Z" + str(i)] = Z_temp
            self.forward_cache["A" + str(i)] = A_temp
    
    def get_prediction(self):
        return self.forward_cache["A" + str(len(self.layers)-1)]


    def back_prop(self, Y, learning_rate=0.01):
        Y = Y.reshape(self.get_prediction().shape)
        n_layers= len(self.layers)
        m = self.forward_cache["A" + str(n_layers-1)].shape[1]
        loss, dA_prev = self.__calculate_loss(Y)
        loss = (1/m)*np.sum(loss)
        self.loss=loss
        for i in reversed(range(n_layers)):
            if i != 0:
                layer_A_prev = self.forward_cache["A" + str(i-1)]
            else:
                layer_A_prev = self.input_data
            layer_z = self.forward_cache["Z" + str(i)]
            layer_activation_function = self.layers[i]["activation"]
            
            W = self.parameters["W"+ str(i)]
            b = self.parameters["b" + str(i)]
            dZ = self.g_prime(dA_prev, layer_activation_function)
            
            dA_prev, dW, db = self.__backward_step(dZ, W, layer_A_prev, b)
            
            W, b = self.__update_parameters(dW, db, W, b, learning_rate)
            self.parameters["W" + str(i)] = W
            self.parameters["b" + str(i)] = b 
            

    def __calculate_loss(self, Y, loss_fn="mse"):
        Y_pred = self.get_prediction()
        if loss_fn == "mse":
            loss = (Y-Y_pred) ** 2
            dA_prev = 2 * (Y-Y_pred)
        return loss, dA_prev


    def g_prime(self, Z, activation_function):
        if activation_function == "relu":
            g_prime = np.array([[1 if p>=0 else 0 for p in q]for q in Z])
        return g_prime


    def __backward_step(self, dZ, W, A_prev, b):
        m = A_prev.shape[1]
        dW = (1/m) * np.dot(dZ, A_prev.T)
        #print(f"dZ{dZ}\nA_Prev{A_prev} \ndW{dW}")
        db = np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def __update_parameters(self, dW, db, W, b, learning_rate):
        W -= learning_rate * dW
        b -= learning_rate * db
        return W, b
            




    










        


