import numpy as np
np.random.seed(0)

class NeuralNetwork:
    def __init__(self):
        self.layer = []
        self.parameters = []
        self.caches = []
        self.gradients = {}
        
    def add(self, layer_size, activation_function="relu"):
        self.layer.append({"neurons": layer_size, "activation":activation_function})

    def initialize_parameters(self):
        L = len(self.layer)
        for i in range(1, L):
            W_temp = np.random.randn(self.layer[i]["neurons"], self.layer[i-1]["neurons"]) * 0.1
            b_temp = np.random.randn(self.layer[i]["neurons"], 1) * 0.1
            # W_temp = 2 * np.ones((self.layer[i]["neurons"], self.layer[i-1]["neurons"]))
            # b_temp = 2 * np.ones((self.layer[i]["neurons"], 1))
            self.parameters.append({"W": W_temp, "b":b_temp})

            """self.parameters["W" + str(i)] = W_temp
            self.parameters["b" + str(i)] = b_temp"""


    def forward_step(self, W, A_prev, b, activation_function):
        # print(f"W: {W}, A_prev : {A_prev}, b: {b}")
        Z = np.dot(W, A_prev) + b
        A = self.activation(Z, activation_function)
        return {"W": W, "b": b, "A_prev": A_prev, "Z": Z, "A": A}


    def activation(self, Z, activation_function):
        if activation_function == "relu":
            A = np.array(np.maximum(0, Z))
        elif activation_function == "sigmoid":
            A = 1/(1+np.exp(-Z))    
        return A    


    def forward_prop(self, inputs):
        self.caches = []
        A_prev = inputs
        L = len(self.parameters)
        for i in range(1, L):
            """print(f"Currently on layer {i} and values are " + 
                    f"W: {self.parameters[i-1]['W']} " +
                    f"A_prev: {A_prev} " +
                    f"b: {self.parameters[i-1]['b']} " +
                    f"Activation function: {self.layer[i]['activation']}")"""
            cache = self.forward_step(self.parameters[i-1]["W"], A_prev, self.parameters[i-1]["b"], self.layer[i]["activation"])
            A_prev = cache["A"]
            #print("Calculated cache is ")
            self.caches.append(cache)
        cache = self.forward_step(self.parameters[-1]["W"], A_prev, self.parameters[-1]["b"], self.layer[-1]["activation"])
        self.caches.append(cache)
        return self.caches

    
    def calculate_cost(self, AL, Y):
        m = AL.shape[1]
        cost = (1/m)*np.sum((Y - AL) ** 2)
        cost = np.squeeze(cost)
        return cost


    def backward_step(self, dZ, cache):
        A_prev = cache["A_prev"]
        W = cache["W"]
        b = cache["b"]
        m = A_prev.shape[1]
        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dW, dA_prev, db


    def get_g_prime(self, dA, cache, activation_function):
        if activation_function == "relu":
            dZ = np.array(dA, copy=True)
            Z = cache["Z"]
            # print(f"Z: {Z}, dZ:{dZ}")
            dZ[Z <= 0] = 0
        elif activation_function == "sigmoid":
            t = 1/(1 + exp(-cache["Z"]))
            dZ = dA * t * (1-t)
        return dZ


    def back_prop(self, Y):
        self.gradients = {}
        AL = self.caches[-1]["A"]
        L = len(self.caches)
        Y = Y.reshape(AL.shape)

        dAL = -2 * (Y-AL)
        current_cache = self.caches[-1]
        dZ = self.get_g_prime(dAL, current_cache, self.layer[-1]["activation"])
        dW, dA_prev, db = self.backward_step(dZ, current_cache)
        self.gradients["dW" + str(L)] = dW
        self.gradients["dA_prev" + str(L-1)] = dA_prev
        self.gradients["db" + str(L)] = db
        
        for l in reversed(range(L-1)):
            # print(f"l is {l} and L is {L}")
            current_cache = self.caches[l]
            dZ = self.get_g_prime(self.gradients["dA_prev" + str(l+1)], current_cache, self.layer[l]["activation"])
            dW, dA_prev, db = self.backward_step(dZ, current_cache)
            self.gradients["dW" + str(l+1)] = dW
            self.gradients["dA_prev" + str(l)] = dA_prev
            self.gradients["db" + str(l+1)] = db
        
        return self.gradients

    def train(self, inputs, outputs, epochs):
        self.initialize_parameters()
        for i in range(epochs):
            self.forward_prop(inputs)
            self.back_prop(outputs)

    def update_parameters(self, learning_rate=0.01):
        L = len(self.parameters)
        for l in range(L):
            # print(f"Inside the for loop, l={l}, W: {self.parameters[l]['W']}, dW: {self.gradients['dW' + str(l+1)]}")
            self.parameters[l]["W"] = self.parameters[l]["W"] - learning_rate*self.gradients["dW" + str(l+1)]
            self.parameters[l]["b"] = self.parameters[l]["b"] - learning_rate*self.gradients["db" + str(l+1)]


    

    
inputs = np.array([[1, 2, 3], [4, 5, 6]])
#print(inputs.shape)
outputs = np.array([5, 7, 9]).reshape(1, 3)
#print(outputs.shape)
model = NeuralNetwork()
model.add(2)
model.add(3)
model.add(1)
model.initialize_parameters()

for i in range(500):
    model.forward_prop(inputs)
    #print(model.caches)
    model.back_prop(outputs)
    #print(model.caches)
    #print(model.gradients)
    model.update_parameters()
    #print(model.forward_cache)
    print(model.caches[-1]["A"])


        
