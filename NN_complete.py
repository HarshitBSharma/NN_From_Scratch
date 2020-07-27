import numpy as np
np.random.seed(0)
class NeuralNetwork:
    def __init__(self):
        self.layer = []
        self.parameters = []
        self.forward_cache = {}

    def add(self, layer_size, activation_function='relu'):
        self.layer.append({"neurons": layer_size, "activation": activation_function})


    def initialize_parameters(self):
        L = len(self.layer)
        self.parameters.append([])
        for i in range(1, L):
            #w_temp =  0.01 * np.random.randn(self.layer[i]["neurons"], self.layer[i-1]["neurons"])
            #b_temp =  0.01 * np.random.randn(self.layer[i]["neurons"], 1)
            w_temp = 2 * np.ones((self.layer[i]["neurons"], self.layer[i-1]["neurons"]))
            b_temp = 2 * np.ones((self.layer[i]["neurons"], 1))
            self.parameters.append({"W": w_temp, "b":b_temp})
        
    def forward_step(self, W, A_prev, b, activation_function):
        Z_temp = np.dot(W, A_prev) + b
        A_temp = self.activation(Z_temp, activation_function)
        return Z_temp, A_temp


    def activation(self, Z, activation_function):
        if activation_function == "relu":
            A = np.array(np.maximum(0, Z))
        return A

    def forward_prop(self, inputs):
        self.inputs = inputs
        A_prev = inputs
        L = len(self.layer)
        for i in range(1, L):
            W = self.parameters[i]["W"]
            b = self.parameters[i]["b"]
            activation_function = self.layer[i]["activation"]
            Z, A = self.forward_step(W, A_prev, b, activation_function)
            self.forward_cache["Z" + str(i)] = Z
            self.forward_cache["A" + str(i)] = A
            A_prev = A
            #print(f"W is {W} and b is {b}")

    
    def back_prop(self, outputs):
        m=3
        L = len(self.layer)
        #Getting parameters
        W2 = self.parameters[2]["W"]

        #Getting Forward cache

        prediction = self.forward_cache["A" + str(L-1)]
        print(f"Prediction:{prediction}")
        Z_last = self.forward_cache["Z" + str(L-1)]
        #print(f"Z_last: {Z_last}")

        dA2 = -2 * (outputs-prediction)
        dZ2 = np.array(dA2, copy=True)
        dZ2[Z_last <= 0] = 0
        print(f"dA2: {dA2}, dZ2 : {dZ2}")
        dW2 = (1/m) * np.dot(dZ2, self.forward_cache["A" + str(1)].T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        learning_rate = 0.01

        Z1 = self.forward_cache["Z" + str(1)]
        A1 = self.forward_cache["A" + str(1)]
        print(f"dW2: {dW2} db2: {db2}")
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = np.array(dA1, copy=True)
        dZ1[Z1 <= 0] = 0
        print(f"dA1 = {dA1}, dZ1 = {dZ1}, Z1 = {Z1}")
        print(f"Before calculating dw1, dZ1 : {dZ1} inputs.T : {self.inputs.T}")
        dW1 = (1/m) * np.dot(dZ1, self.inputs.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims = True)

        print(f"dW1: {dW1}, db1 = {db1}")

        #Updating parameters
        self.parameters[2]["W"] -= learning_rate * dW2
        self.parameters[2]["b"] -= learning_rate * db2

        self.parameters[1]["W"] -= learning_rate * dW1
        self.parameters[1]["b"] -= learning_rate * db1


    def update_parameters(self):
        for i in range(1, len(self.layer)-1):
            self.parameters[i]["W"] -= learning_rate * dW2
            self.parameters[i]["b"] -= learning_rate * db2

"""        W = self.parameters[2]["W"]
        b = self.parameters[2]["b"]
        W -= learning_rate * dW
        b -= learning_rate * db
        self.parameters[1]["W"] = W
        self.parameters[1]["b"] = b """






    

inputs = np.array([[1, 2, 3], [4, 5, 6]])
print(inputs.shape)
outputs = np.array([5, 7, 9]).reshape(1, 3)
print(outputs.shape)
model = NeuralNetwork()
model.add(2)
model.add(3)
model.add(1)
#print(model.layer)
model.initialize_parameters()
print("Before starting backprop")
print(model.parameters)
print("Starting forwardprop")
# model.forward_prop(inputs)
# print(model.forward_cache)
for i in range(2):
    model.forward_prop(inputs)
    model.back_prop(outputs)
    #print(model.forward_cache)
print(model.parameters)