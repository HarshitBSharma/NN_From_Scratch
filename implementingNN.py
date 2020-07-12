import numpy as np
from Updated_NN import NeuralNetwork
inputs = np.array([[1, 2, 3, 4], [2, 4, 6, 8]])
outputs = np.array([3, 6, 9, 12])
model = NeuralNetwork()
model.add(2, input_layer=True)
model.add(3)
model.add(1)
model.initialize_parameters()
model.forward_prop(inputs)
model.back_prop(outputs)

