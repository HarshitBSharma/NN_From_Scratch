from Updated_NN import NeuralNetwork
inputs = [[1, 2, 3], [1, 2, 1], [2, 1, 1], [5, 1, 1]]
outputs = [6, 4, 4, 7]
model = NeuralNetwork()
model.add(2, input_layer=True)
model.add(3)
model.add(2)
model.add(10)
model.add(2)
model.initialize_parameters()
model.forward_prop()
model.calculate_gradients()