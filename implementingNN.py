from Updated_NN import NeuralNetwork
model = NeuralNetwork()
model.add(3, input_layer=True)
model.add(3)
model.add(2)
model.add(1)
model.initialize_parameters()
model.forward_prop([1, 2, 3])
for x in model.parameters:
    print(f"W = {x['W']}\nb = {x['b']}")
for x in model.forward_cache:
    print(x)