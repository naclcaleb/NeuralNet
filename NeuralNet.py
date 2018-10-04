import tensorflow as tf
import numpy



class NN:
    prediction = []
    def __init__(self,input_length):
        self.layers = []
        self.input_length = input_length
    def addLayer(self, layer):
        self.layers.append(layer)
        if len(self.layers) >1:
            self.layers[len(self.layers)-1].setWeights(len(self.layers[len(self.layers)-2].neurons))
        else:
            self.layers[0].setWeights(self.input_length)
    def feedForward(self, inputs):
        _inputs = inputs
        for i in range(len(self.layers)):
            self.layers[i].process(_inputs)
            _inputs = self.layers[i].output
        self.prediction = _inputs

    def calculateErr(self, target):
        out = []
        for i in range(0,len(self.prediction)):
            out.append(  (self.prediction[i] - target[i]) ** 2  )
        return out

     
        

class Layer:

    neurons = []
    weights = []
    biases = []
    output = []
    
    def __init__(self,length,function):
        for i in range(0,length):
            self.neurons.append(Neuron(function))
            self.biases.append(numpy.random.randn())

    def setWeights(self, inlength):
        for i in range(0,inlength):
            self.weights.append([])
            for j in range(0, inlength):
                self.weights[i].append(numpy.random.randn())
    
    def process(self,inputs):
        for i in range(0, len(self.neurons)):
            self.output.append(self.neurons[i].run(inputs,self.weights[i], self.biases[i]))
    

class Neuron:
    output = 0
    def __init__(self, function):
        self.function = function
    def run(self, inputs, weights, bias):
        self.output = self.function(inputs,weights,bias)
        return self.output

def sigmoid(n):
    return 1/(1+numpy.exp(n))


def inputlayer_func(inputs,weights,bias):
    return inputs

def l2_func(inputs,weights,bias):
    out = 0
    
    for i in range(0,len(inputs)):
        out += weights[i] * inputs[i]
    out += bias
    
    return sigmoid(out)

NNet = NN(2)


l2 = Layer(1,l2_func)


NNet.addLayer(l2)
NNet.feedForward([2.0,1.0])
print(NNet.prediction)

