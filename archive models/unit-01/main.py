import numpy as np

class Neuron:
    def __init__(self, w, b, speed_learning, x ,true):
        self.w = w
        self.b = b
        self.speed_learning = speed_learning
        self.true = true
        self.x = x

    def sigmoidProizvodnoe(self, a):
        result = a * (1-a)
        return result
    
    def sigmoid(self, sum):
        result = 1/(1 + np.exp(-sum))
        return result

    def forward(self):
        sum = (self.x * self.w) + self.b
        result = self.sigmoid(sum)
        listik = [result, sum]
        return listik
    
    def backward(self, forward_result):
        loss = forward_result[0] - 1
        gradient_w = loss * self.sigmoidProizvodnoe(forward_result[0]) *  self.x
        gradient_b = loss * self.sigmoidProizvodnoe(forward_result[0]) * self.true
        w_new = self.w - self.speed_learning*(gradient_w)
        b_new = self.b - self.speed_learning*(gradient_b)
        self.w = w_new
        self.b = b_new


