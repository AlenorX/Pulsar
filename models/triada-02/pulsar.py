import numpy as np
import json


class Triada():
    def __init__(self,w,b,x, y_true,speed_learning):
        self.w = w
        self.b = b
        self.x = x
        self.y_true = y_true
        self.speed_learning = speed_learning
    


    def sigmoidProizvodnoe(self, a):
        result = a * (1-a)
        return result
    
    def forward(self):
        self.z1 = self.x * self.w[0] + self.b[0]
        self.a1 = 1/(1 + np.exp(-self.z1))

        self.z2 = self.a1 * self.w[1] + self.b[1]
        self.a2 = 1/(1+np.exp(-self.z2))

        self.z3 = self.a2 * self.w[2] + self.b[2]
        self.a3 = 1/(1+np.exp(-self.z3))
        return self.a3

    def backward(self, forward_result):
        '''
        delta = личная ошибка слоя, вес в дельтах = влияние на эту ошибку, производная = чувствительность.
        Каждый слой (кроме последнего) умножается на вес, который соединяет ЕГО СО СЛЕДУЮЩИМ слоем
        Поэтому:
        δ₃ — последний → нет веса
        δ₂ — есть вес w₃ к слою 3
        δ₁ — есть вес w₂ к слою 2
        Тоесть мы для каждого слоя берём ошибку предыдущего и умножаем на вес (влияние текущего слоя)

Вес всегда идёт ОТ текущего слоя К следующему. Для последнего слоя следующего нет!
        '''
        error = forward_result - self.y_true
        delta3 = error * self.sigmoidProizvodnoe(self.a3)
        grad_w3 = delta3 * self.a2
        grad_b3 = delta3

        delta2 = delta3 * self.w[2] * self.sigmoidProizvodnoe(self.a2)
        grad_w2 = delta2 * self.a1
        grad_b2 = delta2

        delta1 = delta2 * self.w[1] * self.sigmoidProizvodnoe(self.a1)
        grad_w1 = delta1 * self.x
        grad_b1 = delta1

        w3_new = self.w[2] - self.speed_learning * grad_w3
        b3_new = self.b[2] - self.speed_learning * grad_b3
        w2_new = self.w[1] - self.speed_learning * grad_w2
        b2_new = self.b[1] - self.speed_learning * grad_b2
        w1_new = self.w[0] - self.speed_learning * grad_w1
        b1_new = self.b[0] - self.speed_learning * grad_b1

        self.w[0], self.b[0] = w1_new, b1_new
        self.w[1], self.b[1] = w2_new, b2_new
        self.w[2], self.b[2] = w3_new, b3_new


        
        


