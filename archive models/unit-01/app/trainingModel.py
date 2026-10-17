import numpy as np
import math
import os
import ai
import json



def saveDataSet(dataSet):
   with open("F:\\Programming\\Pulsar AI\\data\\dataset.json", "w+") as file:
    json.dump(dataSet, file)
    return "Успешно"
   
def loadDataSet():
   with open("F:\\Programming\\Pulsar AI\\data\\dataset.json", "r+") as file:
    dataSet =  json.load(file)
    return dataSet
   
def saveData(data):
   with open("F:\\Programming\\Pulsar AI\\data\\data.json", "w") as file:
    file.seek(0)
    json.dump(data, file)
    file.truncate()
    return "Успешно"
       
def loadData():
   with open("F:\\Programming\\Pulsar AI\\data\\data.json", "r+") as file:
    data =  json.load(file)
    return data


def training():
  data = loadData()
  dataSet = loadDataSet()

  model = ai.Neuron(data["w"], data["b"], data["speed_learning"], dataSet["x"], 1)

  beforeLearning = model.forward()
  print(beforeLearning[0])
  for i in range(400):
    afterlearning = model.backward(beforeLearning)
    ww = model.forward()
    print(ww[0])
  dictSave = {
    "w": model.w,
    "b": model.b,
    "speed_learning": data["speed_learning"]
  }
  saveData(dictSave)



def checkModel():
  data = loadData()
  dataset = loadDataSet()
  model = ai.Neuron(data["w"], data["b"], data["speed_learning"], dataset["x"], 0)
  result = model.forward()
  print(result[0])


#training()
checkModel()