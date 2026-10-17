import pulsar
import json

def SaveData(w, b, speed_learning):
    with open("data.json", "w+") as file:
        dataset = {"w": w,"b": b,"speed_learning": speed_learning}
        json.dump(dataset, file)

def LoadData():
    with open("data.json", "r") as file:
        source = json.load(file)
        return source

def Training():
    model = pulsar.Triada([0.5, 0.3, 0.1], [0.2, -0.5, 0.3], 0.1, 1 , 0.7)
    source = model.forward()
    for i in range(3000):
        training = model.backward(source)
        roll = model.forward()
        print(roll)
        SaveData(model.w, model.b, model.speed_learning)

def CheckModel():
    test = LoadData()
    model = pulsar.Triada(test["w"], test["b"], 0.1, 1 , test["speed_learning"])
    print(model.forward())

#Training()
CheckModel()
