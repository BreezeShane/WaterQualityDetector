import numpy as np
import pandas as pd
from NeuralNetwork import NeuralNetwork


def load_data():
    datalist = []
    sheet = pd.read_excel('data.xls')
    for ceil in sheet.itertuples():

        datalist.append([data for data in ceil])
    return datalist


def train(net):
    total_train_step = 100000
    for step in range(total_train_step):
        inputs = datalist[step % len(datalist)][1:-1]
        targets = np.zeros(output_nodes) + 0.01
        targets[-1] = 0.985
        net.train(inputs, targets, step, total_train_step, optimize_learning_rate=False)

        if step % 100 == 0:
            net.save_model(step)
            print(f"model {step} iteration saved!")

def predict(net):
    iteration = input("Which model you wanna load?")
    try:
        net.load_model(iteration)
    except:
        return
    temperature = input("Please input the temperature of water.")
    TDS = input("Please input the TDS of water.")
    TU = input("Please input the TU of water.")
    PH = input("Please input the PH of water.")
    inputs = [temperature, TDS, TU, PH]
    result = net.predict(inputs)
    print(result)


if __name__ == '__main__':
    datalist = load_data()
    print(datalist)
    print(f"""
Succeeded to load data!
The number of data is { len(datalist) },
and the number of data in each element is { len(datalist[0]) - 1 }.
    """)
    input_nodes = 4
    hidden_nodes = 1000
    output_nodes = 2
    logic_var = input("Wanna train? [Y/n]")
    if logic_var.upper() == 'Y':
        net = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, trained=False)
        train(net)
    logic_var = input("Wanna predict? [Y/n]")
    if logic_var.upper() == 'Y':
        net = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, trained=True)
        predict(net)



