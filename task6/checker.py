import torch
import array
import os
import torch.nn as nn
import time
batchSize = 1
inputSize = 32 * 32
hiddenSize1 = 16 * 16
hiddenSize2 = 4 * 4
outputSize = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

w1 = array.array('f')
w2 = array.array('f')
w3 = array.array('f')

b1 = array.array('f')
b2 = array.array('f')
b3 = array.array('f')
h_input = array.array('f')
with open('input.bin', 'rb') as input_file:
    # Read data from input file into float array
    h_input.fromfile(input_file, inputSize)

with open('wb.bin', 'rb') as wb_file:
    # Read data from input file into float array
    w1.fromfile(wb_file, inputSize * hiddenSize1)
    w2.fromfile(wb_file, hiddenSize1 * hiddenSize2)
    w3.fromfile(wb_file, hiddenSize2 * outputSize)
    b1.fromfile(wb_file, hiddenSize1)
    b2.fromfile(wb_file, hiddenSize2)
    b3.fromfile(wb_file, outputSize)

tw1 = torch.tensor(w1)
tw1 = tw1.view(hiddenSize1, inputSize)
tw2 = torch.tensor(w2)
tw2 = tw2.view(hiddenSize2, hiddenSize1)
tw3 = torch.tensor(w3)
tw3 = tw3.view(outputSize, hiddenSize2)
tb1 = torch.tensor(b1)
tb2 = torch.tensor(b2)
tb3 = torch.tensor(b3)


class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(32**2, 16**2) # входной слой
        self.fc1.weight = nn.Parameter(tw1)
        self.fc1.bias = nn.Parameter(tb1)

        self.fc2 = nn.Linear(16**2, 4**2) # скрытый слой
        self.fc2.weight = nn.Parameter(tw2)
        self.fc2.bias = nn.Parameter(tb2)

        self.fc3 = nn.Linear(4 ** 2,1) # скрытый слой
        self.fc3.weight = nn.Parameter(tw3)
        self.fc3.bias = nn.Parameter(tb3)
# прямое распространение информации

    def forward(self, x):

        sigmoid = nn.Sigmoid()

        x = sigmoid(self.fc1(x))

        x = sigmoid(self.fc2(x))

        x = sigmoid(self.fc3(x))

        return x


input_layer = torch.tensor(h_input).to(device) # входные данные нейронной сети
net = Net().to(device) # создание объекта "нейронная сеть"
start_time = time.time()
result = net(input_layer).cpu() # запуск прямого распространения информации

print("--- %s seconds ---" % (time.time() - start_time))

print(result)

