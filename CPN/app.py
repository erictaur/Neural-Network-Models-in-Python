from cpn import CPN
import numpy as np


with open('C:/Users/KHT/Desktop/hw6out.csv') as f:
    lines = f.readlines()

lines = [line.strip() for line in lines]

x_list = []
y_list = []
z_list = []

for line in lines:
    data = line.split(',')
    x_list.append(float(data[0]))
    y_list.append(float(data[1]))
    z_list.append(float(data[2]))

input_list = np.array([[a, b] for a, b in zip(x_list, y_list)])
output_list = np.array([a for a in z_list])

for i in range(3):
    test_iter = [9, 15, 50]
    cpn = CPN(input_list, test_iter[i], 0.01, output_list)
    cpn.train()
    cpn.run()



