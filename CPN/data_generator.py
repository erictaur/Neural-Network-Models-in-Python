import numpy as np
import csv

empty_list = []
count = 0
for i in range(1000):
    x = np.random.uniform(1, 100)
    y = np.random.uniform(1, 100)
    z = (np.sin(x)*np.sin(y))/(x*y)
    empty_list.append([x, y, z])


x_list = [empty_list[m][0] for m in range(len(empty_list))]
y_list = [empty_list[m][1] for m in range(len(empty_list))]
z_list = [empty_list[m][2] for m in range(len(empty_list))]
data_list = [x_list, y_list, z_list]

row_dict = {}

for x in range(len(x_list)):
    row_dict["row{0}".format(x)] = [n[x] for n in data_list]

with open('C:/Users/KHT/Desktop/hw6out.csv', 'w', newline='') as f:
    w = csv.writer(f)
    # w.writerow(['x', 'y', 'z'])
    for key, values in row_dict.items():
        w.writerow([values[0], values[1], values[2]])

