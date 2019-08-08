import numpy as np
import csv
from matplotlib import pyplot as plt

input_list = []
plot_list = []

count = 0

for i in range(10000):
    x = np.random.uniform(-5, 5)
    y = np.random.uniform(-5, 5)
    print([x, y])
    if x*x + y*y < 25.0:
        if (x < 0) & (y < 0):
            input_list.append([x, y, 3])
            plot_list.append([x, y])
            count += 1
        elif (x < 0) & (y > 0):
            input_list.append([x, y, 2])
            plot_list.append([x, y])
            count += 1
        elif(x > 0) & (y < 0):
            input_list.append([x, y, 4])
            plot_list.append([x, y])
            count += 1
        else:
            continue
    if count >= 1000:
        break

plt.grid(True)
plt.plot(*zip(*plot_list), 'bs')
plt.show()

x_list = [input_list[m][0] for m in range(len(input_list))]
y_list = [input_list[m][1] for m in range(len(input_list))]
label_list = [input_list[m][2] for m in range(len(input_list))]
data_list = [x_list, y_list, label_list]

row_dict = {}

for x in range(len(x_list)):
    row_dict["row{0}".format(x)] = [n[x] for n in data_list]

with open('C:/Users/KHT/Desktop/hw3out.csv', 'w', newline='') as f:
    w = csv.writer(f)
    # w.writerow(['x', 'y', 'label'])
    for key, values in row_dict.items():
        w.writerow([values[0], values[1], values[2]])

