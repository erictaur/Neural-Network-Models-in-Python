import numpy as np
import csv
from matplotlib import pyplot as plt

empty_list = []
count = 0
for i in range(10000):
    x = np.random.uniform(-6, 6)
    y = np.random.uniform(-3, 3)
    print(count)
    if (2*y - x < 6) & (2*y + x < 6) & (2*y + x > -6) & (2*y - x > -6):
        empty_list.append([x, y])
        count += 1
    if count > 500:
        break

x_list = [empty_list[m][0] for m in range(len(empty_list))]
y_list = [empty_list[m][1] for m in range(len(empty_list))]
data_list = [x_list, y_list]

row_dict = {}

for x in range(len(x_list)):
    row_dict["row{0}".format(x)] = [n[x] for n in data_list]

with open('C:/Users/KHT/Desktop/hw2out.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['x', 'y'])
    for key, values in row_dict.items():
        w.writerow([values[0], values[1]])

plt.grid(True)
plt.plot(*zip(*empty_list), 'bs')
plt.show()

# x = np.random.uniform(-6, 6)
# print(x)
