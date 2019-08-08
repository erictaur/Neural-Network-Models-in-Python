from rbf import RBF
from som import SOM
import numpy as np

with open('C:/Users/KHT/Desktop/hw5out.csv') as f:
    lines = f.readlines()

lines = [line.strip() for line in lines]

x_list = []
y_list = []
z_list = []

for line in lines:
    data = line.split(',')
    x_list.append(float(data[0]))
    y_list.append(float(data[1]))
    z_list.append(int(data[2]))

input_list = [[a, b] for a, b in zip(x_list, y_list)]
output_list = [a for a in z_list]

input_list = np.array(input_list)
output_list = np.array(output_list)

"""
Get Central points from SOM network
"""

som = SOM(2, 2, 2)
print('Running SOM network...')
som.train(input_list, num_epochs=200, init_learning_rate=0.3)
som_result = [np.array(som.output[i]) for i in range(len(som.output))]

"""
Create RBF objects and run RBF network
"""


print('RBF with random c, updated by LMS: \n')
rbf_1 = RBF(input_list, output_list, 9, 'random', som_c_list=None, update='lms')
rbf_1.run()
print("-------------------------------------")

print('RBF with random c, updated by SGA: \n')
rbf_2 = RBF(input_list, output_list, 9, 'random', som_c_list=None, update='sga')
rbf_2.run()
print("-------------------------------------")

print('RBF with som c, updated by LMS: \n')
rbf_3 = RBF(input_list, output_list, 9, 'som', som_c_list=som_result, update='lms')
rbf_3.run()

# Extra test
"""
Smaller number of central points (4)
Update weight by LMS
"""
rbf_4 = RBF(input_list, output_list, 4, 'som', som_c_list=som_result, update='sga')
rbf_4.run()


