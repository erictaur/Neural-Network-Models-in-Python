import numpy as np
import csv

"""
def sqrt(target_list):
    return [n**(1/2) for n in target_list]
"""

empty_list = [None]*7000

# Input 1
var1_list = [np.random.uniform(1, 10) for n in empty_list]
var1_list_sin = np.sin(var1_list)
var1_list_final = [5*n for n in var1_list_sin]  # 5*sin(x)

# Input 2
var2_list = [np.random.uniform(1, 10) for n in empty_list]
var2_list_final = [2*n*n for n in var2_list]  # 2*y^2

# Creating Output 1
temp_list_1 = [var1_list_final, var2_list_final]
output_list = list(map(sum, zip(*temp_list_1)))  # Output = (5sin(x))+2y^2


# Summary
# Inputs
"""
print(var1_list)
print(var2_list)
"""

# Outputs
print(output_list)

"""

Creating a transposed input

"""

var_in = [var1_list, var2_list, output_list]

# Collect inputs as dictionary of
row_dict = {}

for x in range(len(var1_list)):
    row_dict["row{0}".format(x)] = [n[x] for n in var_in]

print(row_dict)

with open('C:/Users/KHT/Desktop/hw1out.csv', 'w', newline='') as f:
    w = csv.writer(f)
    for key, values in row_dict.items():
        w.writerow([key, values[0], values[1], values[2]])

