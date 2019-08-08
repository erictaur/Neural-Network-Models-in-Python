from som import SOM
import pandas as pd
import numpy as np

input_data = pd.read_csv("C:/Users/KHT/Desktop/hw2out.csv")

som_net = SOM(10, 10, 2)


coor_data = input_data.iloc[np.random.permutation(len(input_data))]
trunc_data = coor_data[["x", "y"]]

print(trunc_data.values)
# print(trunc_data.values.shape[0])

som_net.train(trunc_data.values, num_epochs=1000, init_learning_rate=0.1)

"""
def predict(df):
    bmu, bmu_idx = som_net.find_bmu(df.values)
    df['bmu'] = bmu
    df['bmu_idx'] = bmu_idx
    return df


clustered_df = trunc_data.apply(predict, axis=1)
result_array = clustered_df.to_records().tolist()

print("ID number {0}".format(result_array[0][0]))
print("X Coordinate on topological map {0}".format(result_array[0][4][0]))
print("Y Coordinate on topological map {0}".format(result_array[0][4][1]))
"""
