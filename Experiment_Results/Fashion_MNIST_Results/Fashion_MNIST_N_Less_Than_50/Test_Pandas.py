import astropy as asp
import numpy as np
from astropy.table import Table
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import re

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None

tab = Table.read('SSL_MNIST.tex').to_pandas()
df = pd.read_csv('table_ex.csv')

def Over_all_best_accuracy(tab):
    num_of_col = tab.shape[1]
    for i in range(1, num_of_col):
        df = tab.iloc[:, i]
        accuracy_max = df.max()
        index_max = df[df == accuracy_max].index[0]
        name_max = tab.iloc[index_max, 0]
        print(i, 'label per class best accuracy is', accuracy_max, 'with parameter', name_max)

def Fix_N_best_accuracy(tab, step):
    num_of_col = tab.shape[1]
    power_list = list(np.arange(0.1, 5.1, 0.1))
    N_list = []
    block_list = [] #power block
    for j in range(len(power_list)):
        block = tab.iloc[(j * 5):(j * 5 + 5), :]
        block_list.append(block)
    for i in range(5): #for our purpose, we won't generalize this yet
        N = (i + 1) * step
        N_list.append(N)
        rows_in_interest = []
        label_best_accuracy_list = []
        index_max_list = []
        for power in range(len(power_list)):
            rows_in_interest.append(list(block_list[power].iloc[i, 0:6]))
        df = pd.DataFrame(rows_in_interest)
        for k in range(1, df.shape[1]):
            column_in_interest = df.iloc[:, k]
            index_max_list.append(column_in_interest[column_in_interest == column_in_interest.max()].index[0]) 
            label_best_accuracy_list.append(column_in_interest.max())
            print('For N', N, 'and', k , 'label per class best accuracy is', label_best_accuracy_list[k-1], 'with parameter', df.iloc[index_max_list[k-1], 0])

def Fix_m_best_accuracy(tab, step):
    num_of_col = tab.shape[1]
    power_list = list(np.arange(0.1, 5.1, 0.1))
    N_list = []
    block_list = [] #power block
    for j in range(len(power_list)):
        block = tab.iloc[(j * 5):(j * 5 + 5), :]
        block_list.append(block)
    for i in range(5): #for our purpose, we won't generalize this yet
        N = (i + 1) * step
        N_list.append(N)
    for power in range(len(block_list)):
        b = block_list[power]
        label_best_accuracy_list = []
        index_max_list = []
        for col in range(1, num_of_col):
            column_in_interest = b.iloc[:, col]
            accuracy_max = column_in_interest.max()
            index_max_list.append(column_in_interest[column_in_interest == accuracy_max].index[0]) #per possible label
            label_best_accuracy_list.append(accuracy_max)
            #print(index_max_list)
            #print(label_best_accuracy_list)
            #print('token')
            #print(col - 1)
            print(col, 'label per class best accuracy is', label_best_accuracy_list[col - 1], 'with parameter', b.iloc[index_max_list[col - 1] - (5 * power), 0])

def plot_3D_surface(df):
    x = list(df.iloc[:, 0])
    y = list(df.iloc[:, 1])
    label_per_class = input('How many labels per class?')
    z = df.iloc[:, int(label_per_class) + 1]
    for i in range(len(z)):
        z.iloc[i] = float(re.sub("[\(\[].*?[\)\]]", "", z.iloc[i]))
        #print(column_in_interest.iloc[i])
    z = list(z)
    ax = plt.axes(projection='3d')
    #ax.scatter(x, y, z, c = 'r', marker = 'o')
    ax.plot_trisurf(x, y, z, edgecolor='none')
    plt.show()
