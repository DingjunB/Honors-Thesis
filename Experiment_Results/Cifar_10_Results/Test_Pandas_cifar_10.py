import astropy as asp
import numpy as np
from astropy.table import Table
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import re

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import rcParams

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None

tab = Table.read('SSL_MNIST.tex').to_pandas()
df = pd.read_csv('table_ex.csv')
df_full = pd.read_csv('table_ex_full.csv')

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
        block = tab.iloc[(j * 6):(j * 6 + 6), :]
        block_list.append(block)
    for i in range(6): #for our purpose, we won't generalize this yet
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
        block = tab.iloc[(j * 6):(j * 6 + 6), :]
        block_list.append(block)
    for i in range(6): #for our purpose, we won't generalize this yet
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
            print(col, 'label per class best accuracy is', label_best_accuracy_list[col - 1], 'with parameter', b.iloc[index_max_list[col - 1] - (6 * power), 0])

def plot_3D_surface(df):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = list(df.iloc[:, 0])
    y = list(df.iloc[:, 1])
    #X, Y = np.meshgrid(x, y)
    label_per_class = input('How many labels per class?')
    z = df.iloc[:, int(label_per_class) + 1]
    for i in range(len(z)):
        z.iloc[i] = float(re.sub("[\(\[].*?[\)\]]", "", z.iloc[i]))
        #print(column_in_interest.iloc[i])
    z = list(z)
    #ax = plt.axes(projection='3d')
    #ax.scatter(x, y, z, c = 'r', marker = 'o')
    my_cmap = plt.get_cmap('gist_heat')
    trisurf = ax.plot_trisurf(x, y, z, cmap = my_cmap, edgecolor='none')
    fig.colorbar(trisurf, ax = ax, shrink = 0.5, aspect = 5)
    
    ax.set_title('Accuracy Score with ' + label_per_class + ' labeled data in Cifar-10')
    ax.set_xlabel('Spectral Cutoff N', fontweight ='bold') 
    ax.set_ylabel('Laplacian Order m', fontweight ='bold') 
    ax.set_zlabel('Accuracy Score', fontweight ='bold')
    #surf = ax.plot_surface(X, Y, np.array(z), cmap=cm.coolwarm,
                       #linewidth=0, antialiased=False)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plot_contour_surface(df):
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    x = list(df.iloc[:, 0])
    y = list(df.iloc[:, 1])
    #X, Y = np.meshgrid(x, y)
    label_per_class = input('How many labels per class?')
    z = df.iloc[:, int(label_per_class) + 1]
    for i in range(len(z)):
        z.iloc[i] = float(re.sub("[\(\[].*?[\)\]]", "", z.iloc[i]))
        #print(column_in_interest.iloc[i])
    z = list(z)
    data_construct = {'x':x, 'y':y, 'z':z}
    data_frame = pd.DataFrame(data = data_construct)
    Z = data_frame.pivot_table(index='x', columns='y', values='z').T.values
    X_unique = np.sort(data_frame.x.unique())
    Y_unique = np.sort(data_frame.y.unique())
    X, Y = np.meshgrid(X_unique, Y_unique)
    # Initialize plot objects
    rcParams['figure.figsize'] = 5, 5 # sets plot size
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Generate a contour plot
    #cp = ax.contour(X, Y, Z)
    cpf = ax.contourf(X,Y,Z, cmap=cm.Reds)
    line_colors = ['black' for l in cpf.levels]
    cp = ax.contour(X, Y, Z, colors=line_colors)
    ax.clabel(cp, fontsize=7, colors=line_colors, fmt = '%1.0f', manual = True)
    ax.set_xlabel('Spectral Cutoff N', fontweight ='bold')
    ax.set_ylabel('Laplacian Order m', fontweight ='bold')
    ax.set_title('Contour Map of Accuracy Score with ' + label_per_class + ' labeled data in Cifar-10')
    #Z = np.tile(np.array(z), (X.shape[1], 1))
    
    #my_cmap = plt.get_cmap('gist_heat')
    #trisurf = ax.plot_trisurf(x, y, z, cmap = my_cmap, edgecolor='none')
    #fig.colorbar(trisurf, ax = ax, shrink = 0.5, aspect = 5)
    
    #ax.set_title('Accuracy Score with ' + label_per_class + ' labeled data in MNIST')
    #ax.set_xlabel('Spectral Cutoff N', fontweight ='bold') 
    #ax.set_ylabel('Laplacian Order m', fontweight ='bold') 
    #ax.set_zlabel('Accuracy Score', fontweight ='bold')

    plt.show()
