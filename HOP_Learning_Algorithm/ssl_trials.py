#ssl_trials.py
#
#This script shows how to use the ssl_trials
#function to run many trials of random test/train splits
#to compare various SSL algorithms. The code automatically
#generates LaTeX tables and plots (the files SSL_MNIST.png/pdf 
#that are included in the examples folder).
#
#The code also shows how to write a new SSL algorithm 'alg_name' and 
#plug it into the ssl_trials environment. The code will look for a file
#alg_name.py and import and run the function 'ssl' in this file. 
#See alg_name.py for details.
#
#ssl_trials supports parallel processing, via num_cores=

import graphlearning as gl
import numpy as np
import os

dataset = 'MNIST'
metric = 'vae' #Uses variational autoencoder to consruct graph
num_classes = 10
algorithm_list = ['high_order_poisson']
results_files = []
legend_list = []
#'laplace','poisson','nearestneighbor','volumembo'
#Create a new label permutation, with 100 randomized trials at 1,2,4,8,16 labels per class
gl.create_label_permutations(gl.load_labels(dataset),100,[1,2,3,4,5],dataset='MNIST',name='new',overwrite=True)

#Parameters specific to the new algorithm alg_name
for i in np.arange(0.1, 5.1, 0.1):
    results_files_for_graphs = []
    legend_list_for_graphs = []
    for j in range(50, 350, 50):
        params = {'power': i, 'N' : j}
#'vals' : vals, 'vecs': vecs
#Run experiments (we'll just do t=4 trials to save time)
        for alg in algorithm_list:
            results = gl.ssl_trials(dataset=dataset,metric=metric,algorithm=alg,num_cores=3,t=500,params=params, require_eigen_data = True)
            f_old = os.path.join('Results', results + '_accuracy.csv')
            #print(f_old)
            f_new =  os.path.join('Results', results + '_N%d_m%.2f'%(j,i) + '_accuracy.csv')
            f_new_table = results + '_N%d_m%.2f'%(j,i)
            #print(f_new)
            os.rename(f_old,f_new)
            results_files.append(f_new_table)
            legend_list.append('High order Poisson learning' + 'N = ' + str(j) + 'm = ' + str(i))
            #results_files.append(results)
#label_perm='new',
#Generate plots
        results_files_for_graphs.append(f_new_table)
        legend_list_for_graphs.append('High order Poisson learning' + 'N = ' + str(j) + 'm = ' + str(i))
        graph_name = 'SSL_MNIST_' + 'm_' + str(i) + '.png'
    gl.accuracy_plot(results_files_for_graphs,legend_list_for_graphs,num_classes,title='SSL Comparison: MNIST',errorbars=False,testerror=False,loglog=False,savefile=graph_name)
#legend_list = ['High order laplace learning']
#'Laplace learning','Poisson learning','Nearest Neighbor','VolumeMBO',
#gl.accuracy_plot(results_files,legend_list,num_classes,title='SSL Comparison: MNIST',errorbars=False,testerror=False,loglog=False,savefile='SSL_MNIST.png')

#Generate a table showing accuracy scores
gl.accuracy_table_icml(results_files,legend_list,num_classes,savefile='SSL_MNIST.tex',title="SSL Comparison: MNIST",quantile=False,append=False)
os.system("pdflatex SSL_MNIST.tex")

