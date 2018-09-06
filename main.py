# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:16:17 2017

@author: Vicky
"""


import numpy as np
from sklearn.cluster import KMeans


#Student Information
print('UBitName = vvasanth')
print('personNumber = 50248708')

print('UBitName = rajvinod')
print('personNumber = 50247214')

print('UBitName = ss623')
print('personNumber = 50247317')

#importing the letor dataset
letor_input_data = np.genfromtxt('datafiles/Querylevelnorm_X.csv', delimiter=',')
letor_output_data = np.genfromtxt('datafiles/Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])


#partitioning the data set into training, validation and test sets
letor_input_data_columns=letor_input_data.shape[1]
train_rows=round(letor_input_data.shape[0]*0.8)
test_validation_rows=round(letor_input_data.shape[0]*0.1)
train_x=letor_input_data[:train_rows+1,:letor_input_data_columns]
validation_x=letor_input_data[train_rows+1:train_rows+1+test_validation_rows,:letor_input_data_columns]
test_x=letor_input_data[train_rows+1+test_validation_rows:,:letor_input_data_columns]
train_y=letor_output_data[:train_rows+1,:letor_input_data_columns]

validation_y=letor_output_data[train_rows+1:train_rows+1+test_validation_rows,:letor_input_data_columns]
test_y=letor_output_data[train_rows+1+test_validation_rows:,:letor_input_data_columns]

#applying k-means on the letor dataset
kmeans=KMeans(n_clusters=4, random_state=0).fit(train_x)
cluster_0=[]
cluster_1=[]
cluster_2=[]
cluster_3=[]

#design matrix function
def compute_design_matrix(X, centers, spreads):
# use broadcast
    basis_func_outputs = np.exp(
            np.sum(
                    np.matmul(X - centers, spreads) * (X - centers),
                    axis=2
                    ) / (-2)
            ).T
# insert ones to the 1st col
    return np.insert(basis_func_outputs, 0, 1, axis=1)

for i in range(0,train_x.shape[0]):
    eval('cluster_'+str(kmeans.labels_[i])).append(train_x[i])

#calculating the spreads and centers
cov_cluster_0=np.linalg.pinv(np.cov(np.array(cluster_0).T))       
cov_cluster_1=np.linalg.pinv(np.cov(np.array(cluster_1).T))
cov_cluster_2=np.linalg.pinv(np.cov(np.array(cluster_2).T))
cov_cluster_3=np.linalg.pinv(np.cov(np.array(cluster_3).T))
centers=kmeans.cluster_centers_
centers=centers[:, np.newaxis, :]
spreads=np.array([cov_cluster_0,cov_cluster_1,cov_cluster_2,cov_cluster_3])
X=train_x[np.newaxis, :, :]
design_matrix=compute_design_matrix(X,centers,spreads)

def closed_form_sol(L2_lambda, design_matrix, output_data):
    return np.linalg.solve(
            L2_lambda * np.identity(design_matrix.shape[1]) +
            np.matmul(design_matrix.T, design_matrix),
            np.matmul(design_matrix.T, output_data)
            ).flatten()
    
closed_form_solution=np.reshape(closed_form_sol(0.1,design_matrix,train_y),(1,5))
print("LETOR_closed_form_solution:"+str(closed_form_solution))
#final_val=np.matmul(closed_form_solution,np.reshape(design_matrix[7],(5,1)))

#Applying k-means on validation data set
cluster_val_0=[]
cluster_val_1=[]
cluster_val_2=[]
cluster_val_3=[]

kmeans_val=KMeans(n_clusters=4, random_state=0).fit(validation_x)
for i in range(0,6962):
    eval('cluster_val_'+str(kmeans_val.labels_[i])).append(validation_x[i])

#calculating the spreads and centers
cov_cluster_val_0=np.linalg.pinv(np.cov(np.array(cluster_val_0).T))       
cov_cluster_val_1=np.linalg.pinv(np.cov(np.array(cluster_val_1).T))
cov_cluster_val_2=np.linalg.pinv(np.cov(np.array(cluster_val_2).T))
cov_cluster_val_3=np.linalg.pinv(np.cov(np.array(cluster_val_3).T))

centers_val=kmeans_val.cluster_centers_
centers_val=centers_val[:, np.newaxis, :]
spreads_val=np.array([cov_cluster_val_0,cov_cluster_val_1,cov_cluster_val_2,cov_cluster_val_3])
X_val=validation_x[np.newaxis, :, :]
design_matrix_val=compute_design_matrix(X_val,centers_val,spreads_val)
#final_val=np.matmul(closed_form_solution,np.reshape(design_matrix_val[105],(5,1)))
design_matrix_val=np.matrix(design_matrix_val)

#applying k-means to test data
cluster_test_0=[]
cluster_test_1=[]
cluster_test_2=[]
cluster_test_3=[]

kmeans_test=KMeans(n_clusters=4, random_state=0).fit(test_x)
for i in range(0,test_x.shape[0]):
    eval('cluster_test_'+str(kmeans_test.labels_[i])).append(test_x[i])
    
cov_cluster_test_0=np.linalg.pinv(np.cov(np.array(cluster_test_0).T))       
cov_cluster_test_1=np.linalg.pinv(np.cov(np.array(cluster_test_1).T))
cov_cluster_test_2=np.linalg.pinv(np.cov(np.array(cluster_test_2).T))
cov_cluster_test_3=np.linalg.pinv(np.cov(np.array(cluster_test_3).T))

#calculating spreads and centers
centers_test=kmeans_test.cluster_centers_
centers_test=centers_test[:, np.newaxis, :]
spreads_test=np.array([cov_cluster_test_0,cov_cluster_test_1,cov_cluster_test_2,cov_cluster_test_3])
X_test=test_x[np.newaxis, :, :]
design_matrix_test=compute_design_matrix(X_test,centers_test,spreads_test)
#final_val=np.matmul(closed_form_solution,np.reshape(design_matrix_val[105],(5,1)))
design_matrix_val=np.matrix(design_matrix_test)

    
#computing the error value
def err_func(weights,design_matrix_validation,val_y):
    predicted_values=np.zeros(design_matrix_validation.shape[0])
    for i in range(0,design_matrix_validation.shape[0]):
        for j in range(0,design_matrix_validation.shape[1]):
            predicted_values[i]=predicted_values[i]+design_matrix_validation[i,j]*weights[j]
    predicted_values=np.matrix(predicted_values)
    predicted_values=predicted_values.T
    err_value_temp=np.zeros(val_y.shape[0])
    for i in range(0,val_y.shape[0]):
        err_value_temp[i]=((val_y[i]-predicted_values[i])**2)
        
    err_value=np.sum(err_value_temp);
    err_value=err_value/2;
    err_value=(err_value*2)/val_y.shape[0];
    err_value=err_value**(0.5);
    return err_value;

print("LETOR_closed_form_E_RMS-train:"+str(err_func(closed_form_solution.T,design_matrix,train_y)))
print("LETOR_closed_form_E_RMS-validation:"+str(err_func(closed_form_solution.T,design_matrix_val,validation_y)))
print("LETOR_closed_form_E_RMS-test:"+str(err_func(closed_form_solution.T,design_matrix_test,test_y)))

#importing the synthetic dataset
syn_input_data = np.loadtxt('datafiles/input.csv', delimiter=',')
syn_output_data = np.loadtxt('datafiles/output.csv', delimiter=',').reshape([-1, 1])

#partitioning the data set into training, validation and test sets
#syn_train_x,syn_test_x,syn_train_y,syn_test_y=train_test_split(syn_input_data,syn_output_data,test_size=0.1)
#syn_train_x,syn_val_x,syn_train_y,syn_val_y=train_test_split(syn_train_x,syn_train_y,test_size=0.1)

syn_input_data_columns=syn_input_data.shape[1]
syn_train_rows=round(syn_input_data.shape[0]*0.8)
syn_test_val_rows=round(syn_input_data.shape[0]*0.1)
syn_train_x=syn_input_data[:syn_train_rows+1,:syn_input_data_columns]
syn_val_x=syn_input_data[syn_train_rows+1:syn_train_rows+1+syn_test_val_rows,:syn_input_data_columns]
syn_test_x=syn_input_data[syn_train_rows+1+syn_test_val_rows:,:syn_input_data_columns]
syn_train_y=syn_output_data[:syn_train_rows+1,:syn_input_data_columns]

syn_val_y=syn_output_data[syn_train_rows+1:syn_train_rows+1+syn_test_val_rows,:syn_input_data_columns]
syn_test_y=syn_output_data[syn_train_rows+1+syn_test_val_rows:,:syn_input_data_columns]



#applying kmeans on the synthetic dataset
kmeans_syn=KMeans(n_clusters=4, random_state=0).fit(syn_train_x)

#initializing the clusters
syn_cluster_0=[]
syn_cluster_1=[]
syn_cluster_2=[]
syn_cluster_3=[]

#Grouping each point according to its cluster
for i in range(0,syn_train_x.shape[0]):
    eval('syn_cluster_'+str(kmeans_syn.labels_[i])).append(syn_train_x[i])
    
'''
def compute_design_matrix(X, centers, spreads):
# use broadcast
    basis_func_outputs = np.exp(
            np.sum(
                    np.matmul(X - centers, spreads) * (X - centers),
                    axis=2
                    ) / (-2)
            ).T
# insert ones to the 1st col
    return np.insert(basis_func_outputs, 0, 1, axis=1)
'''
#calculating the spreads
syn_cov_cluster_0=np.linalg.pinv(np.cov(np.array(syn_cluster_0).T))       
syn_cov_cluster_1=np.linalg.pinv(np.cov(np.array(syn_cluster_1).T))
syn_cov_cluster_2=np.linalg.pinv(np.cov(np.array(syn_cluster_2).T))
syn_cov_cluster_3=np.linalg.pinv(np.cov(np.array(syn_cluster_3).T))

#calculating centers and spreads
syn_centers=kmeans_syn.cluster_centers_
syn_centers=syn_centers[:, np.newaxis, :]
syn_spreads=np.array([syn_cov_cluster_0,syn_cov_cluster_1,syn_cov_cluster_2,syn_cov_cluster_3])
syn_X=syn_train_x[np.newaxis, :, :]

#calculating the design matrix for the training dataset
syn_design_matrix=compute_design_matrix(syn_X,syn_centers,syn_spreads)
'''
def closed_form_sol(L2_lambda, design_matrix, output_data):
    return np.linalg.solve(
            L2_lambda * np.identity(design_matrix.shape[1]) +
            np.matmul(design_matrix.T, design_matrix),
            np.matmul(design_matrix.T, output_data)
            ).flatten()
'''
#find the closed form solution
syn_closed_form_solution=np.reshape(closed_form_sol(0.1,syn_design_matrix,syn_train_y),(1,5))
print("synthetic_closed_form_solution:"+str(syn_closed_form_solution))
#syn_predicted_value=np.matmul(syn_closed_form_solution,np.reshape(syn_design_matrix[7],(5,1)))
#print(syn_predicted_value)

#Applying k-means
kmeans_val_syn=KMeans(n_clusters=4, random_state=0).fit(syn_val_x)
syn_cluster_val_0=[]
syn_cluster_val_1=[]
syn_cluster_val_2=[]
syn_cluster_val_3=[]

for i in range(0,syn_val_x.shape[0]):
    eval('syn_cluster_val_'+str(kmeans_val_syn.labels_[i])).append(syn_val_x[i])
    
#calculating the spreads and centers
syn_cov_cluster_val_0=np.linalg.pinv(np.cov(np.array(syn_cluster_val_0).T))       
syn_cov_cluster_val_1=np.linalg.pinv(np.cov(np.array(syn_cluster_val_1).T))
syn_cov_cluster_val_2=np.linalg.pinv(np.cov(np.array(syn_cluster_val_2).T))
syn_cov_cluster_val_3=np.linalg.pinv(np.cov(np.array(syn_cluster_val_3).T))

syn_centers_val=kmeans_val_syn.cluster_centers_
syn_centers_val=syn_centers_val[:, np.newaxis, :]
syn_spreads_val=np.array([syn_cov_cluster_val_0,syn_cov_cluster_val_1,syn_cov_cluster_val_2,syn_cov_cluster_val_3])

#calculating the design matrix for the validation data set
syn_design_matrix_val=compute_design_matrix(syn_val_x,syn_centers_val,syn_spreads_val)

#Applying k-means
kmeans_test_syn=KMeans(n_clusters=4, random_state=0).fit(syn_test_x)
syn_cluster_test_0=[]
syn_cluster_test_1=[]
syn_cluster_test_2=[]
syn_cluster_test_3=[]

for i in range(0,syn_test_x.shape[0]):
    eval('syn_cluster_test_'+str(kmeans_test_syn.labels_[i])).append(syn_test_x[i])
    
#calculating the spreads and centers
syn_cov_cluster_test_0=np.linalg.pinv(np.cov(np.array(syn_cluster_test_0).T))       
syn_cov_cluster_test_1=np.linalg.pinv(np.cov(np.array(syn_cluster_test_1).T))
syn_cov_cluster_test_2=np.linalg.pinv(np.cov(np.array(syn_cluster_test_2).T))
syn_cov_cluster_test_3=np.linalg.pinv(np.cov(np.array(syn_cluster_test_3).T))

syn_centers_test=kmeans_test_syn.cluster_centers_
syn_centers_test=syn_centers_test[:, np.newaxis, :]
syn_spreads_test=np.array([syn_cov_cluster_test_0,syn_cov_cluster_test_1,syn_cov_cluster_test_2,syn_cov_cluster_test_3])

#calculating the design matrix for the validation data set
syn_design_matrix_test=compute_design_matrix(syn_test_x,syn_centers_test,syn_spreads_test)
#syn_final_val=np.matmul(syn_closed_form_solution,np.reshape(syn_design_matrix_val[15],(5,1)))


print("synthetic_closed_form_E_RMS-train:"+str(err_func(syn_closed_form_solution.T,syn_design_matrix,syn_train_y)))
print("synthetic_closed_form_E_RMS-validation:"+str(err_func(syn_closed_form_solution.T,syn_design_matrix_val,syn_val_y)))
print("synthetic_closed_form_E_RMS-test:"+str(err_func(syn_closed_form_solution.T,design_matrix_test,syn_test_y)))

#print(err_func(syn_closed_form_solution.T,syn_design_matrix_val,syn_val_y))
#print(err_func(syn_closed_form_solution.T,syn_design_matrix_val,syn_test_y))
#print(err_func(syn_closed_form_solution.T,syn_design_matrix,syn_train_y))

'''
def err_func2(design_matrix,weights,y):
    return np.matmul((np.matmul(weights,design_matrix)-y).T,design_matrix)

syn_closed_form_solution=np.matrix(syn_closed_form_solution)
syn_train_y=np.matrix(syn_train_y)
syn_design_matrix=np.matrix(syn_design_matrix)
#print(err_func2(syn_design_matrix,syn_closed_form_solution,syn_train_y))
print(syn_design_matrix.T.shape)
print(syn_design_matrix.T)

E_D=0.5*np.sum((syn_train_y.T-np.matmul(syn_closed_form_solution,syn_design_matrix.T))**2)
print(E_D)
'''

def SGD_sol_LETOR(learning_rate,minibatch_size,num_epochs,L2_lambda,design_matrix,output_data):
    N, _ = design_matrix.shape
    prev=0.6
    p=5;s=0.5644
    weights = np.zeros([1, 5])
    w=np.zeros([1, 5])
    for epoch in range(num_epochs):
        for i in range(int(N / minibatch_size)):
            lower_bound = i * minibatch_size
            upper_bound = min((i+1)*minibatch_size, N)
            Phi = design_matrix[lower_bound : upper_bound, :]
            t = output_data[lower_bound : upper_bound, :]
            E_D = np.matmul(
                    (np.matmul(Phi, weights.T)-t).T,
                    Phi
                    )
            E = (E_D + L2_lambda * weights) / minibatch_size
            weights = weights - learning_rate * E
            ee=err_func(weights.T,design_matrix,output_data)
            if ee<prev:
                if(ee<s):
                    w=weights;
                    return w.flatten();
                prev=ee;
                w=weights;
                
            else:
                p=p-1;
                if p==0:
                    #print(epoch);
                    return w.flatten();
        #print np.linalg.norm(E)
    return weights.flatten()

def SGD_sol(learning_rate,minibatch_size,num_epochs,L2_lambda,design_matrix,output_data):
    N, _ = design_matrix.shape
    weights = np.zeros([1, 5])
    for epoch in range(num_epochs):
        for i in range(int(N / minibatch_size)):
            lower_bound = i * minibatch_size
            upper_bound = min((i+1)*minibatch_size, N)
            Phi = design_matrix[lower_bound : upper_bound, :]
            t = output_data[lower_bound : upper_bound, :]
            E_D = np.matmul(
                    (np.matmul(Phi, weights.T)-t).T,
                    Phi
                    )
            E = (E_D + L2_lambda * weights) / minibatch_size
            weights = weights - learning_rate * E
        #print np.linalg.norm(E)
    return weights.flatten()

letor_sgd_sol=SGD_sol_LETOR(1.9,design_matrix.shape[0],10000,0.1,design_matrix,train_y);
#print(letor_sgd_sol)
print("LETOR_stochastic_gradient_descent_solution:"+str(letor_sgd_sol))

print("LETOR_stochastic_gradient_descent_E_RMS-train:"+str(err_func(letor_sgd_sol.T,design_matrix,train_y)))
print("LETOR_stochastic_gradient_descent_E_RMS-validation:"+str(err_func(letor_sgd_sol.T,design_matrix_val,validation_y)))
print("LETOR_stochastic_gradient_descent_E_RMS-test:"+str(err_func(letor_sgd_sol.T,design_matrix_test,test_y)))

syn_sgd_sol=SGD_sol(0.5,syn_design_matrix.shape[0],10000,0.1,syn_design_matrix,syn_train_y);
print("synthetic_stochastic_gradient_descent_solution:"+str(syn_sgd_sol))


print("synthetic_stochastic_gradient_descent_E_RMS-train:"+str(err_func(syn_sgd_sol.T,syn_design_matrix,syn_train_y)))
print("synthetic_stochastic_gradient_descent_E_RMS-validation:"+str(err_func(syn_sgd_sol.T,syn_design_matrix_val,syn_val_y)))
print("synthetic_stochastic_gradient_descent_E_RMS-test:"+str(err_func(syn_sgd_sol.T,design_matrix_test,syn_test_y)))


    
    
    


