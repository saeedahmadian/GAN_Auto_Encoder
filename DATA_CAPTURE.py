import pandas as pd
import numpy as np

# Assign spreadsheet filename to `file`
real_fname = 'real_data.xlsx'
desired_fname = 'desired_data.xlsx'
num_batch = 20
num_features = 96
num_nodes = 3

# Load spreadsheet
x1 = pd.read_excel(real_fname, header=None).values
x2 = pd.read_excel(desired_fname, header=None).values


real_data=np.zeros([num_batch,num_nodes,num_features,1],dtype='float32')

#for i in range(len(x1[0])):
#for j in range(0, len(x1), num_features):
    #temp=x1[j:j+num_features-1,:]




# reshape for convolution
real_data = np.reshape(x1, (-1, num_features, num_nodes))
desired_data = np.reshape(x2, (-1, num_features, num_nodes))

real_data = np.rollaxis(real_data, -1, 1)
desired_data = np.rollaxis(desired_data, -1, 1)

real_data=np.expand_dims(real_data,axis=-1)
desired_data=np.expand_dims(desired_data,axis=-1)

np.save('real_data',real_data)
np.save('desired_data',desired_data)
a=1