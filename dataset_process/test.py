import h5py
import csv
import numpy as np

hdf5_file_path_7 = './rdy_dataset/1000_UMi_LoS_train.hdf5'

il_file = h5py.File(hdf5_file_path_7,'r')
il_tr_lb = np.array(il_file.get('label_train'))
H_tr = np.array(il_file.get('H_train'))

print(H_tr)

H_tr = np.reshape(H_tr,(5400,3,2,32,4))
H_tr = np.squeeze(H_tr[:,2,:,:,:])


corr_tr = np.zeros((5400,4,4)) 
    #[13900,2,32,4]
for i in range(5400):
    for j in range(4):
        for k in range(4):
            H_1 = H_tr[i,0,:,j] + H_tr[i,1,:,j] * 1j
            H_2 = H_tr[i,0,:,k] + H_tr[i,1,:,k] * 1j
            corr_tr[i,j,k] = abs(np.dot(H_1.conj().T, H_2)) / (np.linalg.norm(H_1) * np.linalg.norm(H_2))
            # print(corr_tr[i,j,k])
