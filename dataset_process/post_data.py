import h5py
import csv
import numpy as np

frame = 500
hdf5_file_path_1 = './raw_dataset/500_RF_label.hdf5'
hdf5_file_path_2 = './raw_dataset/500_RF_channel.hdf5'

rf_file = h5py.File(hdf5_file_path_1,'r')
label_rf_raw = np.array(rf_file.get('lb'))
print("Channel Matrix shape is:", label_rf_raw.shape)

rfc_file = h5py.File(hdf5_file_path_2,'r')
rf_H_r_raw = np.array(rfc_file.get('H_real'))
rf_H_i_raw = np.array(rfc_file.get('H_imag'))
print("Channel Matrix shape is:", rf_H_r_raw.shape)


# print(label_rf_raw[1,0])
label_rf = np.reshape(label_rf_raw,(frame,))
# print(label_rf[500])

rf_H_r = np.reshape(rf_H_r_raw,(frame,32,4))
rf_H_i = np.reshape(rf_H_i_raw,(frame,32,4))

# print(rf_H_r)


rf_H_r_t = np.zeros((frame,3,32,4))
rf_H_i_t = np.zeros((frame,3,32,4))

for i in range (0,frame):
    if i == 0:
        rf_H_r_t[i,:,:,:] = np.tile(rf_H_r[i,:,:],(3,1,1))
        rf_H_i_t[i,:,:,:] = np.tile(rf_H_i[i,:,:],(3,1,1))
    elif i == 1:
        rf_H_r_t[i,:,:,:] = np.concatenate( (np.tile(rf_H_r[i-1,:,:],(2,1,1)), rf_H_r[i,np.newaxis,:,:] ), axis = 0)
        rf_H_i_t[i,:,:,:] = np.concatenate( (np.tile(rf_H_i[i-1,:,:],(2,1,1)), rf_H_i[i,np.newaxis,:,:] ), axis = 0)
    else:
        rf_H_r_t[i,:,:,:] = np.concatenate((rf_H_r[i,np.newaxis,:,:],rf_H_r[i-1,np.newaxis,:,:],rf_H_r[i-2,np.newaxis,:,:]),axis = 0)
        rf_H_i_t[i,:,:,:] = np.concatenate((rf_H_i[i,np.newaxis,:,:],rf_H_i[i-1,np.newaxis,:,:],rf_H_i[i-2,np.newaxis,:,:]),axis = 0)

H_r_train = np.delete(rf_H_r_t, np.arange(0, rf_H_r_t.shape[0], 10), axis=0)
H_i_train = np.delete(rf_H_i_t, np.arange(0, rf_H_i_t.shape[0], 10), axis=0)
label_train = np.delete(label_rf, np.arange(0, label_rf.shape[0], 10), axis=0)

H_r_test = rf_H_r_t[::10]
H_i_test = rf_H_i_t[::10]
label_test = label_rf[::10]


H_train = np.concatenate((H_r_train[:,:,:,:,np.newaxis],H_i_train[:,:,:,:,np.newaxis]), axis = 4)
H_test = np.concatenate((H_r_test[:,:,:,:,np.newaxis],H_i_test[:,:,:,:,np.newaxis]), axis = 4)



print(H_train)
print(label_train)


print(H_train.shape)
print(H_test.shape)







with h5py.File("./rdy_dataset/500_RF_train.hdf5", "w") as data_file:
    data_file.create_dataset("H_train", data=H_train)
    data_file.create_dataset("label_train", data=label_train)

with h5py.File("./rdy_dataset/500_RF_test.hdf5", "w") as data_file:
    data_file.create_dataset("H_test", data=H_test)
    data_file.create_dataset("label_test", data=label_test)