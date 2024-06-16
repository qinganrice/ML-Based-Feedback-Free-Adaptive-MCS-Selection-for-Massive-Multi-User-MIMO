import h5py
import numpy as np
import torch



def get_training_set():
    hdf5_file_path_1 = './channel_label/rdy_dataset/500_UMi_NLoS_train.hdf5'
    hdf5_file_path_3 = './channel_label/rdy_dataset/500_UMa_NLoS_train.hdf5'
    hdf5_file_path_5 = './channel_label/rdy_dataset/500_UMa_LoS_train.hdf5'
    hdf5_file_path_7 = './channel_label/rdy_dataset/1000_UMi_LoS_train.hdf5'
    hdf5_file_path_9 = './channel_label/rdy_dataset/500_RF_train.hdf5'

    rf_file = h5py.File(hdf5_file_path_9,'r')
    rf_tr_lb = np.array(rf_file.get('label_train'))
    rf_tr_H = np.array(rf_file.get('H_train'))
    # print("Label shape is:", rf_tr_lb.shape)
    # print("Channel Matrix shape is:", rf_tr_H.shape)

    il_file = h5py.File(hdf5_file_path_7,'r')
    il_tr_lb = np.array(il_file.get('label_train'))
    il_tr_H = np.array(il_file.get('H_train'))
    # print("Label shape is:", il_tr_lb.shape)
    # print("Channel Matrix shape is:", il_tr_H.shape)

    in_file = h5py.File(hdf5_file_path_1,'r')
    in_tr_lb = np.array(in_file.get('label_train'))
    in_tr_H = np.array(in_file.get('H_train'))
    # print("Label shape is:", in_tr_lb.shape)
    # print("Channel Matrix shape is:", in_tr_H.shape)

    al_file = h5py.File(hdf5_file_path_5,'r')
    al_tr_lb = np.array(al_file.get('label_train'))
    al_tr_H = np.array(al_file.get('H_train'))
    # print("Label shape is:", al_tr_lb.shape)
    # print("Channel Matrix shape is:", al_tr_H.shape)

    an_file = h5py.File(hdf5_file_path_3,'r')
    an_tr_lb = np.array(an_file.get('label_train'))
    an_tr_H = np.array(an_file.get('H_train'))
    # print("Label shape is:", an_tr_lb.shape)
    # print("Channel Matrix shape is:", an_tr_H.shape)

    H_tr = np.concatenate((rf_tr_H, il_tr_H, in_tr_H, al_tr_H, an_tr_H), axis = 0)
    lb_tr = np.concatenate((rf_tr_lb, il_tr_lb, in_tr_lb, al_tr_lb, an_tr_lb), axis = 0)

    H_tr = np.reshape(H_tr,(13950,3,2,32,4))

    print("Training Label shape is:", lb_tr.shape)
    print("Training Channel Matrix shape is:", H_tr.shape)

    # handle numpy array
    H_tr = torch.from_numpy(H_tr)
    # corr_tr = torch.from_numpy(corr_tr)
    lb_tr = torch.from_numpy(lb_tr)

    training_data  = torch.utils.data.TensorDataset(H_tr.float(), lb_tr)

    # backward compatibility
    return training_data

def get_validation_set():
    
    hdf5_file_path_2 = './channel_label/rdy_dataset/500_UMi_NLoS_test.hdf5'
    hdf5_file_path_4 = './channel_label/rdy_dataset/500_UMa_NLoS_test.hdf5'
    hdf5_file_path_6 = './channel_label/rdy_dataset/500_UMa_LoS_test.hdf5'
    hdf5_file_path_8 = './channel_label/rdy_dataset/1000_UMi_LoS_test.hdf5'
    hdf5_file_path_0 = './channel_label/rdy_dataset/500_RF_test.hdf5'
    
    rf_file = h5py.File(hdf5_file_path_0,'r')
    rf_tt_lb = np.array(rf_file.get('label_test'))
    rf_tt_H = np.array(rf_file.get('H_test'))
    print("Label shape is:", rf_tt_lb.shape)
    print("Channel Matrix shape is:", rf_tt_H.shape)

    il_file = h5py.File(hdf5_file_path_8,'r')
    il_tt_lb = np.array(il_file.get('label_test'))
    il_tt_H = np.array(il_file.get('H_test'))
    print("Label shape is:", il_tt_lb.shape)
    print("Channel Matrix shape is:", il_tt_H.shape)


    in_file = h5py.File(hdf5_file_path_2,'r')
    in_tt_lb = np.array(in_file.get('label_test'))
    in_tt_H = np.array(in_file.get('H_test'))
    print("Label shape is:", in_tt_lb.shape)
    print("Channel Matrix shape is:", in_tt_H.shape)

    al_file = h5py.File(hdf5_file_path_6,'r')
    al_tt_lb = np.array(al_file.get('label_test'))
    al_tt_H = np.array(al_file.get('H_test'))
    print("Label shape is:", al_tt_lb.shape)
    print("Channel Matrix shape is:", al_tt_H.shape)

    an_file = h5py.File(hdf5_file_path_4,'r')
    an_tt_lb = np.array(an_file.get('label_test'))
    an_tt_H = np.array(an_file.get('H_test'))
    print("Label shape is:", an_tt_lb.shape)
    print("Channel Matrix shape is:", an_tt_H.shape)


    H_tt = np.concatenate((rf_tt_H, il_tt_H, in_tt_H, al_tt_H, an_tt_H), axis = 0)
    lb_tt = np.concatenate((rf_tt_lb, il_tt_lb, in_tt_lb, al_tt_lb, an_tt_lb), axis = 0)

    # ones_arr = np.ones((1600, 3, 32, 4, 1))
    print(H_tt.shape, lb_tt.shape)
    H_tt = np.reshape(H_tt,(1550,3,2,32,4))
    # H_tt = np.squeeze(H_tt[:,2,:,:,:])

    # corr_tt = np.zeros((1600,4,4)) 
    # for i in range(1600):
    #     for j in range(4):
    #         for k in range(4):
    #             H_1 = H_tt[i,0,:,j] + H_tt[i,1,:,j] * 1j
    #             H_2 = H_tt[i,0,:,k] + H_tt[i,1,:,k] * 1j
    #             corr_tt[i,j,k] = abs(np.dot(H_1.conj().T, H_2)) / (np.linalg.norm(H_1) * np.linalg.norm(H_2))
    #             # print(corr_tt[i,j,k])

    print("Test Label shape is:", lb_tt.shape)
    print("Test Channel Matrix shape is:", H_tt.shape)
    # handle numpy array
    H_tt = torch.from_numpy(H_tt)
    # corr_tt = torch.from_numpy(corr_tt)
    lb_tt = torch.from_numpy(lb_tt)
    # backward compatibility

    validation_data  = torch.utils.data.TensorDataset(H_tt.float(), lb_tt)
    return validation_data