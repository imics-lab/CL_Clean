#Author: Gentry Atkinson
#Organization: Texas University
#Data: 10 August, 2022
#Placeholder data loader for SH Locomotion

import numpy as np
from sklearn.model_selection import train_test_split

DOWN_SAMPLE = True

if DOWN_SAMPLE:
    path = 'src/data/Sussex_Huawei_DS/'
else:
    path = 'src/data/Sussex_Huawei/'

def sh_loco_load_dataset(incl_xyz_accel=True, incl_rms_accel=False, incl_val_group=False):
    X = np.load(path+'x_train.npy')
    if not incl_rms_accel:
        X = np.delete(X, 3, 2)
    y = np.load(path+'y_train.npy')
    X_test = np.load(path+'x_test.npy')
    if not incl_rms_accel:
        X_test = np.delete(X_test, 3, 2)
    y_test = np.load(path+'y_test.npy')
    if not incl_val_group:
        return  X, y, X_test, y_test
    else: 
        X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=1899)
        return X, y, X_val, y_val, X_test, y_test