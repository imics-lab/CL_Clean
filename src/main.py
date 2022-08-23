#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 22 Aug, 2022

#Run our 3 experiments on all datasets

from cleanup import cleanup
from load_data_time_series.HAR.e4_wristband_Nov2019 import e4_load_dataset

CLEANUP = False

#Dataset are returned in channels-last format
datasets = {
    'unimib' :  unimib_load_dataset,
    'twister' : e4_load_dataset,
    'uci har' : uci_har_load_dataset,
    'sussex huawei' : sh_loco_load_dataset
}

if __name__ == '__main__':
    """
    Run each experiment on each dataset, so... 12?
    """

    for set in datasets.keys():
        print (f"###   Running {set} ### ")
        if CLEANUP:
            cleanup()