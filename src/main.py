#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 22 Aug, 2022

#Run our 3 experiments on all datasets

#Dataset are returned in channels-last format
datasets = {
    'unimib' :  unimib_load_dataset,
    'twister' : e4_load_dataset,
    'uci har' : uci_har_load_dataset,
    'sussex huawei' : sh_loco_load_dataset
}