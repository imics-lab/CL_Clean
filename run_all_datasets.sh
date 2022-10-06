#!/usr/bin/env bash

#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 06 October, 2021
#Run experiments 1-3 on all datasets

# datasets = {
#     'synthetic' : load_synthetic_dataset,
#     'unimib' :  unimib_load_dataset,
#     'twister' : e4_load_dataset,
#     'uci har' : uci_har_load_dataset,
#     'sussex huawei' : sh_loco_load_dataset
# }

echo "Synthetic"
python3 -Wignore src/main.py --set 'synthetic' > logs/syn_log.txt

echo "UniMiB"
python3 -Wignore src/main.py --set 'unimib' > logs/uni_log.txt

echo "Twister"
python3 -Wignore src/main.py --set 'twister' > logs/twis_log.txt

echo "UCI HAR"
python3 -Wignore src/main.py --set 'uci har' > logs/uci_log.txt

echo "Locomotion"
python3 -Wignore src/main.py --set 'sussex huawei' > logs/ssx_log.txt

