# CL Clean
Contrastive Learning for Label Noise Recognition

## Introduction
Contrastive learning has been pushing the benchmarks for representation learning without the use of (reliable) labels. 
This project will leverage the power of these approaches to automatically flag instances in time-series dataset that
are the most likely to be mislabeled.

## Sources
Contrastive Learning frameworks: https://github.com/tian0426/cl-har

4 real world datasets have been used in this work:
  -UniMiB SHAR
  -UCI HAR
  -TWristAR
  -Sussex-HuaWei Locomotion
Data loaders for these datasets are provided in src/load_data_time_series

## Use

To run the 4 provided experiments on one dataset use:

  python3 main.py --set [dataset name]

The supported names are:
  -synthetic
  -unimib
  -uci har
  -sussex huawei
