#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 10 Apr, 2023

import pandas

if __name__ == '__main__':
    #results/exp4_results_unimib_2023-04-07 16:25:30.220737.csv
    #results/exp4_results_uci har_2023-04-07 17:01:49.144810.csv
    #results/exp4_results_twister_2023-04-07 16:19:35.195200.csv
    #results/exp4_results_sussex huawei_2023-04-07 20:40:49.757283.csv
    #results/exp4_results_sussex huawei_2023-04-10 17:44:45.729410.csv

    df = pandas.read_csv('results/exp4_results_sussex huawei_2023-04-10 17:44:45.729410.csv')
    print(df.dtypes)
    print(df[[
        'set', 'features', 'noise percent',
        'prec of labels at 1%',
        'prec of labels at 2%',
        'prec of labels at 3%',
        'prec of labels at 5%',
        'prec of labels at 10%',
        ]]
    )
    print(df[[
        'set', 'features', 'noise percent',
        'rec of labels at 1%',
        'rec of labels at 2%',
        'rec of labels at 3%',
        'rec of labels at 5%',
        'rec of labels at 10%',
        ]]
    )