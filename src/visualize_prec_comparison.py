#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 10 Apr, 2023

from matplotlib import pyplot as plt

rk_dic = {
    'unimib' : [0.96, 0.91, 0.95, 0.90],
    'uci' : [0.92, 0.87, 0.93, 0.87],
    'twristar' : [0.94, 0.86, 0.97, 0.87],
    'sh_loco' : [0.94, 0.90, 0.95, 0.88]
}

lf_dic = {
    'unimib' : [0.98, 1.00, 0.85, 0.86],
    'uci' : [0.07, 0.12, 0.32, 0.33],
    'twristar' : [0.53, 0.53, 0.67, 0.73],
    'sh_loco' : [0.31, 0.04, 0.09, 0.04]
}

ticks = ['UniMiB', 'UCI-HAR', 'TWristAR', 'SH Loco']

if __name__ == '__main__':
    # plt.figure()
    # rk_pos = range(0,8, 2)
    # lf_pos = range(1,9, 2)
    
    # plt.title('')
    # plt.xlabel('Average Precision')
    # plt.barh(rk_pos, [sum(i)/4 for i in rk_dic.values()], color='maroon', tick_label=ticks)
    # plt.barh(lf_pos, [sum(i)/4 for i in lf_dic.values()], color='royalblue', tick_label=ticks)
    # plt.legend(['Rising K', 'Labelfix'])
    # plt.yticks(rotation = 45)
    # plt.savefig('imgs/prec_comparison.pdf')

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    #plt.ylim([50,100])
    bplot1 = ax1.boxplot(rk_dic.values(), labels=ticks, patch_artist=True)
    ax1.set_title('Rising K')
    ax1.yaxis.grid(True)
    ax1.set_ylim([0,1])    
    bplot2 = ax2.boxplot(lf_dic.values(), labels=ticks, patch_artist=True)
    ax2.set_title('Labelfix')
    ax2.yaxis.grid(True)
    ax2.set_ylim([0,1])
    ax2.yaxis.set_tick_params(labelleft=False)
    for patch in bplot1['boxes']:
        patch.set_facecolor('maroon')
    for patch in bplot2['boxes']:
        patch.set_facecolor('royalblue')
    plt.savefig('imgs/prec_comparison.pdf')

    