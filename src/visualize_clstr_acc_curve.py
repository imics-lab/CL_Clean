#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 09 Oct, 2022
#
#Make some nice models with a common interface

from audioop import avg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_clstr = {
    'Twristar' : [
        0.721052631578947,0.721052631578947,0.721052631578947,0.897368421052632
        ,0.913157894736842,0.894736842105263,0.871052631578947,0.876315789473684
        ,0.878947368421053,0.713157894736842,0.707894736842105,0.789473684210526
        ,0.9,0.918421052631579,0.91578947368421,0.81578947368421,0.771052631578947
        ,0.831578947368421,0.921052631578947,0.91578947368421,0.931578947368421
        ,0.93421052631579,0.93421052631579,0.942105263157895
    ],
    'UniMiB' : [
        0.316887633123234,0.316887633123234,0.316887633123234,0.835035861769181
        ,0.926537709193654,0.918495979134971,0.929580525972614,0.914583786133449
        ,0.903716583351445,0.740056509454466,0.790697674418605,0.72266898500326
        ,0.950010867202782,0.953705716148663,0.948489458813302,0.900239078461204
        ,0.885677026733319,0.88871984351228,0.94457726581178,0.911975657465768
        ,0.952401651814823,0.655074983699196,0.618778526407303,0.63855683547055
    ],
    'Synthetic' : [
        0.695304695304695,0.695304695304695,0.695304695304695,0.97002997002997
        ,0.989010989010989,0.988011988011988,0.965034965034965,0.956043956043956
        ,0.948051948051948,0.711288711288711,0.686313686313686,0.993006993006993
        ,0.989010989010989,0.984015984015984,0.986013986013986,0.883116883116883
        ,0.9000999000999,0.867132867132867,0.987012987012987,0.986013986013986
        ,0.992007992007992,1,0.999000999000999,0.998001998001998
    ],
    'SH Locomotion' : [
        0.603476752828475,0.603476752828475,0.603476752828475,0.845156495168883
        ,0.851474110165992,0.855231645883227,0.840490544223305,0.837558840531836
        ,0.842431249483855,0.544347179783632,0.553307457263193,0.545296886613263
        ,0.854612271863903,0.856511685523165,0.857048476339912,0.843133206705756
        ,0.840944751837476,0.836278800891898,0.856800726732183,0.859278222809481
        ,0.853001899413659,0.713312412255347,0.711702039805104,0.715129242712032
    ],
    'UCI HAR' : [
        0.36525208560029,0.36525208560029,0.36525208560029,0.656873413130214
        ,0.621871599564744,0.647080159593761,0.51614073268045,0.534820457018498
        ,0.456655785273848,0.526115342763874,0.484403336960464,0.529742473703301
        ,0.683170112441059,0.663220892274211,0.680449764236489,0.513783097569822
        ,0.473703300689155,0.481864345302865,0.6563293434893,0.669024301777294
        ,0.686071817192601,0.776931447225245,0.784185709104099,0.788900979325354
    ]
}

test_clstr = {
    'Twristar' : [
        0.745562130177515,0.745562130177515,0.745562130177515,0.899408284023669
        ,0.896449704142012,0.92603550295858,0.940828402366864,0.91715976331361
        ,0.937869822485207,0.772189349112426,0.772189349112426,0.85207100591716
        ,0.91715976331361,0.890532544378698,0.92603550295858,0.872781065088757
        ,0.772189349112426,0.78698224852071,0.902366863905325,0.908284023668639
        ,0.946745562130178,0.920118343195266,0.940828402366864,0.931952662721894
    ],
    'UniMiB' : [
        0.429133858267717,0.429133858267717,0.429133858267717,0.903543307086614
        ,0.967191601049869,0.961942257217848,0.959317585301837,0.94488188976378
        ,0.946850393700787,0.830708661417323,0.873359580052493,0.793307086614173
        ,0.970472440944882,0.973097112860892,0.967191601049869,0.939632545931758
        ,0.935695538057743,0.933070866141732,0.965223097112861,0.949475065616798
        ,0.965223097112861,0.860892388451444,0.841863517060367,0.818897637795276
    ],
    'Synthetic' : [
        0.663366336633663,0.663366336633663,0.663366336633663,0.94059405940594
        ,0.95049504950495,0.97029702970297,0.94059405940594,0.930693069306931
        ,0.900990099009901,0.623762376237624,0.623762376237624,1
        ,0.96039603960396,0.930693069306931,0.97029702970297,0.861386138613861
        ,0.841584158415841,0.841584158415841,0.930693069306931,0.96039603960396
        ,0.96039603960396,1,0.99009900990099,0.99009900990099
    ],
    'SH Locomotion' : [
        0.717460317460317,0.717460317460317,0.717460317460317,0.861111111111111
        ,0.87005772005772,0.868181818181818,0.858946608946609,0.853823953823954
        ,0.861976911976912,0.588600288600289,0.586940836940837,0.595887445887446
        ,0.865223665223665,0.867821067821068,0.868686868686869,0.856709956709957
        ,0.856493506493507,0.853318903318903,0.857070707070707,0.873737373737374
        ,0.867748917748918,0.646248196248196,0.641774891774892,0.64018759018759
    ],
    'UCI HAR' : [
        0.433322022395657,0.433322022395657,0.433322022395657,0.67458432304038
        ,0.650492025788938,0.66610111978283,0.484221241940957,0.524601289446895
        ,0.447573803868341,0.525619273837801,0.541567695961995,0.534102477095351
        ,0.692568713946386,0.69833729216152,0.72718018323719,0.524940617577197
        ,0.481167288768239,0.465897522904649,0.666779776043434,0.70512385476756
        ,0.69833729216152,0.853070919579233,0.850356294536817,0.839497794367153
    ]
}

accuracis = {
     'Twristar' : [
        0.609467455621302,0.642011834319527,0.775147928994083,0.872781065088757
        ,0.923076923076923,0.822485207100592,0.517751479289941,0.692307692307692
        ,0.748520710059172,0.630177514792899,0.562130177514793,0.553254437869823
        ,0.840236686390533,0.920118343195266,0.85207100591716,0.86094674556213
        ,0.718934911242604,0.742603550295858,0.923076923076923,0.949704142011834
        ,0.840236686390533,0.514792899408284,0.529585798816568,0.473372781065089
     ],
     'UniMiB' : [
        0.440944881889764,0.400262467191601,0.358267716535433,0.66010498687664
        ,0.627952755905512,0.631233595800525,0.702755905511811,0.64763779527559
        ,0.681102362204724,0.739501312335958,0.732283464566929,0.72244094488189
        ,0.709317585301837,0.706692913385827,0.675853018372703,0.503280839895013
        ,0.492782152230971,0.496062992125984,0.693569553805774,0.685695538057743
        ,0.609580052493438,0.35761154855643,0.36745406824147,0.331364829396325
     ],
     'Synthetic' : [
        0.841584158415841,0.732673267326733,0.693069306930693,1
        ,0.99009900990099,0.98019801980198,0.98019801980198,0.97029702970297
        ,0.97029702970297,0.851485148514851,0.831683168316832,0.841584158415841
        ,1,1,0.98019801980198,0.98019801980198
        ,0.930693069306931,0.95049504950495,1,1
        ,0.97029702970297,1,0.99009900990099,0.99009900990099
     ],
     'SH Locomotion' : [
        0.123015873015873,0.226767676767677,0.266594516594517,0.204112554112554
        ,0.191053391053391,0.192568542568543,0.1995670995671,0.118831168831169
        ,0.162121212121212,0.317027417027417,0.292857142857143,0.317604617604618
        ,0.153751803751804,0.189321789321789,0.168253968253968,0.157070707070707
        ,0.157792207792208,0.159018759018759,0.214790764790765,0.221139971139971
        ,0.182395382395382,0.198773448773449,0.191630591630592,0.197258297258297
     ],
     'UCI HAR' : [
        0.525619273837801,0.541567695961995,0.512385476756023,0.694265354597896
        ,0.727519511367492,0.687139463861554,0.32880895826264,0.34916864608076
        ,0.281981676280964,0.675602307431286,0.658635900916186,0.665761791652528
        ,0.767899558873431,0.766542246352223,0.712249745503902,0.587037665422464
        ,0.60162877502545,0.602646759416356,0.757380386834068,0.727519511367492
        ,0.70173057346454,0.797081778079403,0.794367153036987,0.765863590091618
     ]
}



p = ['maroon', 'royalblue', 'forestgreen', 'sandybrown', 'rebeccapurple']

avg_delta_acc = [
    [0.06,	0.02,	0.03,	0.07,	0.01],
    [0.03,	0.06,	0.01,	-0.04,	0.01],
    [0.00,	-0.01,	0.02,	0.18,	-0.03],
    [0.00,	0.02,	-0.01,	-0.05,	0.00],
    [0.05,	0.06,	0.01,	0.03,	0.01],
    [0.00,	-0.03,	-0.01,	0.03,	0.00],
    [0.00,	0.01,	0.02,	0.00,	0.00],
    [0.00,	0.04,	0.03,	-0.07,	0.00]
]

if __name__ == '__main__':
    
    # #Plot train vs. test clusterability
    # plt.figure()
    # plt.title('Train vs. Test Clusterability', fontsize=14)
    # for i, set in enumerate(test_clstr.keys()):
    #     a, b = np.polyfit(train_clstr[set], test_clstr[set], 1)
    #     plt.scatter(train_clstr[set], test_clstr[set], c=p[i], marker='.')
    #     plt.plot(train_clstr[set], [a*j+b for j in train_clstr[set]], c=p[i], label=set)
    # plt.xlabel('Train', fontsize=14)
    # plt.ylabel('Test', fontsize=14)
    # plt.legend()
    # plt.savefig('imgs/train_v_test_clusterability.pdf')

    # #Plot train clstr vs. final accuracy
    # plt.figure()
    # plt.title("Train Clusterability vs. Accuracy", fontsize=14)
    # for i, set in enumerate(accuracis.keys()):
    #     a, b = np.polyfit(train_clstr[set], accuracis[set], 1)
    #     plt.scatter(train_clstr[set], accuracis[set], c=p[i], marker='.')
    #     plt.plot(train_clstr[set], [a*j+b for j in train_clstr[set]], c=p[i], label=set)
    # plt.xlabel('Clusterability', fontsize=14)
    # plt.ylabel('Accuracy', fontsize=14)
    # plt.legend()
    # plt.savefig('imgs/train_clusterability_vs_accuracy.pdf')

    # #Plot test clstr vs. final accuracy
    # plt.figure()
    # plt.title("Test Clusterability vs. Accuracy", fontsize=14)
    # for i, set in enumerate(accuracis.keys()):
    #     plt.scatter(test_clstr[set], accuracis[set], c=p[i], marker='.')
    # plt.xlabel('Clusterability', fontsize=14)
    # plt.ylabel('Accuracy', fontsize=14)
    # plt.savefig('imgs/test_clusterability_vs_accuracy.pdf')

    # #Plot delta accuracy heatmap
    # avg_delta_acc_percents = np.zeros((8, 5))
    # for i in range(8):
    #     for j in range(5):
    #         avg_delta_acc_percents[i, j] = avg_delta_acc[i][j]*100
    # fig, ax = plt.subplots()
    # plt.title('Avg Change in Acc. After Cleaning.')
    # e = ['Engineered', 'SimCLR+CNN', 'SimCLR+Tran', 'SimCLR+LSTM', 'NNCLR+CNN', 'NNCLR+Tran', 'NNCLR+LSTM', 'Sup. CNN']
    # s = ['Synth.', 'UniMiB', 'UCI', 'Twristar', 'SH Loco.']
    # ax.imshow(avg_delta_acc, cmap='bone')
    # ax.set_yticks(range(8), e)
    # ax.set_xticks(range(5), s)
    # plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
    #      rotation_mode="anchor")
    # plt.setp(ax.get_yticklabels(), rotation=30, ha="right",
    #     rotation_mode="anchor")
    # for i in range(8):
    #     for j in range(5):
    #         ax.text(j, i, '{:.1f}%'.format(avg_delta_acc_percents[i][j]), ha="center", va="center", color="w" if abs(avg_delta_acc_percents[i][j]) < 10 else "black")
    # plt.savefig('imgs/accuracy_heatmap.pdf')
    # #plt.show()

    #Plot change in accuracy bargraph
    FEATURES = 1
    NOISE_PERCENT = 2
    ACC = 5

    exp3_df = pd.read_csv('results/canonical_results_exp3.csv')
    averaged_raw_accuracies = {}

    sets = exp3_df['set']
    feats = exp3_df['features']
    accs = exp3_df['accuracy on clean test']
    noises = exp3_df['noise percent']

    for i in range(len(sets)):

        if feats[i] not in averaged_raw_accuracies.keys():
            averaged_raw_accuracies[feats[i]] = {
                'Zero' : 0,
                'Five Uncleaned' : 0,
                'Five Cleaned' : 0,
                'Ten Uncleaned' : 0,
                'Ten Cleaned' : 0
            }

        #print(f'{sets[i]}, {feats[i]}, {noises[i]}, {accs[i]}')
        if noises[i] == '0' : 
            averaged_raw_accuracies[feats[i]]['Zero'] += accs[i]
        elif noises[i] == '5' : 
            averaged_raw_accuracies[feats[i]]['Five Uncleaned'] += accs[i]
        elif noises[i] == '5-cleaned' :
            #print(sets[i], ' ', feats[i], ' ', noises[i]) 
            averaged_raw_accuracies[feats[i]]['Five Cleaned'] += accs[i]
        elif noises[i] == '10' : 
            averaged_raw_accuracies[feats[i]]['Ten Uncleaned'] += accs[i]
        elif noises[i] == '10-cleaned' : 
            averaged_raw_accuracies[feats[i]]['Ten Cleaned'] += accs[i]
        else: print('Encountered the unknowable')

    zero_bars = []
    f_u_bars = []
    f_c_bars = []
    t_u_bars = []
    t_c_bars = []

    for f in averaged_raw_accuracies.keys():
        for n in averaged_raw_accuracies[f].keys():
            averaged_raw_accuracies[f][n] /= 5 #divide by number of sets
            print(f'{f} {n}: {averaged_raw_accuracies[f][n]}')

    for f in averaged_raw_accuracies.keys():
        zero_bars.append(averaged_raw_accuracies[f]['Zero'])
        f_u_bars.append(averaged_raw_accuracies[f]['Five Uncleaned'])
        f_c_bars.append(averaged_raw_accuracies[f]['Five Cleaned'])
        t_u_bars.append(averaged_raw_accuracies[f]['Ten Uncleaned'])
        t_c_bars.append(averaged_raw_accuracies[f]['Ten Cleaned'])

    labels = ['', 'Engineered', 'SimCLR+CNN', 'SimCLR+Tran', 'SimCLR+LSTM', 'NNCLR+CNN', 'NNCLR+Tran', 'NNCLR+LSTM', 'Sup. CNN']
    
    x = np.arange(len(labels)-1)
    width = 0.15
    fig, ax = plt.subplots()
    
    rects1 = ax.bar(x - 2*width, zero_bars, width, label='Zero Noise', color=p[0])
    rects2 = ax.bar(x - width, f_u_bars, width, label='5% Uncleaned', color=p[1])
    rects3 = ax.bar(x, f_c_bars, width, label='5% Cleaned', color=p[2])
    rects4 = ax.bar(x + width, t_u_bars, width, label='10% Uncleaned', color=p[3])
    rects5 = ax.bar(x + 2*width, t_c_bars, width, label='10% Cleaned', color=p[4])

    ax.set_ylabel('Accuracy')
    ax.set_title('Avg Accuracy by Feature Extractor')
    #ax.set_xticks(x, labels)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim([.35, .80])

    plt.setp(ax.get_xticklabels(), rotation=330, ha="left", rotation_mode="anchor")

    fig.tight_layout()

    #plt.show()
    plt.savefig('imgs/accuracry bar chart.pdf')
    






        

        



