from __future__ import division, print_function
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import wfdb


def left_bundle_image_generation():

    paths = glob.glob('./mit-bih-arrhythmia-database-1.0.0/*.atr')
    paths = [path[:-4] for path in paths]
    paths.sort()
    print(paths)
    for name in paths:
        print(name)

    left_bundle = []
    for e in paths:
        print('Jentre dans la boucle principale')
        signals, fields = wfdb.rdsamp(e, channels=[0])
        #left_bundle = []
        ann = wfdb.rdann(e, 'atr')
        good = ['L']
        ids = np.in1d(ann.symbol, good)
        imp_beats = ann.sample[ids]
        beats = (ann.sample)
        for count, i in enumerate(imp_beats):

            beats = list(beats)
            j = beats.index(i)

            if(j!=0 and j!=(len(beats)-1)):
                x = beats[j-1]
                y = beats[j+1]
                diff1 = abs(x - beats[j])//2
                diff2 = abs(y - beats[j])//2
                left_bundle.append(signals[beats[j] - diff1: beats[j] + diff2, 0])
                #print(Normal[count])


    for count, i in enumerate(left_bundle):
        print('jentre')
        fig = plt.figure(frameon=False)
        plt.plot(i)
        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        filename = './images_left_bundle' + '/' + str(count)+'.png'
        fig.savefig(filename)
        im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.resize(im_gray, (128, 128), interpolation = cv2.INTER_AREA)
        cv2.imwrite(filename, im_gray)
        plt.close('all')