import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import svm
import numpy as np
import pickle
import cv2
import sys


class phoneDetect(object):
    # Parameters for HOG descriptor as class variables
    winSize, blockSize, blockStride, cellSize, nbins = (32, 32), (16, 16), (8, 8), (8, 8), 9

    # Class constructor
    def __init__(self):
        # Paths to training related data
        self._numPosExamples = 0
        self._patchHalfHeight = 32
        self._patchHalfWidth = 32
        self._slidingWinSz = [2*self._patchHalfHeight, 2*self._patchHalfWidth]
        self._slidingStep = 8
        self._trainingItr = 10
        self._hog = cv2.HOGDescriptor(phoneDetect.winSize, phoneDetect.blockSize,
                                phoneDetect.blockStride, phoneDetect.cellSize, phoneDetect.nbins)
        self._clf = svm.SVC(gamma='scale')
        self._modelParamFileName ='finalized_model.sav'

    # Detect Phone
    def detectPhone(self, imgPath, showDetect=True):
        loadedModel = pickle.load(open(self._modelParamFileName, 'rb'))
        img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
        imgH, imgW = img.shape[0], img.shape[1]
        scoreList = list()
        topLeftList = list()
        sofarMaxScore = 0
        counter = 0
        for yv in range(0, img.shape[0]-self._slidingWinSz[0], self._slidingStep):
            for xv in range(0, img.shape[1]-self._slidingWinSz[1], self._slidingStep):
                currPatchGray = cv2.cvtColor(img[yv:yv+self._slidingWinSz[0], xv:xv+self._slidingWinSz[1]], cv2.COLOR_BGR2GRAY)
                if self._slidingWinSz[0] != 2*self._patchHalfHeight or self._slidingWinSz[1] != 2*self._patchHalfWidth:
                    currPatchGray = cv2.resize(currPatchGray, (2*self._patchHalfHeight, 2*self._patchHalfWidth))
                hf = self._hog.compute(currPatchGray)
                currScore = np.asscalar(loadedModel.decision_function(hf.T))
                if counter == 0:
                    sofarMaxScore = currScore
                    topLeftSafe = [yv, xv]
                else:
                    if currScore > sofarMaxScore:
                        sofarMaxScore = currScore
                        topLeftSafe = [yv, xv]
                if currScore > 0:
                    scoreList.append(currScore)
                    topLeftList.append([yv, xv])
                counter += 1
        if len(scoreList) > 0:
            scoreMax = max(scoreList)
            scoreMaxInd = scoreList.index(scoreMax)
            centerY, centerX = (topLeftList[scoreMaxInd][0] + self._slidingWinSz[0]/2)/imgH, \
                           (topLeftList[scoreMaxInd][1] + self._slidingWinSz[1]/2)/imgW
        else:
            centerY, centerX = (topLeftSafe[0] + self._slidingWinSz[0] / 2) / imgH, \
                               (topLeftSafe[1] + self._slidingWinSz[1] / 2) / imgW
        if showDetect:
            if len(scoreList) > 0:
                # cv2.rectangle(img, (x, y), (x + slidingWinSz[1], y + slidingWinSz[0]), (0, 255, 0), 3)
                cv2.rectangle(img, (topLeftList[scoreMaxInd][1], topLeftList[scoreMaxInd][0]),
                              (topLeftList[scoreMaxInd][1] + self._slidingWinSz[1], topLeftList[scoreMaxInd][0] + self._slidingWinSz[0]),
                              (0, 255, 0), 3)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.show()
            else:
                cv2.rectangle(img, (topLeftSafe[1], topLeftSafe[0]),
                              (topLeftSafe[1] + self._slidingWinSz[1],
                               topLeftSafe[0] + self._slidingWinSz[0]),
                              (0, 255, 0), 3)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.show()
        print(centerX, centerY)
        return centerY, centerX

testData = sys.argv[1]
pF = phoneDetect()
pF.detectPhone(testData)
