import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import svm
import numpy as np
import pickle
import cv2
import sys


class phoneFinder(object):
    # Parameters for HOG descriptor as class variables
    winSize, blockSize, blockStride, cellSize, nbins = (32, 32), (16, 16), (8, 8), (8, 8), 9

    # Class constructor
    def __init__(self, dataPath):
        # Paths to training related data
        self._PATH2 = dataPath
        self._numPosExamples = 0
        self._patchHalfHeight = 32
        self._patchHalfWidth = 32
        self._slidingWinSz = [2*self._patchHalfHeight, 2*self._patchHalfWidth]
        self._slidingStep = 8
        self._trainingItr = 10
        self._hog = cv2.HOGDescriptor(phoneFinder.winSize, phoneFinder.blockSize,
                                phoneFinder.blockStride, phoneFinder.cellSize, phoneFinder.nbins)
        self._clf = svm.SVC(gamma='scale')
        self._labelFileName = "labels.txt"
        self._annoFileName = "anno.txt"
        self._modelParamFileName ='finalized_model.sav'

    @staticmethod
    def checkOverlap(rectA, rectB):
        overLap = True
        if (rectA[0, 0] > rectB[1, 0]) or (rectB[0, 0] > rectA[1, 0]):
            overLap = False
        if (rectA[0, 1] > rectB[1, 1]) or (rectA[0, 1] > rectB[1, 1]):
            overLap = False
        return overLap

    # Generate anno.txt
    def genAnnoFile(self, showPatch = False):
        annoFileHandle = open(trainDataPath + self._annoFileName, 'w')
        labelData = open(trainDataPath + self._labelFileName, 'r')
        countLine = 0
        while True:
            line = labelData.readline()
            if not line:
                break
            elif countLine > 0:
                annoFileHandle.write("\n")
                countLine += 1
            else:
                countLine += 1
            val = line.split(" ")
            train_img_name = trainDataPath + val[0]
            config = Path(train_img_name)
            if config.is_file():
                img = cv2.imread(train_img_name, cv2.IMREAD_UNCHANGED)
                imgH, imgW = img.shape[0], img.shape[1]
                centerPixelX = int(float(val[1]) * img.shape[1])
                centerPixelY = int(float(val[2]) * img.shape[0])
                topLeftPxl = [centerPixelY - self._patchHalfHeight, centerPixelX - self._patchHalfWidth]
                bottomRightPxl = [centerPixelY + self._patchHalfHeight, centerPixelX + self._patchHalfWidth]
                if topLeftPxl[0] < 0:
                    topLeftPxl[0] = 0
                if topLeftPxl[1] < 0:
                    topLeftPxl[1] = 0
                if bottomRightPxl[0] > imgH:
                    bottomRightPxl[0] = imgH
                if bottomRightPxl[1] > imgW:
                    bottomRightPxl[1] = imgW
                annoFileHandle.write(val[0] + "," + str(topLeftPxl[0]) + "," + str(topLeftPxl[1])
                                     + "," + str(bottomRightPxl[0]) + "," + str(bottomRightPxl[1]))
                posPatch = img[topLeftPxl[0]:bottomRightPxl[0], topLeftPxl[1]:bottomRightPxl[1]]
                # posPatch = cv2.resize(posPatch, (2 * self._patchHalfHeight, 2 * self._patchHalfWidth))
                if showPatch:
                    plt.imshow(cv2.cvtColor(posPatch, cv2.COLOR_BGR2RGB))
                    plt.show()

    # Generate feature vector from Negative Examples
    def genNegFeat(self, negSampImg=8, showPatch=False):
        # Initialize containers to hold features and labels
        count = 0
        anno_data = open(self._PATH2 + self._annoFileName, 'r')
        while True:
            line = anno_data.readline()
            if not line:
                break
            val = line.split(",")
            train_img_name = self._PATH2 + val[0]
            config = Path(train_img_name)
            if config.is_file():
                img = cv2.imread(train_img_name, cv2.IMREAD_UNCHANGED)
                imgH, imgW = img.shape[0], img.shape[1]
                validNeg = 0
                while validNeg < negSampImg:
                    topLeft = [int(np.asscalar(np.random.uniform(0, imgH, 1))), int(np.asscalar(np.random.uniform(0, imgW, 1)))]
                    botRight = [topLeft[0]+2*self._patchHalfHeight, topLeft[1]+2*self._patchHalfWidth]
                    negRect = np.array([topLeft, botRight])
                    posRect = np.array([[int(val[1]), int(val[2])], [int(val[3]), int(val[4])]])
                    if negRect[1, 0] < imgH and negRect[1, 1] < imgW:
                        if not self.checkOverlap(posRect, negRect):
                            validNeg += 1
                            img_gray = cv2.cvtColor(img[negRect[0, 0]:negRect[1, 0], negRect[0, 1]:negRect[1, 1]], cv2.COLOR_BGR2GRAY)
                            hf = self._hog.compute(img_gray)
                            if count == 0:
                                neg_feature_mat = hf
                                neg_lbl = np.array(-1)
                            else:
                                neg_feature_mat = np.hstack((neg_feature_mat, hf))
                                neg_lbl = np.hstack((neg_lbl, -1))
                            count += 1
                            if showPatch:
                                plt.imshow(img_gray, cmap='gray')
                                plt.show()
        anno_data.close()
        return neg_feature_mat, neg_lbl

    # Generate feature vector from Positive Examples
    def genPosFeat(self, showPatch=False):
        anno_data = open(self._PATH2 + self._annoFileName, 'r')
        # Initialize containers to hold features and labels
        count = 0
        while True:
            line = anno_data.readline()
            if not line:
                break
            val = line.split(",")
            train_img_name = self._PATH2 + val[0]
            config = Path(train_img_name)
            if config.is_file():
                img = cv2.imread(train_img_name, cv2.IMREAD_UNCHANGED)
                img_gray = cv2.cvtColor(img[int(val[1]):int(val[3]), int(val[2]):int(val[4])], cv2.COLOR_BGR2GRAY)
                if img_gray.shape[0] != 2*self._patchHalfHeight or img_gray.shape[1] != 2*self._patchHalfWidth:
                    img_gray = cv2.resize(img_gray, (2*self._patchHalfHeight, 2*self._patchHalfWidth))
                hf = self._hog.compute(img_gray)
                if count == 0:
                    pos_feature_mat = hf
                    pos_lbl = np.array(1)
                else:
                    pos_feature_mat = np.hstack((pos_feature_mat, hf))
                    pos_lbl = np.hstack((pos_lbl, 1))
                count += 1
                if showPatch:
                    cv2.rectangle(img, (int(val[2]), int(val[1])),
                                  (int(val[2])+self._featPatchWidth, int(val[1])+self._featPatchHeight), (0, 255, 0), 3)
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.show()
        anno_data.close()
        self._numPosExamples = pos_feature_mat.shape[1]
        return pos_feature_mat, pos_lbl

    # Generate Complete Features and Labels
    def genPosNegFeat(self):
        # Generate Features from Positive Examples
        print("Generating positive features . . .")
        featsPos, lblPos = self.genPosFeat()
        print("Dimension of positive feature matrix: {}".format(featsPos.shape))
        # Generate Features from Negative Examples
        print("Generating negative features . . .")
        featsNeg, lblNeg = self.genNegFeat()
        print("Dimension of negative feature matrix: {}".format(featsNeg.shape))
        return np.hstack((featsPos, featsNeg)), np.hstack((lblPos, lblNeg))

    # Train SVM Model
    def trainSVM(self):
        print("Feature extraction started ....")
        featureX, labelY = self.genPosNegFeat()
        print("Feature extraction completed ....")
        print("Fitting data with nonlinear SVM kernel")
        self._clf.fit(featureX.T, labelY)
        print("Data fitted")

        # Identify hard-negative examples
        print("Fitting data with hard negative examples")
        for numItr in range(self._trainingItr):
            hardNegCount = 0
            for i in range(self._numPosExamples, featureX.shape[1]):
                negEx = featureX[:, i].reshape((featureX.shape[0], 1))
                if self._clf.decision_function(negEx.T)[0] > -0.3:
                    if hardNegCount > 0:
                        hardNegFeat = np.hstack((hardNegFeat, negEx))
                        hardNegLabel = np.hstack((hardNegLabel, -1))
                    else:
                        hardNegFeat = negEx
                        hardNegLabel = np.array(-1)
                    hardNegCount += 1
                    print("scores of negative examples: {}".format(self._clf.decision_function(negEx.T)[0]))
            # print("hardNegFeat: {}".format(hardNegFeat.shape))
            # Again generate Positive and Negative Features
            featureX, labelY = self.genPosNegFeat()
            # Add the Hard Negative Examples
            if hardNegCount > 0:
                featureX = np.hstack((featureX, hardNegFeat))
                labelY = np.hstack((labelY, hardNegLabel))
            shuffleInd = np.arange(0, featureX.shape[1])
            np.random.shuffle(shuffleInd)
            featureXT = featureX.T
            featureXT = featureXT[shuffleInd]
            labelY = labelY[shuffleInd]
            print("Curret number of features: {}".format(featureX.shape[1]))
            self._clf.fit(featureXT, labelY)
            # self._clf.fit(featureX.T, labelY)

        # Write trained parameters into an external file
        pickle.dump(self._clf, open(self._modelParamFileName, 'wb'))
        print("Trained parameters have been written to file: {}".format(self._modelParamFileName))


trainDataPath = sys.argv[1] + "/"
pF = phoneFinder(trainDataPath)
pF.genAnnoFile()
pF.trainSVM()
