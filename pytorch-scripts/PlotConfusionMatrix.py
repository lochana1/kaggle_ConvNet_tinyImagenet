"""
================
Confusion matrix
================

Example of confusion matrix usage to evaluate the quality
of the output of a classifier on the iris data set. The
diagonal elements represent the number of points for which
the predicted label is equal to the true label, while
off-diagonal elements are those that are mislabeled by the
classifier. The higher the diagonal values of the confusion
matrix the better, indicating many correct predictions.

The figures show the confusion matrix with and without
normalization by class support size (number of elements
in each class). This kind of normalization can be
interesting in case of class imbalance to have a more
visual interpretation of which class is being misclassified.

Here the results are not as good as they could be as our
choice for the regularization parameter C was not the best.
In real life applications this parameter is usually chosen
using :ref:`grid_search`.

"""



import itertools
import numpy as np
import matplotlib.pyplot as plt
import Augmentation as ag
import torch
from torchvision import datasets
from torch.autograd import Variable

import numpy as np
from sklearn.metrics import confusion_matrix

import os

class PlotConfusionMatrix():

    def __init__(self, aug, model, use_gpu=True):
        # Define augmentation strategy
        self.augmentation_strategy = ag.Augmentation(aug);
        self.data_transforms = self.augmentation_strategy.applyTransforms();
        self.model = model;
        self.model.train(False)
        self.use_gpu = use_gpu


    def loadPreTrainedModel(self, datapath, batch_size):

        # Root directory
        data_dir = datapath;
        ##

        ######### Data Loader ###########
        dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), self.data_transforms[x])
                 for x in ['val']}
        dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                       shuffle=True, num_workers=16)
                        # set num_workers higher for more cores and faster data loading
                        for x in ['val']}

        dset_sizes = {x: len(dsets[x]) for x in ['val']}
        dset_classes = dsets['val'].classes

        cm_label_list = []
        cm_pred_list = []

        for count, data in enumerate(dset_loaders['val']):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if self.use_gpu:
                inputs, labels = Variable(inputs.cuda()), \
                                 Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # forward
            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data, 1)

            for j in range(inputs.size()[0]):
                cm_pred_list.append(preds[j])
                cm_label_list.append(labels.data[j])

        return (dset_classes, cm_label_list, cm_pred_list)


    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')





    def createConfusionMatrix(self, datapath, batch_size):

        dset_classes, cm_label_list, cm_pred_list = \
            self.loadPreTrainedModel(datapath, batch_size)

        cnf_matrix = confusion_matrix(cm_label_list, cm_pred_list)[:20, :20]
        np.set_printoptions(precision=2)

        # Plot normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=dset_classes[:20], normalize=True,
                              title='Normalized confusion matrix')

        plt.show()






