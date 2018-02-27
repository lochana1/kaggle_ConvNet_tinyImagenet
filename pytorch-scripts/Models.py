from torchvision import models
import torch.nn as nn
import math
import pdb


class ModelT1(nn.Module):


    def __init__(self, nClasses=200):
        super(ModelT1, self).__init__();

        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=4, bias =True)

        self.relu_1 = nn.ReLU(True);
        self.batch_norm_1 = nn.BatchNorm2d(16);
        out = ((224 - 3 + (2 * 4)) / 1) + 1
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        out = ((out - 2 + (2 * 0)) / 2) + 1

        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_2 = nn.ReLU(True);
        self.batch_norm_2 = nn.BatchNorm2d(32);
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        out = ((out - 2 + (2 * 0)) / 2) + 1

        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_3 = nn.ReLU(True);
        self.batch_norm_3 = nn.BatchNorm2d(64);
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        out = ((out - 2 + (2 * 0)) / 2) + 1


        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_4 = nn.ReLU(True);
        self.batch_norm_4 = nn.BatchNorm2d(64);
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        out = ((out - 2 + (2 * 0)) / 2) + 1



        fc_input = out * out * 64
        self.fc_1 = nn.Linear(fc_input, 1024);

        self.relu_6 = nn.Softmax(1)
        self.batch_norm_6 = nn.BatchNorm1d(1024);

        self.fc_2 = nn.Linear(1024, nClasses);

    def forward(self, x):

        y = self.conv_1(x)
        y = self.relu_1(y)
        y = self.batch_norm_1(y)
        y = self.pool_1(y)

        y = self.conv_2(y)
        y = self.relu_2(y)
        y = self.batch_norm_2(y)
        y = self.pool_2(y)

        y = self.conv_3(y)
        y = self.relu_3(y)
        y = self.batch_norm_3(y)
        y = self.pool_3(y)

        y = self.conv_4(y)
        y = self.relu_4(y)
        y = self.batch_norm_4(y)
        y = self.pool_4(y)



        y = y.view(y.size(0), -1)
        y = self.fc_1(y)
        y = self.relu_6(y)
        y = self.batch_norm_6(y)

        y = self.fc_2(y)
        return (y)

class ModelT2(nn.Module):


    def __init__(self, nClasses=200):
        super(ModelT2, self).__init__();

        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=4, bias =True)

        self.relu_1 = nn.ReLU(True);
        self.batch_norm_1 = nn.BatchNorm2d(16);
        out = ((224 - 3 + (2 * 4)) / 1) + 1
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        out = ((out - 2 + (2 * 0)) / 2) + 1


        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_2 = nn.ReLU(True);
        self.batch_norm_2 = nn.BatchNorm2d(32);
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        out = ((out - 2 + (2 * 0)) / 2) + 1


        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_3 = nn.ReLU(True);
        self.batch_norm_3 = nn.BatchNorm2d(64);
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        out = ((out - 2 + (2 * 0)) / 2) + 1


        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_4 = nn.ReLU(True);
        self.batch_norm_4 = nn.BatchNorm2d(64);
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        out = ((out - 2 + (2 * 0)) / 2) + 1

        self.conv_5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_5 = nn.ReLU(True);
        self.batch_norm_5 = nn.BatchNorm2d(128);

        fc_input = out * out * 128
        self.fc_1 = nn.Linear(fc_input, 1024);

        self.relu_6 = nn.Softmax(1)
        self.batch_norm_6 = nn.BatchNorm1d(1024);
        self.dropout_1 = nn.Dropout(p=0.6);
        self.fc_2 = nn.Linear(1024, nClasses);

    def forward(self, x):

        y = self.conv_1(x)
        y = self.relu_1(y)
        y = self.batch_norm_1(y)
        y = self.pool_1(y)

        y = self.conv_2(y)
        y = self.relu_2(y)
        y = self.batch_norm_2(y)
        y = self.pool_2(y)

        y = self.conv_3(y)
        y = self.relu_3(y)
        y = self.batch_norm_3(y)
        y = self.pool_3(y)

        y = self.conv_4(y)
        y = self.relu_4(y)
        y = self.batch_norm_4(y)
        y = self.pool_4(y)

        y = self.conv_5(y)
        y = self.relu_5(y)
        y = self.batch_norm_5(y)


        y = y.view(y.size(0), -1)
        y = self.fc_1(y)
        y = self.relu_6(y)
        y = self.batch_norm_6(y)
        y = self.dropout_1(y)
        y = self.fc_2(y)
        return (y)

class ModelT3(nn.Module):


    def __init__(self, nClasses=200):
        super(ModelT3, self).__init__();

        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=4,bias =True)
        nn.init.xavier_uniform(self.conv_1.weight)

        out = ((224 - 3 + (2 * 4)) / 1) + 1
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        out = ((out - 2 + (2 * 0)) / 2) + 1


        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_2 = nn.ReLU(True);


        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_3 = nn.ReLU(True);
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        out = ((out - 2 + (2 * 0)) / 2) + 1


        self.conv_4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_4 = nn.ReLU(True);
        self.batch_norm_4 = nn.BatchNorm2d(64);

        self.conv_5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_5 = nn.ReLU(True);
        self.batch_norm_5 = nn.BatchNorm2d(128);
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1)
        out = ((out - 2 + (2 * 0)) / 2) + 1

        self.conv_6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_6 = nn.ReLU(True);

        self.conv_7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_6 = nn.ReLU(True);

        self.conv_8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_6 = nn.ReLU(True);

        print(out * out * 256)

        fc_input = out * out * 256
        self.fc_1 = nn.Linear(fc_input, 1024);

        self.relu_6 = nn.Softmax(1)
        self.batch_norm_6 = nn.BatchNorm1d(1024);
        self.dropout_1 = nn.Dropout(p=0.5);
        self.fc_2 = nn.Linear(1024, nClasses);

    def forward(self, x):

        y = self.conv_1(x)
        y = self.relu_1(y)
        y = self.batch_norm_1(y)
        y = self.pool_1(y)

        y = self.conv_2(y)
        y = self.relu_2(y)
        y = self.batch_norm_2(y)
        y = self.pool_2(y)

        y = self.conv_3(y)
        y = self.relu_3(y)
        y = self.batch_norm_3(y)
        y = self.pool_3(y)

        y = self.conv_4(y)
        y = self.relu_4(y)
        y = self.batch_norm_4(y)
        y = self.pool_4(y)

        y = self.conv_5(y)
        y = self.relu_5(y)
        y = self.batch_norm_5(y)


        y = y.view(y.size(0), -1)
        y = self.fc_1(y)
        y = self.relu_6(y)
        y = self.batch_norm_6(y)
        y = self.dropout_1(y)
        y = self.fc_2(y)
        return (y)

class ModelT4(nn.Module):


    def __init__(self, nClasses=200):
        super(ModelT4, self).__init__();



        self.conv_1 = nn.Conv2d(3,128, kernel_size=3, stride=1, padding=4)
        nn.init.xavier_uniform(self.conv_1.weight)

        self.relu_1 = nn.ReLU(True);
        self.batch_norm_1 = nn.BatchNorm2d(128);
        out = ((64 - 3 + (2 * 4)) / 1) + 1


        self.conv_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_2 = nn.ReLU(True);
        self.batch_norm_2 = nn.BatchNorm2d(256);
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        out = ((out - 2 + (2 * 0)) / 2) + 1


        self.conv_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_3 = nn.ReLU(True);
        self.batch_norm_3 = nn.BatchNorm2d(256);
        self.pool_3 = nn.AvgPool2d(kernel_size=2, stride=2)
        out = ((out - 2 + (2 * 0)) / 2) + 1


        self.conv_4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_4 = nn.ReLU(True);
        self.batch_norm_4 = nn.BatchNorm2d(128);
        #self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        #out = ((out - 2 + (2 * 0)) / 2) + 1

        self.conv_5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_5 = nn.ReLU(True);
        self.batch_norm_5 = nn.BatchNorm2d(128);
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1)
        out = ((out - 3 + (2 * 0)) / 1) + 1

        fc_input = out * out * 128
        print(fc_input)


        self.fc_1 = nn.Linear(fc_input, 1024);

        self.relu_6 = nn.Softmax(1)
        self.batch_norm_6 = nn.BatchNorm1d(1024);
        self.dropout_1 = nn.Dropout(p=0.6);
        self.fc_2 = nn.Linear(1024, nClasses);

    def forward(self, x):

        y = self.conv_1(x)
        y = self.relu_1(y)
        y = self.batch_norm_1(y)


        y = self.conv_2(y)
        y = self.relu_2(y)
        y = self.batch_norm_2(y)
        y = self.pool_2(y)

        y = self.conv_3(y)
        y = self.relu_3(y)
        y = self.batch_norm_3(y)
        y = self.pool_3(y)

        y = self.conv_4(y)
        y = self.relu_4(y)
        y = self.batch_norm_4(y)


        y = self.conv_5(y)
        y = self.relu_5(y)
        y = self.batch_norm_5(y)


        y = y.view(y.size(0), -1)
        y = self.fc_1(y)
        y = self.relu_6(y)
        y = self.batch_norm_6(y)
        y = self.dropout_1(y)
        y = self.fc_2(y)
        return (y)

class ModelT5(nn.Module):


    def __init__(self, nClasses=200):
        super(ModelT5, self).__init__();


        self.conv_1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=4)
        self.relu_1 = nn.ReLU(True);
        self.batch_norm_1 = nn.BatchNorm2d(128);
        out = ((64 - 3 + (2 * 4)) / 1) + 1


        self.conv_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_2 = nn.ReLU(True);
        self.batch_norm_2 = nn.BatchNorm2d(256);
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        out = ((out - 2 + (2 * 0)) / 2) + 1


        self.conv_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_3 = nn.ReLU(True);
        self.batch_norm_3 = nn.BatchNorm2d(256);


        self.conv_4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_4 = nn.ReLU(True);
        self.batch_norm_4 = nn.BatchNorm2d(128);
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        out = ((out - 2 + (2 * 0)) / 2) + 1

        self.conv_5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_5 = nn.ReLU(True);
        self.batch_norm_5 = nn.BatchNorm2d(128);


        fc_input = out * out * 128
        self.fc_1 = nn.Linear(fc_input, 1024);


        self.relu_6 = nn.Softmax(1)
        self.batch_norm_6 = nn.BatchNorm1d(1024);
        self.dropout_1 = nn.Dropout(p=0.4);
        self.fc_2 = nn.Linear(1024, nClasses);

    def forward(self, x):

        y = self.conv_1(x)
        y = self.relu_1(y)
        y = self.batch_norm_1(y)


        y = self.conv_2(y)
        y = self.relu_2(y)
        y = self.batch_norm_2(y)
        y = self.pool_2(y)

        y = self.conv_3(y)
        y = self.relu_3(y)
        y = self.batch_norm_3(y)

        y = self.conv_4(y)
        y = self.relu_4(y)
        y = self.batch_norm_4(y)
        y = self.pool_4(y)

        y = self.conv_5(y)
        y = self.relu_5(y)
        y = self.batch_norm_5(y)


        y = y.view(y.size(0), -1)
        y = self.fc_1(y)
        y = self.relu_6(y)
        y = self.batch_norm_6(y)
        y = self.dropout_1(y)
        y = self.fc_2(y)
        return (y)

class ModelT6(nn.Module):

    def __init__(self, nClasses=200):
        super(ModelT6, self).__init__();

        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=2)

        nn.init.orthogonal(self.conv_1.weight)

        self.relu_1 = nn.ReLU(True);

        self.batch_norm_1 = nn.BatchNorm2d(16);
        out = ((64 - 3 + (2 * 2)) / 1) + 1
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        out = ((out - 2 + (2 * 0)) / 2) + 1

        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=4)

        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_2 = nn.ReLU(True);

        self.batch_norm_2 = nn.BatchNorm2d(32);
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        out = ((out - 2 + (2 * 0)) / 2) + 1



        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        #self.relu_3 = nn.ReLU(True);
        self.relu_3 = nn.LeakyReLU(0.1);
        self.batch_norm_3 = nn.BatchNorm2d(64);
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        out = ((out - 2 + (2 * 0)) / 2) + 1



        self.conv_4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=4)
        out = ((out - 3 + (2 * 4)) / 1) + 1
        self.relu_4 = nn.LeakyReLU(0.1);
        self.batch_norm_4 = nn.BatchNorm2d(128);
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        out = ((out - 2 + (2 * 0)) / 2) + 1

        self.conv_5 = nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=4)
        out = ((out - 2 + (2 * 4)) / 1) + 1
        self.relu_5 = nn.LeakyReLU(0.1);
        self.batch_norm_5 = nn.BatchNorm2d(256);


        fc_input = out * out * 256
        self.fc_1 = nn.Linear(fc_input, 1100);
        self.relu_6 = nn.ReLU(True)
        self.batch_norm_6 = nn.BatchNorm1d(1100);
        self.dropout_1 = nn.Dropout(p=0.5);
        self.fc_2 = nn.Linear(1100, nClasses);


    def forward(self, x):


        y = self.conv_1(x)
        y = self.relu_1(y)
        y = self.batch_norm_1(y)
        y = self.pool_1(y)



        y = self.conv_2(y)
        y = self.relu_2(y)
        y = self.batch_norm_2(y)
        y = self.pool_2(y)

        y = self.conv_3(y)
        y = self.relu_3(y)
        y = self.batch_norm_3(y)
        y = self.pool_3(y)

        y = self.conv_4(y)
        y = self.relu_4(y)
        y = self.batch_norm_4(y)
        y = self.pool_4(y)

        y = self.conv_5(y)
        y = self.relu_5(y)
        y = self.batch_norm_5(y)

        y = y.view(y.size(0), -1)
        y = self.fc_1(y)
        y = self.relu_6(y)
        y = self.batch_norm_6(y)
        y = self.dropout_1(y)
        y = self.fc_2(y)

        return (y)

class ModelT7(nn.Module):

   def __init__(self, nClasses=200):

       super(ModelT7, self).__init__();

       self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=2)

       nn.init.constant(self.conv_1.bias, 0.1);
       self.relu_1 = nn.ReLU(True);

       self.batch_norm_1 = nn.BatchNorm2d(16);
       out = ((64 - 3 + (2 * 2)) / 1) + 1

       self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2);
       out = ((out - 2 + (2 * 0)) / 2) + 1
       print('Layer 1', out)


       self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=3)
       out = ((out - 3 + (2 * 3)) / 1) + 1
       self.relu_2 = nn.ReLU(True);
       # self.relu_2 = nn.LeakyReLU(0.1);
       self.batch_norm_2 = nn.BatchNorm2d(32);
       self.pool_2 = nn.MaxPool2d(kernel_size = 2, stride =2)
       out = ((out - 2 + (2 * 0)) / 2) + 1
       print('Layer 2', out)

       self.conv_3 = nn.Conv2d(32, 64,kernel_size=3,stride=1, padding=3)
       out = ((out - 3 + (2 * 3)) / 1) + 1
       self.relu_3 = nn.ReLU(True);
       # self.relu_3 = nn.LeakyReLU(0.1);
       self.batch_norm_3 = nn.BatchNorm2d(64);
       self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
       out = ((out - 2 + (2 * 0)) / 2) + 1
       print('Layer 3', out)

       self.conv_4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=4)
       out = ((out - 3 + (2 * 4)) / 1) + 1
       self.relu_4 = nn.ReLU(True);
       #self.relu_4 = nn.LeakyReLU(0.1);
       self.batch_norm_4 = nn.BatchNorm2d(128);
       self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
       out = ((out - 2 + (2 * 0)) / 2) + 1
       print('Layer 4', out)

       self.conv_5 = nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=4)
       out = ((out - 2 + (2 * 4)) / 1) + 1
       self.relu_5 = nn.ReLU(True);
       #self.relu_5 = nn.LeakyReLU(0.1);
       self.batch_norm_5 = nn.BatchNorm2d(256);
       # self.pool5 = nn.MaxPool2d()
       print('Layer 5', out)

       self.conv_6 = nn.Conv2d(256, 256, kernel_size=2, stride=3, padding=5)
       out = ((out - 2 + (2 * 5)) / 3) + 1
       # self.relu_5 = nn.ReLU(True);
       self.relu_6 = nn.LeakyReLU(0.1);
       self.batch_norm_6 = nn.BatchNorm2d(256);
       self.pool_6 = nn.MaxPool2d(kernel_size=2, stride=2)
       out = ((out - 2 + (2 * 0)) / 2) + 1
       print('Layer 6', out)

       self.conv_7 = nn.Conv2d(256, 128, kernel_size=2, stride=3, padding=2)
       out = ((out - 2 + (2 * 2)) / 3) + 1
       self.relu_7 = nn.ReLU(True);
       #self.relu_6 = nn.LeakyReLU(0.1);
       self.batch_norm_7 = nn.BatchNorm2d(128);
       #self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)
       print('Layer 7', out)


       self.conv_8 = nn.Conv2d(128, 64, kernel_size=2, stride=1, padding=2)
       out = ((out - 2 + (2 * 2)) / 1) + 1
       self.relu_8 = nn.ReLU(True);
       #self.relu_6 = nn.LeakyReLU(0.1);
       self.batch_norm_8 = nn.BatchNorm2d(64);
       self.pool_8 = nn.MaxPool2d(kernel_size=2, stride=2)
       out = ((out - 2 + (2 * 0)) / 2) + 1
       print('Layer 8', out)


       print(out*out*64)

       fc_input = out*out*64
       self.fc_1 = nn.Linear(fc_input, 1100);
       self.relu_9 = nn.ReLU(True)
       self.batch_norm_9 = nn.BatchNorm1d(1100);
       self.dropout_1 = nn.Dropout(p=0.5);
       self.fc_2 = nn.Linear(1100, nClasses);


   def forward(self, x):


       y = self.conv_1(x)
       y = self.relu_1(y)
       y = self.batch_norm_1(y)
       y = self.pool_1(y)

       y = self.conv_2(y)
       y = self.relu_2(y)
       y = self.batch_norm_2(y)
       y = self.pool_2(y)

       y = self.conv_3(y)
       y = self.relu_3(y)
       y = self.batch_norm_3(y)
       y = self.pool_3(y)

       y = self.conv_4(y)
       y = self.relu_4(y)
       y = self.batch_norm_4(y)
       y = self.pool_4(y)

       y = self.conv_5(y)
       y = self.relu_5(y)
       y = self.batch_norm_5(y)

       y = self.conv_6(y)
       y = self.relu_6(y)
       y = self.batch_norm_6(y)
       y = self.pool_6(y)

       y = self.conv_7(y)
       y = self.relu_7(y)
       y = self.batch_norm_7(y)


       y = self.conv_8(y)
       y = self.relu_8(y)
       y = self.batch_norm_8(y)
       y = self.pool_8(y)


       y = y.view(y.size(0), -1)
       y = self.fc_1(y)
       y = self.relu_9(y)
       y = self.batch_norm_9(y)
       y = self.dropout_1(y)
       y = self.fc_2(y)

       return (y)

class ModelT8(nn.Module):

   def __init__(self, nClasses=200):

       super(ModelT8, self).__init__();

       self.conv_1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=4)


       nn.init.orthogonal(self.conv_1.weight)

       self.relu_1 = nn.ReLU(True);
       # self.relu_1 = nn.LeakyReLU(0.1);
       self.batch_norm_1 = nn.BatchNorm2d(128);
       out = ((64 - 3 + (2 * 4)) / 1) + 1
       print('Conv 1', out)
       self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2);
       out = ((out - 2 + (2 * 0)) / 2) + 1
       print('Pool 1', out)


       self.conv_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=6)
       out = ((out - 3 + (2 * 6)) / 1) + 1
       print('Conv 2', out)
       self.relu_2 = nn.ReLU(True);
       # self.relu_2 = nn.LeakyReLU(0.1);
       self.batch_norm_2 = nn.BatchNorm2d(256);
       self.pool_2 = nn.MaxPool2d(kernel_size = 2, stride =2)
       out = ((out - 2 + (2 * 0)) / 2) + 1
       print('Pool 2', out)

       self.conv_3 = nn.Conv2d(256, 64,kernel_size=3,stride=1, padding=3)
       out = ((out - 3 + (2 * 3)) / 1) + 1
       print('Conv 3', out)
       self.relu_3 = nn.ReLU(True);
       # self.relu_3 = nn.LeakyReLU(0.1);
       self.batch_norm_3 = nn.BatchNorm2d(64);
       self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
       out = ((out - 2 + (2 * 0)) / 2) + 1
       print('Pool 3', out)

       self.conv_4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=4)
       out = ((out - 3 + (2 * 4)) / 1) + 1
       print('Conv 4', out)
       self.relu_4 = nn.ReLU(True);
       #self.relu_4 = nn.LeakyReLU(0.1);
       self.batch_norm_4 = nn.BatchNorm2d(128);
       self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
       out = ((out - 2 + (2 * 0)) / 2) + 1
       print('Pool 4', out)

       self.conv_5 = nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=4)
       out = ((out - 2 + (2 * 4)) / 1) + 1
       print('Conv 5', out)
       self.relu_5 = nn.ReLU(True);
       #self.relu_5 = nn.LeakyReLU(0.1);
       self.batch_norm_5 = nn.BatchNorm2d(256);
       # self.pool5 = nn.MaxPool2d()
       print('Pool 5', out)

       self.conv_6 = nn.Conv2d(256, 256, kernel_size=2, stride=3, padding=5)
       out = ((out - 2 + (2 * 5)) / 3) + 1
       print('Conv 6', out)
       # self.relu_5 = nn.ReLU(True);
       self.relu_6 = nn.LeakyReLU(0.1);
       self.batch_norm_6 = nn.BatchNorm2d(256);
       self.pool_6 = nn.MaxPool2d(kernel_size=2, stride=2)
       out = ((out - 2 + (2 * 0)) / 2) + 1
       print('Pool 6', out)



       print(out*out*256)
       fc_input = out*out*256
       self.fc_1 = nn.Linear(fc_input, 2);
       self.relu_9 = nn.ReLU(True)
       self.batch_norm_9 = nn.BatchNorm1d(1100);
       self.dropout_1 = nn.Dropout(p=0.4);
       self.fc_2 = nn.Linear(1100, nClasses);


   def forward(self, x):


       y = self.conv_1(x)
       y = self.relu_1(y)
       y = self.batch_norm_1(y)
       y = self.pool_1(y)


       y = self.conv_2(y)
       y = self.relu_2(y)
       y = self.batch_norm_2(y)
       y = self.pool_2(y)

       y = self.conv_3(y)
       y = self.relu_3(y)
       y = self.batch_norm_3(y)
       y = self.pool_3(y)

       y = self.conv_4(y)
       y = self.relu_4(y)
       y = self.batch_norm_4(y)
       y = self.pool_4(y)

       y = self.conv_5(y)
       y = self.relu_5(y)
       y = self.batch_norm_5(y)
       y = self.conv_6(y)
       y = self.relu_6(y)
       y = self.batch_norm_6(y)
       y = self.pool_6(y)

       y = y.view(y.size(0), -1)
       y = self.fc_1(y)
       y = self.relu_9(y)
       y = self.batch_norm_9(y)
       y = self.dropout_1(y)
       y = self.fc_2(y)


       return (y)

def resnet18():
    return models.resnet18()

def alexnet():
    return models.alexnet()

