import torch.nn as nn
import torch
import os,sys
from osgeo import ogr
from osgeo import gdal
from osgeo.gdalconst import *
import numpy as np
from numpy import *
import random
import torch.nn.functional as F
import keras
import torch.utils.data as Data
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchkeras
from sklearn.metrics import accuracy_score,confusion_matrix,cohen_kappa_score,f1_score,mean_squared_error,roc_auc_score,precision_score,recall_score
from torchkeras import summary, Model
from keras import backend as K
import matplotlib.pyplot as plt
from bottleneck_transformer_pytorch import BottleStack
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=5,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
    )


class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=64, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 15, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet50():
    return ResNet([3, 4, 6, 3])

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, # i=64 h=15
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.bn1 = nn.BatchNorm2d(input_dim + hidden_dim)
        self.bnct = nn.BatchNorm2d(hidden_dim)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        self._check_kernel_size_consistency(kernel_size)
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class CNNConvLSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim,kernel_size,numlayers,timestep,patch_size):
        super(CNNConvLSTM,self).__init__()
        self.saresnet = cnnmodel
        self.myconvlstm = ConvLSTM(input_dim,hidden_dim, (kernel_size[1],kernel_size[2]), numlayers, True, True, False)
        self.bn1 = nn.BatchNorm2d(2)
        self.relu1 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        x = self.saresnet(x)
        _,last_states = self.myconvlstm(x)
        h = last_states[0][0]
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.avgpool(h)
        h = h.view(h.size(0),-1)
        return h

class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0),1,64,x.size(2),x.size(3))
        # 如果数据集最后一个batch样本数量小于定义的batch_batch大小，会出现mismatch问题。可以自己修改下，如只传入后面的shape，然后通过x.szie(0)，来输入。
        return x


def cnnmodel():

    layer = BottleStack(
        dim=256,
        fmap_size=14,  # set specifically for imagenet's 224 x 224
        dim_out=2048,
        proj_factor=4,
        downsample=False,
        heads=5,
        dim_head=128,
        rel_pos_emb=True,
        activation=nn.ReLU()
    )

    resnet = ResNet50()
    backbone = list(resnet.children())


    cnnmodel = nn.Sequential(
        *backbone[:2],
        layer,
        nn.Conv2d(2048, 64, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        Reshape(),
    )
    return cnnmodel

def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    y_ground_cls = torch.argmax(nn.Softmax(dim=1)(y_true),dim=1).data
    return accuracy_score(y_ground_cls.cpu(),y_pred_cls.cpu())

def rmse(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    y_ground_cls = torch.argmax(nn.Softmax(dim=1)(y_true), dim=1).data
    return np.sqrt(mean_squared_error(y_ground_cls.cpu(),y_pred_cls.cpu()))

def kappa(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    y_ground_cls = torch.argmax(nn.Softmax(dim=1)(y_true), dim=1).data
    return cohen_kappa_score(y_ground_cls.cpu(),y_pred_cls.cpu())

def recall(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    y_ground_cls = torch.argmax(nn.Softmax(dim=1)(y_true), dim=1).data
    return recall_score(y_ground_cls.cpu(), y_pred_cls.cpu())

def auc(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    y_ground_cls = torch.argmax(nn.Softmax(dim=1)(y_true), dim=1).data
    return roc_auc_score(y_ground_cls.cpu(), y_pred_cls.cpu())

def f1(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    y_ground_cls = torch.argmax(nn.Softmax(dim=1)(y_true), dim=1).data
    return f1_score(y_ground_cls.cpu(), y_pred_cls.cpu())

def precision(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    y_ground_cls = torch.argmax(nn.Softmax(dim=1)(y_true), dim=1).data
    return precision_score(y_ground_cls.cpu(), y_pred_cls.cpu())

def specificity(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    y_ground_cls = torch.argmax(nn.Softmax(dim=1)(y_true), dim=1).data
    tn, fp, fn, tp = confusion_matrix(y_ground_cls.cpu(), y_pred_cls.cpu()).ravel()
    return tn/(tn+fp)

def false_alarm(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    y_ground_cls = torch.argmax(nn.Softmax(dim=1)(y_true), dim=1).data
    tn, fp, fn, tp = confusion_matrix(y_ground_cls.cpu(), y_pred_cls.cpu()).ravel()
    return fp/(fp+tn)

def tn(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    y_ground_cls = torch.argmax(nn.Softmax(dim=1)(y_true), dim=1).data
    tn, _, _, _ = confusion_matrix(y_ground_cls.cpu(), y_pred_cls.cpu()).ravel()
    return tn

def fp(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    y_ground_cls = torch.argmax(nn.Softmax(dim=1)(y_true), dim=1).data
    _, fp, _, _ = confusion_matrix(y_ground_cls.cpu(), y_pred_cls.cpu()).ravel()
    return fp

def fn(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    y_ground_cls = torch.argmax(nn.Softmax(dim=1)(y_true), dim=1).data
    _, _, fn, _ = confusion_matrix(y_ground_cls.cpu(), y_pred_cls.cpu()).ravel()
    return fn

def tp(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    y_ground_cls = torch.argmax(nn.Softmax(dim=1)(y_true), dim=1).data
    _, _, _, tp = confusion_matrix(y_ground_cls.cpu(), y_pred_cls.cpu()).ravel()
    return tp

model = torch.load('./saresnet_convlstm.pt').to(device)
print(type(model))
model.eval()
with torch.no_grad():
    tifDir = './tif/in'
    tifName = 'composite' + 'yyyyddd' + '.tif'
    absTifName = os.path.join(tifDir, tifName)
    raster_ds = gdal.Open(absTifName, gdal.GA_ReadOnly)

    outDriver = gdal.GetDriverByName("GTiff")
    bands = 1
    col = 1733
    row = 1623
    dataType = gdal.GDT_Float32
    outPath1 = './saresnet-convlstm.tif'

    outDataset1 = outDriver.Create(outPath1,col,row,bands,dataType)
    template_path = './dem.tif'
    template_ds = gdal.Open(template_path,gdal.GA_ReadOnly)
    geoTrans = template_ds.GetGeoTransform()
    outDataset1.SetGeoTransform(geoTrans)
    proj = template_ds.GetProjection()
    outDataset1.SetProjection(proj)

    outBand1 = outDataset1.GetRasterBand(1)
    outArray1 = outBand1.ReadAsArray()

    fireMap = []
    windowsize = 25
    height = 1623-25
    width = 1733-25


    for h in range(height):
        print(h)
        for w in range(width):
            curData = []
            dataset_array = raster_ds.ReadAsArray(w, h, windowsize,windowsize)
            curData.append(dataset_array)
            testData = np.array(curData)
            testData = torch.from_numpy(testData)
            testData = testData.unsqueeze(0).to(device)
            testData = testData[:,0,:,:,:]
            output = model(testData).cpu()
            probability = F.softmax(output).detach().numpy()[0]
            fireMap.append(probability[1])

    fireMapArray = np.array(fireMap).reshape(1623-25,1733-25)

    fireMapArray = np.pad(fireMapArray,((12,12),(12,12)),'constant',constant_values=np.nan)
    fireMapArray[np.isnan(fireMapArray)] = 0
    outArray1 = fireMapArray.copy()
    outBand1.WriteArray(outArray1)

    print('finished')

    outDataset = None
    raster_ds = None