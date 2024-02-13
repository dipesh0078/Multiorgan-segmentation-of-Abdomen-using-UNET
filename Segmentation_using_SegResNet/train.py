from monai.networks.nets import SegResNet
from monai.networks.layers import Norm
from enum import Enum
from monai.losses import DiceLoss,GeneralizedDiceLoss

import torch
from preprocess import prepare
from utilities import train


data_dir = 'D:\\minor test1\\Data_Train_Test'
model_dir = 'D:\\minor test1\\SegResNet' 
data_in = prepare(data_dir, cache=True)

device = torch.device("cuda:0")
model = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=5,
    act=('RELU',{'inplace':True}),
    norm=Norm.BATCH,
).to(device)

#loss_function=GeneralizedDiceLoss(include_background=True,to_onehot_y=True,softmax=True,w_type="square",smooth_nr=1e-05, smooth_dr=1e-05,batch=False)
#loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))
loss_function = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5, amsgrad=True)

if __name__ == '__main__':
    train(model, data_in, loss_function, optimizer, 200, model_dir)