
"""
Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url


from .method.util import normalize_tensor

from .util import initialize_weights

import matplotlib.pyplot as plt

from torchvision import datasets, models, transforms

import copy

__all__ = ['vgg16']

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
}

def normalize_tensor(x):
    channel_vector = x.view(x.size()[0], x.size()[1], -1)
    minimum, _ = torch.min(channel_vector, dim=-1, keepdim=True)
    maximum, _ = torch.max(channel_vector, dim=-1, keepdim=True)
    normalized_vector = torch.div(channel_vector - minimum, maximum - minimum)
    normalized_tensor = normalized_vector.view(x.size())
    return normalized_tensor

def initialize_weights(modules, init_mode):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            if init_mode == 'he':
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif init_mode == 'xavier':
                nn.init.xavier_uniform_(m.weight.data)
            else:
                raise ValueError('Invalid init_mode {}'.format(init_mode))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

configs_dict = {
    'mymodel47': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 'M', 512, 512, 512, 'I'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'I', 512,
                  512, 512, 'I', 512, 512, 512, 'I'],
    }
}

class MyModel2(nn.Module):
  def __init__(self):
      super(MyModel2, self).__init__()
      #self.attention = None

  def forward(self, input_):
      if not self.training:
          return input_
      else:
          attention = torch.mean(input_, dim=1, keepdim=True)
          importance_map = torch.sigmoid(attention)
          return input_.mul(importance_map)

class myModel47(nn.Module):
    def __init__(self, features, num_classes=14, **kwargs):
        super(myModel47, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv8 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv9 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv10 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv11 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        #self.conv12 = nn.Conv2d(512,  1024, kernel_size=3, padding=1) 
        #self.conv13 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.mymod2 = MyModel2()
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024, num_classes)
        initialize_weights(self.modules(), init_mode='he')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        x1 = self.features(x)
        x1 = self.conv6(x1)
        x1 = self.relu(x1)
        x1 = self.mymod2(x1)
        x1 = self.conv7(x1)
        x1 = self.relu(x1)
        
        x2 = self.features(x)
        x2 = self.conv8(x2)
        x2 = self.relu(x2)
        x2 = self.conv9(x2)
        x2 = self.relu(x2)

        x3 = self.features(x)
        x3 = self.conv10(x3)
        x3 = self.relu(x3)
        x3 = self.conv11(x3)
        x3 = self.relu(x3)

        x4 = self.features(x)
        x4 = self.conv12(x4)
        x4 = self.relu(x4)
        x4 = self.conv13(x4)
        x4 = self.relu(x4)
        
        x = torch.max(x1 ,x2)
        x = torch.max(x ,x3)
        x = torch.max(x,x4)
              
        if return_cam:
            x = x1.detach().clone()
            x = x + x2.detach().clone()
            x = x + x3.detach().clone()
            x = x + x4.detach().clone()
            x = normalize_tensor(x.detach().clone())
            x = x[range(batch_size), labels]
            return x
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return {'logits': x}
    
def adjust_pretrained_model(pretrained_model, current_model):
    def _get_keys(obj, split):
        keys = []
        iterator = obj.items() if split == 'pretrained' else obj
        for key, _ in iterator:
            if key.startswith('features.'):
                keys.append(int(key.strip().split('.')[1].strip()))
        return sorted(list(set(keys)), reverse=True)

    def _align_keys(obj, key1, key2):
        for suffix in ['.weight', '.bias']:
            old_key = 'features.' + str(key1) + suffix
            new_key = 'features.' + str(key2) + suffix
            obj[new_key] = obj.pop(old_key)
        return obj

    pretrained_keys = _get_keys(pretrained_model, 'pretrained')
    current_keys = _get_keys(current_model.named_parameters(), 'model')

    for p_key, c_key in zip(pretrained_keys, current_keys):
        pretrained_model = _align_keys(pretrained_model, p_key, c_key)

    return pretrained_model

def load_pretrained_model(model, architecture_type, path=None):
    if path is not None:
        state_dict = torch.load(os.path.join(path, 'vgg16.pth'))
    else:
        state_dict = load_url(model_urls['vgg16'], progress=True)

    if(architecture_type in ('mymodel36','mymodel37','mymodel38')):
        mymodel_state_dict = model.state_dict()
        mymodel3bweightassign(mymodel_state_dict, state_dict)
        state_dict = mymodel_state_dict
    else:
        if architecture_type == 'spg':
            state_dict = batch_replace_layer(state_dict)
        state_dict = remove_layer(state_dict, 'classifier.')
        state_dict = adjust_pretrained_model(state_dict, model)

    model.load_state_dict(state_dict, strict=False)
    return model


def make_layers(cfg, **kwargs):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M1':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'M2':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        elif v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'I':
            layers += [
                MyModel2()]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def set_parameter_requires_grad2(model):
  for i,j in enumerate(model.named_parameters()):
    if(i <=16):
      j[1].requires_grad = False
    
def vgg16(architecture_type, pretrained=False, pretrained_path=None):
    config_key = '28x28' 
    layers = make_layers(configs_dict[architecture_type][config_key])
    model = {'mymodel47': myModel47}[architecture_type](layers)
    if pretrained:
        model = load_pretrained_model(model, architecture_type,
                                      path=pretrained_path)
        if(architecture_type in ('mymodel47')):
          set_parameter_requires_grad2(model)  
    return model
