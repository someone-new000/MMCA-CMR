import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x).view(x.shape[0], -1)
        features = self.fc_features(features)
        return features


class ImgNN1(nn.Module):
    """Network to learn image representations"""
    def __init__(self, input_dim=4096, output_dim=1024):   #  1024
        super(ImgNN1, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out


class TextNN1(nn.Module):
    """Network to learn text representations"""
    def __init__(self, input_dim=1024, output_dim=1024):   #   1024
        super(TextNN1, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out


class ImgNN2(nn.Module):
    """Network to learn image representations"""

    def __init__(self, input_dim=2048, output_dim=2048):
        super(ImgNN2, self).__init__()
        self.denseL2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL2(x))
        return out


class TextNN2(nn.Module):
    """Network to learn text representations"""

    def __init__(self, input_dim=2048, output_dim=2048):
        super(TextNN2, self).__init__()
        self.denseL2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL2(x))
        return out


class ImgNN3(nn.Module):
    """Network to learn image representations"""

    def __init__(self, input_dim=2048, output_dim=1024):
        super(ImgNN3, self).__init__()
        self.denseL3 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL3(x))
        return out


class TextNN3(nn.Module):
    """Network to learn text representations"""

    def __init__(self, input_dim=2048, output_dim=1024):
        super(TextNN3, self).__init__()
        self.denseL3 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL3(x))
        return out


class ImgNN4(nn.Module):
    """Network to learn image representations"""

    def __init__(self, input_dim=1024, output_dim=1024):
        super(ImgNN4, self).__init__()
        self.denseL4 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL4(x))
        return out


class TextNN4(nn.Module):
    """Network to learn text representations"""

    def __init__(self, input_dim=1024, output_dim=1024):
        super(TextNN4, self).__init__()
        self.denseL4 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL4(x))
        return out


class IDCM_NN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, img_input_dim=4096, img_output_dim=2048, output_one_dim=1024,
                 text_input_dim=1024, text_output_dim=2048, minus_one_dim=512, output_dim=10):
        super(IDCM_NN, self).__init__()
        self.img_net = ImgNN1(img_input_dim, img_output_dim)#img_output_dim)
        self.text_net = TextNN1(text_input_dim, text_output_dim)#text_output_dim)
        self.img_net2 = ImgNN2(img_output_dim, output_one_dim)
        self.text_net2 = TextNN2(text_output_dim, output_one_dim)
#        self.linearLayer3 = nn.Linear(img_output_dim, img_output_dim)
#        self.linearLayer3 = nn.Linear(text_output_dim, text_output_dim)
#        self.img_net3 = ImgNN3(img_output_dim, output_one_dim)
#        self.text_net3 = TextNN3(text_output_dim, output_one_dim)
#        self.img_net4 = ImgNN4(output_one_dim, minus_one_dim)
#        self.text_net4 = TextNN4(output_one_dim, minus_one_dim)
        self.linearLayer = nn.Linear(output_one_dim, output_one_dim)#minus_one_dim, minus_one_dim)
        self.linearLayer2 = nn.Linear(output_one_dim, output_dim)#minus_one_dim, output_dim)

    def forward(self, img, text):
        view1_feature = self.img_net(img)
        view2_feature = self.text_net(text)

        view1_feature = self.img_net2(view1_feature)
        view2_feature = self.text_net2(view2_feature)

#        view1_feature = self.linearLayer3(view1_feature)
#        view2_feature = self.linearLayer3(view2_feature)

#        view1_feature = self.img_net3(view1_feature)
#        view2_feature = self.text_net3(view2_feature)

#        view1_feature = self.img_net4(view1_feature)
#        view2_feature = self.text_net4(view2_feature)

        view1_feature = self.linearLayer(view1_feature)
        view2_feature = self.linearLayer(view2_feature)

        view1_predict = self.linearLayer2(view1_feature)
        view2_predict = self.linearLayer2(view2_feature)
        return view1_feature, view2_feature, view1_predict, view2_predict
