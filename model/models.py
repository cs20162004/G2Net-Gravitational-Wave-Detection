from torch import nn
import sys
sys.path.append("/home/hero/Downloads/Ali/practice/g2net/pytorch-image-models")
import timm


class Efficientnet7(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.model = timm.create_model("tf_efficientnet_b7_ns", pretrained=pretrained, in_chans=1)
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, 1)

    def forward(self, x):
        output = self.model(x)
        return output