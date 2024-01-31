import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet, Bottleneck
from torch import Tensor
from models.SENet import SELayer
from models.SKNet import SKConv
from models.CBAM import CBAMBlock
from models.BAM import BAM
from models.non_local import NONLocalBlock2D
from models.GCNet import ContextBlock


class ResNetAttention(ResNet):
    def __init__(self, model_name="resnet18", classes=10, use_attention=False, attention_name=None):
        assert attention_name in ["SENet", "SKNet", "CBAM", "BAM", "NL", "GCNet"]
        self.model_name = model_name
        self.use_attention = use_attention
        self.attention_name = attention_name
        self.classes = classes

        if self.model_name == "resnet18":
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif self.model_name == "resnet34":
            block = BasicBlock
            layers = [3, 4, 6, 3]
        elif self.model_name == "resnet50":
            block = Bottleneck
            layers = [3, 4, 6, 3]
        super(ResNetAttention, self).__init__(block, layers)

        if self.model_name in ["resnet18", "resnet34"]:
            self.fc = nn.Linear(in_features=512, out_features=self.classes)
        elif self.model_name in ["resnet50"]:
            self.fc = nn.Linear(in_features=2048, out_features=self.classes)
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_in", nonlinearity="relu")

        if self.use_attention:
            if self.model_name == "resnet18":
                layer_channel = [64, 128, 256, 512]
            if self.model_name == "resnet34":
                layer_channel = [64, 128, 256, 512]
            if self.model_name == "resnet50":
                layer_channel = [256, 512, 1024, 2048]

            if self.attention_name == "SENet":
                self.attention_1 = SELayer(channel=layer_channel[0])
                self.attention_2 = SELayer(channel=layer_channel[1])
                self.attention_3 = SELayer(channel=layer_channel[2])
                self.attention_4 = SELayer(channel=layer_channel[3])
            if self.attention_name == "SKNet":
                self.attention_1 = SKConv(layer_channel[0], WH=1, M=2, G=1, r=2)
                self.attention_2 = SKConv(layer_channel[1], WH=1, M=2, G=1, r=2)
                self.attention_3 = SKConv(layer_channel[2], WH=1, M=2, G=1, r=2)
                self.attention_4 = SKConv(layer_channel[3], WH=1, M=2, G=1, r=2)
            if self.attention_name == "CBAM":
                self.attention_1 = CBAMBlock(layer_channel[0], layer_channel[0])
                self.attention_2 = CBAMBlock(layer_channel[1], layer_channel[1])
                self.attention_3 = CBAMBlock(layer_channel[2], layer_channel[2])
                self.attention_4 = CBAMBlock(layer_channel[3], layer_channel[3])
            if self.attention_name == "BAM":
                self.attention_1 = BAM(layer_channel[0])
                self.attention_2 = BAM(layer_channel[1])
                self.attention_3 = BAM(layer_channel[2])
                self.attention_4 = BAM(layer_channel[3])
            if self.attention_name == "NL":
                self.attention_1 = NONLocalBlock2D(layer_channel[0], sub_sample=False, bn_layer=True)
                self.attention_2 = NONLocalBlock2D(layer_channel[1], sub_sample=False, bn_layer=True)
                self.attention_3 = NONLocalBlock2D(layer_channel[2], sub_sample=False, bn_layer=True)
                self.attention_4 = NONLocalBlock2D(layer_channel[3], sub_sample=False, bn_layer=True)
            if self.attention_name == "GCNet":
                self.attention_1 = ContextBlock(inplanes=layer_channel[0], ratio=1. / 16., pooling_type='att')
                self.attention_2 = ContextBlock(inplanes=layer_channel[1], ratio=1. / 16., pooling_type='att')
                self.attention_3 = ContextBlock(inplanes=layer_channel[2], ratio=1. / 16., pooling_type='att')
                self.attention_4 = ContextBlock(inplanes=layer_channel[3], ratio=1. / 16., pooling_type='att')

    def load_state(self, model_path, map_location=None):
        state_dict = torch.load(model_path, map_location=map_location)

        for k in list(state_dict.keys()):
            # print(k)
            if k in ["fc.weight", "fc.bias"]:
                state_dict.pop(k)

        self.load_state_dict(state_dict, strict=False)
        return state_dict

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.use_attention:
            x = self.attention_1(x)
            print("att1: {}".format(x.shape))
        x = self.layer2(x)
        if self.use_attention:
            x = self.attention_2(x)
            print("att2: {}".format(x.shape))
        x = self.layer3(x)
        if self.use_attention:
            x = self.attention_3(x)
            print("att3: {}".format(x.shape))
        x = self.layer4(x)
        if self.use_attention:
            x = self.attention_4(x)
            print("att4: {}".format(x.shape))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        print("flatten: {}".format(x.shape))
        x = self.fc(x)

        return x

    def forward(self, x):
        out = self._forward_impl(x)
        return out


if __name__ == '__main__':
    net = ResNetAttention(model_name="resnet18", classes=100, use_attention=True, attention_name="SENet")
    net.load_state("../pretrained_models/hub/checkpoints/resnet18-f37072fd.pth", map_location=None)
    # print(net)

    x = torch.randn(4, 3, 128, 128)
    out = net(x)
    print(out.shape)
    for k, v in net.state_dict().items():
        print(k, id(v))
    print("#" * 100)

    for param in net.parameters():
        print(id(param))
    print("#" * 100)

    train_parameters_id = []
    pretrained_parameters_id = []
    for name, param in net.named_parameters(recurse=True):
        if name in net.state_dict().keys():
            pretrained_parameters_id.append(id(param))
        else:
            train_parameters_id.append(id(param))
    print(train_parameters_id)
    print(pretrained_parameters_id)

    # from torchsummary import summary
    # summary(net, input_size=(4, 3, 128, 128), batch_size=0)
