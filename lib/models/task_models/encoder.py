import sys
import torch
import torch.nn as nn
from pathlib import Path
from torchsummary import summary
import torchvision.models as models

lib_dir = (Path(__file__).parent / '..' / '..').resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from models.net_infer.net_macro import MacroNet
from models.net_infer.cell_micro import ResNetBasicblock
from models.net_search.net_search_darts_v1 import SuperNetDartsV1


class FFEncoder(nn.Module):
    """Encoder class for the definition of backbone including resnet50 and MacroNet()"""
    def __init__(self, encoder_str, task_name=None):
        super(FFEncoder, self).__init__()
        self.encoder_str = encoder_str

        # Initialize network
        if self.encoder_str == 'resnet50':
            self.network = models.resnet50() # resnet50: Bottleneck, [3,4,6,3]
            # Adjust according to task
            if task_name in ['autoencoder', 'normal', 'inpainting', 'segmentsemantic']:
                self.network.inplanes = 1024
                self.network.layer4 = self.network._make_layer(
                    models.resnet.Bottleneck, 512, 3, stride=1, dilate=False)
                self.network = nn.Sequential(
                    *list(self.network.children())[:-2],
                )
            else:
                self.network = nn.Sequential(*list(self.network.children())[:-2])
        elif self.encoder_str == '64-41414-super_0123':
            self.network = SuperNetDartsV1(encoder_str, structure='backbone')
        else:
            self.network = MacroNet(encoder_str, structure='backbone')

    def forward(self, x):
        x = self.network(x)
        return x


if __name__ == "__main__":
    # net = FFEncoder("64-41414-3_33_333", 'segmentsemantic').cuda()
    net = FFEncoder("resnet50", 'autoencoder').cuda()
    # x = torch.randn([2, 3, 256, 256])
    # print(net(x).shape)
    # print(net)
    summary(net, (3, 256, 256))
