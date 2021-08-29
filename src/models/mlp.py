import torch
import torch.nn as nn

# MLP:
# ----------------------------------------------
class Context_MLP(nn.Module):

    # ------------------------------------------
    def __init__(self, in_size):
        super().__init__()

        self.layer1 = nn.Linear((in_size*2), in_size)
        self.prelu1 = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(in_size)

        self.layer2 = nn.Linear(in_size, in_size)
        self.prelu2 = nn.PReLU()
        self.bn2 = nn.BatchNorm1d(in_size)

        self.layer3 = nn.Linear(in_size, in_size//2)
        self.prelu3 = nn.PReLU()
        self.bn3 = nn.BatchNorm1d(in_size//2)

        self.layer4_left = nn.Linear(in_size//2, 1)
        self.layer4_right = nn.Linear(in_size//2, 1)

        self.do = nn.Dropout(p=0.2)
    # ------------------------------------------

    # ------------------------------------------
    def forward(self, left_in , right_in):

        the_catted_one = torch.cat([left_in, right_in], dim=1)

        x = self.do(self.bn1(self.prelu1(self.layer1(the_catted_one))))
        x = self.do(self.bn2(self.prelu2(self.layer2(x))))
        x = self.do(self.bn3(self.prelu3(self.layer3(x))))
        left_out = self.layer4_left(x)
        right_out = self.layer4_right(x)

        return left_out, right_out
    # ------------------------------------------
# ----------------------------------------------
