from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv2d, AvgPool2d, Linear

from util import split


class LeNetFromPaper(nn.Module):

    def __init__(self):
        super(LeNetFromPaper, self).__init__()
        self.c1 = Conv2d(1, 6, 5)
        self.s2 = AvgPool2d(2, 2)
        self.c3 = Conv2d(6, 16, 5)
        self.s4 = AvgPool2d(2, 2)
        self.f5 = Linear(16 * 5 * 5, 120)
        self.f6 = Linear(120, 84)
        self.output = Linear(84, 10)

    def forward(self, x):
        x = self.s2(F.tanh(self.c1(x)))
        x = self.s4(F.tanh(self.c3(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.tanh(self.f5(x))
        x = F.tanh(self.f6(x))
        x = self.output(x)
        return x

    @staticmethod
    def network_from_weight_list(weight_list):
        custom_state_dict = LeNetFromPaper.list_to_state_dict(weight_list)
        model = LeNetFromPaper()
        model.load_state_dict(custom_state_dict)
        return model

    @staticmethod
    def list_to_state_dict(weight_list):
        cursor = 0
        # c1 weigths
        c1_weight_number = 6 * 1 * 5 * 5
        c1_weights_from_list = weight_list[cursor:cursor + c1_weight_number]
        c1_weight_params = Tensor(
            list(split(list(split(list(split(list(split(c1_weights_from_list, 5)), 5)), 1)), 6))[0])
        cursor += c1_weight_number

        # c1 bias
        c1_bias_number = 6
        c1_bias_params = Tensor(weight_list[cursor:cursor + c1_bias_number])
        cursor += c1_bias_number

        # c3 weights
        c3_weight_number = 16 * 6 * 5 * 5
        c3_weights_from_list = weight_list[cursor:cursor + c3_weight_number]
        c3_weight_params = Tensor(
            list(split(list(split(list(split(list(split(c3_weights_from_list, 5)), 5)), 6)), 16))[0])
        cursor += c3_weight_number

        # c3 bias
        c3_bias_number = 16
        c3_bias_params = Tensor(weight_list[cursor:cursor + c3_bias_number])
        cursor += c3_bias_number

        # f5 weights
        f5_weight_number = 120 * 400
        f5_weight_from_list = weight_list[cursor:cursor + f5_weight_number]
        f5_weight_params = Tensor(list(split(list(split(f5_weight_from_list, 400)), 120))[0])
        cursor += f5_weight_number

        f5_bias_number = 120
        f5_bias_params = Tensor(weight_list[cursor:cursor + f5_bias_number])
        cursor += f5_bias_number

        # f6 weights
        f6_weight_number = 84 * 120
        f6_weight_from_list = weight_list[cursor:cursor + f6_weight_number]
        f6_weight_params = Tensor(list(split(list(split(f6_weight_from_list, 120)), 84))[0])
        cursor += f6_weight_number

        f6_bias_number = 84
        f6_bias_params = Tensor(weight_list[cursor:cursor + f6_bias_number])
        cursor += f6_bias_number

        # output
        output_weight_number = 10 * 84
        output_weight_from_list = weight_list[cursor:cursor + output_weight_number]
        output_weight_params = Tensor(list(split(list(split(output_weight_from_list, 84)), 10))[0])
        cursor += output_weight_number

        output_bias_number = 10
        output_bias_params = Tensor(weight_list[cursor:cursor + output_bias_number])
        cursor += output_bias_number

        return OrderedDict([
            ('c1.weight', c1_weight_params),
            ('c1.bias', c1_bias_params),
            ('c3.weight', c3_weight_params),
            ('c3.bias', c3_bias_params),
            ('f5.weight', f5_weight_params),
            ('f5.bias', f5_bias_params),
            ('f6.weight', f6_weight_params),
            ('f6.bias', f6_bias_params),
            ('output.weight', output_weight_params),
            ('output.bias', output_bias_params)
        ])
