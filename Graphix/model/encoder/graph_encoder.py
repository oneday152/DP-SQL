#coding=utf8
import torch
import torch.nn as nn
from Graphix.model.encoder.graph_input import *
from Graphix.model.encoder.rgatsql import RGATSQL
from Graphix.model.encoder.graph_output import *
from Graphix.model.model_utils import Registrable

@Registrable.register('encoder_text2sql')
class Text2SQLEncoder(nn.Module):

    def __init__(self, args):
        super(Text2SQLEncoder, self).__init__()
        self.input_layer = RGATInputLayer(args)
        self.hidden_layer = Registrable.by_name(args.model)(args)
        self.output_layer = RGATOutputlayer(args)

    def forward(self, batch):
        outputs = self.input_layer(batch)
        outputs = self.hidden_layer(outputs, batch)
        return self.output_layer(outputs, batch)
