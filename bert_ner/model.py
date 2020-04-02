import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class NERModel(nn.Module):

    def __init__(self, language_model, nout, ninp=200, dropout=0.2):
        super().__init__()
        self.language_model = language_model
        self.model_type = 'BERTNER'
        self.dropout = dropout
        nhead = 2
        nhid = 200
        nlayers = 2
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder_in = TransformerEncoder(encoder_layers, nlayers)

        self.input_linear = nn.Linear(768, ninp)
        self.linear_out = nn.Linear(ninp, nout)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_linear.weight.data.uniform_(-initrange, initrange)
        self.linear_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        output = self.language_model(src)[0]
        output = self.input_linear(output)
        output = F.relu(output)
        output = self.transformer_encoder_in(output)

        output = self.linear_out(output)
        out = F.softmax(output, dim=-1)

        return out
