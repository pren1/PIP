r"""
    Utils for RNN including networks, datasets, and loss wrappers.
"""


__all__ = ['RNN', 'RNNWithInit']


import os
import torch.utils.data
from torch.nn.functional import relu
from torch.nn.utils.rnn import *
import pdb

class RNN(torch.nn.Module):
    r"""
    An RNN net including a linear input layer, an RNN, and a linear output layer.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_rnn_layer: int,
                 simplified: bool,
                 rnn_type='lstm', bidirectional=False, dropout=0., load_weight_file: str = None):
        r"""
        Init an RNN.

        :param input_size: Input size.
        :param output_size: Output size.
        :param hidden_size: Hidden size for RNN.
        :param num_rnn_layer: Number of RNN layers.
        :param rnn_type: Select from 'rnn', 'lstm', 'gru'.
        :param bidirectional: Whether if the RNN is bidirectional.
        :param dropout: Dropout after the input linear layer and in the rnn.
        :param load_weight_file: If not None and exists, weights will be loaded.
        """
        super().__init__()
        self.rnn = getattr(torch.nn, rnn_type.upper())(hidden_size, hidden_size, num_rnn_layer,
                                                       bidirectional=bidirectional, dropout=dropout)
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.previous_hidden_states = None
        self.simplified = simplified # Whether we do rnn per step
        if load_weight_file and os.path.exists(load_weight_file):
            self.load_state_dict(torch.load(load_weight_file))
            self.eval()

    def forward(self, x, init=None):
        r"""
        Forward.

        :param x: A list in length [batch_size] which contains tensors in shape [num_frames, input_size].
        :param init: Initial hidden states.
        :return: A list in length [batch_size] which contains tensors in shape [num_frames, output_size].
        """
        if not self.simplified:
            length = [_.shape[0] for _ in x]
            x = self.dropout(relu(self.linear1(pad_sequence(x))))

            # Output shape of self.rnn(pack_padded_sequence(x, length, enforce_sorted=False), init) is:
            # [2, 1, 256], 1 means only 1 sequence is here~, 2 means there are h and c
            x = self.rnn(pack_padded_sequence(x, length, enforce_sorted=False), init)[0]
            # pad_packed_sequence(x)[0] has shape:
            # [1176, 1, 256], [0] picks the data from padded sequences
            x = self.linear2(pad_packed_sequence(x)[0])
            return [x[:l, i].clone() for i, l in enumerate(length)]
        else:
            # Here we implement the single step
            # random_tensor = torch.randn(72)
            # Shape of x should be: [1,xshape]
            x = x.unsqueeze(0) # add a dim, this is required by the rnn
            x = self.dropout(relu(self.linear1(x)))
            if self.previous_hidden_states is None:
                output, self.previous_hidden_states = self.rnn(x, init)
            else:
                output, self.previous_hidden_states = self.rnn(x, self.previous_hidden_states)
            output = self.linear2(output)
            return output.squeeze()

class RNNWithInit(RNN):
    r"""
    RNN with the initial hidden states regressed from the first output.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_rnn_layer: int,
                 simplified: bool,
                 rnn_type='lstm', bidirectional=False, dropout=0., load_weight_file: str = None):
        r"""
        Init an RNNWithInit net.

        :param input_size: Input size.
        :param output_size: Output size.
        :param hidden_size: Hidden size for RNN.
        :param num_rnn_layer: Number of RNN layers.
        :param rnn_type: Select from 'rnn', 'lstm', 'gru'.
        :param bidirectional: Whether if the RNN is bidirectional.
        :param dropout: Dropout after the input linear layer and in the rnn.
        :param load_weight_file: If not None and exists, weights will be loaded.
        """
        assert rnn_type == 'lstm' and bidirectional is False
        super().__init__(input_size, output_size, hidden_size, num_rnn_layer, simplified, rnn_type, bidirectional, dropout)
        self.simplified = simplified
        self.init_net = torch.nn.Sequential(
            # Here the input size is output_size, makes sense!
            torch.nn.Linear(output_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size * num_rnn_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size * num_rnn_layer, 2 * (2 if bidirectional else 1) * num_rnn_layer * hidden_size)
        )

        if load_weight_file and os.path.exists(load_weight_file):
            self.load_state_dict(torch.load(load_weight_file))
            self.eval()

    def forward(self, x, _=None):
        r"""
        Forward.

        :param x: A list in length [batch_size] which contains 2-tuple
                  (Tensor[num_frames, input_size], Tensor[output_size]).
        :param _: Not used.
        :return: A list in length [batch_size] which contains tensors in shape [num_frames, output_size].
        """
        nd, nh = self.rnn.num_layers * (2 if self.rnn.bidirectional else 1), self.rnn.hidden_size
        if not self.simplified:
            x, x_init = list(zip(*x))
            h, c = self.init_net(torch.stack(x_init)).view(-1, 2, nd, nh).permute(1, 2, 0, 3)
        else:
            x, x_init = x
            h, c = self.init_net(x_init).view(-1,nd,nh)
        return super(RNNWithInit, self).forward(x, (h, c))
