import torch
import torch.nn as nn


class DilatedLSTM(nn.Module):
    """
    self.index: a simple arange used for indexing the hx, cx tensors.
    self.dilation: incremental dilation index to be added to self.index
    """

    def __init__(self, input_size, hidden_size, r=10):
        super().__init__()
        self.r = r
        self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.index = torch.arange(0, r * hidden_size, r)
        self.dilation = 0

    def forward(self, state, hidden):
        d_idx = self.dilation + self.index
        self.dilation = (self.dilation + 1) % self.r

        hx, cx = hidden
        hx_, cx_ = self.rnn(state, (hx[:, d_idx], cx[:, d_idx]))
        hx[:, d_idx] = hx_
        cx[:, d_idx] = cx_

        return hx[:, d_idx], (hx, cx)


if "__main__" == __name__:
    r = 4

    d_LSTM = DilatedLSTM(256, 64, r)
    hx = cx = torch.zeros(5, r * 256, requires_grad=True)
    hidden = (hx, cx)

    for _ in range(11):
        x = torch.ones(5, 256)
        output, hidden = d_LSTM(x, hidden)
        assert output.requires_grad is hidden[0].requires_grad \
            is hidden[1].requires_grad is True, "Gradient not working."
