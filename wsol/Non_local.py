import torch
import torch.nn as nn

class Self_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // 8, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.mix_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.mix_conv.weight.data.fill_(0.0)
        # self.mix_conv.bias.data.fill_(0.0)
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(in_dim)
        nn.init.constant_(self.bn.weight, 0.0)
        nn.init.constant_(self.bn.bias, 0.0)
        self.relu = nn.ReLU()

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height)
        # out = self.relu(self.bn(self.mix_conv(out)))
        out = (self.bn(self.mix_conv(out)))
        out = out + x
        # print(self.gamma)
        return out

class Self_Attn_low(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn_low, self).__init__()
        self.channel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.mix_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.mix_conv.weight.data.fill_(0.0)
        # self.mix_conv.bias.data.fill_(0.0)
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(in_dim)
        nn.init.constant_(self.bn.weight, 0.0)
        nn.init.constant_(self.bn.bias, 0.0)
        self.relu = nn.ReLU()

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height)
        # out = self.relu(self.bn(self.mix_conv(out)))
        out = (self.bn(self.mix_conv(out)))
        out = out + x
        # print(self.gamma)
        return out

