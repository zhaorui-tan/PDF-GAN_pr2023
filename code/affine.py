import torch
import torch.nn as nn
from collections import OrderedDict

'''
modified form 
CBN (Conditional Batch Normalization layer)
    uses an MLP to predict the beta and gamma parameters in the batch norm equation
    Reference : https://papers.nips.cc/paper/7237-modulating-early-visual-processing-by-language.pdf
'''


class CBN(nn.Module):

    def __init__(self, cond_size, in_channel, out_channel, use_betas=True, use_gammas=True, eps=1.0e-5):
        super(CBN, self).__init__()

        self.cond_size = cond_size  # size of the lstm emb which is input to MLP
        self.in_channel = in_channel  # size of hidden layer of MLP  # in_channel
        self.out_channel = out_channel  # output of the MLP - for each channel  # out_channel
        self.use_betas = use_betas
        self.use_gammas = use_gammas

        self.batch_size = None
        self.channels = None
        self.height = None
        self.width = None

        # beta and gamma parameters for each channel - defined as trainable parameters
        # self.betas = nn.Parameter(torch.zeros(self.batch_size, self.channels).cuda())
        # self.gammas = nn.Parameter(torch.ones(self.batch_size, self.channels).cuda())
        self.betas = nn.Parameter(torch.zeros(1, self.out_channel))
        self.gammas = nn.Parameter(torch.ones(1, self.out_channel))
        self.eps = eps

        # MLP used to predict betas and gammas
        self.fc_gamma = nn.Sequential(
            nn.Linear(self.cond_size, self.in_channel),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channel, self.out_channel),
        )

        self.fc_beta = nn.Sequential(
            nn.Linear(self.cond_size, self.in_channel),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channel, self.out_channel),
        )

        # initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0.1)

    '''
    Predicts the value of delta beta and delta gamma for each channel
    Arguments:
        lstm_emb : lstm embedding of the question
    Returns:
        delta_betas, delta_gammas : for each layer
    '''

    def create_cbn_input(self, cond_emb):

        if self.use_betas:
            delta_betas = self.fc_beta(cond_emb)
        else:
            delta_betas = torch.zeros(self.batch_size, self.channels)

        if self.use_gammas:
            delta_gammas = self.fc_gamma(cond_emb)
        else:
            delta_gammas = torch.zeros(self.batch_size, self.channels)

        return delta_betas, delta_gammas

    '''
    Computer Normalized feature map with the updated beta and gamma values
    Arguments:
        feature : feature map from the previous layer
        cond_emb : lstm embedding of the question
    Returns:
        out : beta and gamma normalized feature map
        cond_emb : lstm embedding of the question (unchanged)
    Note : cond_emb needs to be returned since CBN is defined within nn.Sequential
           and subsequent CBN layers will also require lstm question embeddings
    '''

    def forward(self, feature, cond_emb):
        self.batch_size, self.channels, self.height, self.width = feature.data.shape

        # get delta values
        delta_betas, delta_gammas = self.create_cbn_input(cond_emb)

        # betas_cloned = self.betas.clone()
        # gammas_cloned = self.gammas.clone()

        betas_cloned = self.betas.repeat(self.batch_size, 1)
        gammas_cloned = self.gammas.repeat(self.batch_size, 1)
        # update the values of beta and gamma
        betas_cloned += delta_betas
        gammas_cloned += delta_gammas

        # get the mean and variance for the batch norm layer
        # feature: (batch, channel, height, width) -> mean, var: (channel)
        feature_tmp = feature.permute(1, 0, 2, 3).contiguous()
        batch_mean = torch.mean(feature_tmp.view(self.channels, -1), 1)
        batch_var = torch.var(feature_tmp.view(self.channels, -1), 1)

        batch_mean = batch_mean.repeat(self.batch_size, 1)
        batch_var = batch_var.repeat(self.batch_size, 1)

        def extend2map(x, height, width):
            x = torch.stack([x] * height, dim=2)
            x = torch.stack([x] * width, dim=3)
            return x

        batch_mean = extend2map(batch_mean, self.height, self.width)
        batch_var = extend2map(batch_var, self.height, self.width)

        # extend the betas and gammas of each channel across the height and width of feature map
        betas_expanded = extend2map(betas_cloned, self.height, self.width)
        gammas_expanded = extend2map(gammas_cloned, self.height, self.width)

        # normalize the feature map
        feature_normalized = (feature - batch_mean) / torch.sqrt(batch_var + self.eps)

        # get the normalized feature map with the updated beta and gamma values
        out = torch.mul(feature_normalized, gammas_expanded) + betas_expanded

        return out, cond_emb


class affine(nn.Module):

    def __init__(self, num_features):
        super(affine, self).__init__()
        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(512, 512)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(512, num_features)),
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(512, 512)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(512, num_features)),
        ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
        # x [batch, channel, w, h]
        # y [batch, emb_dim]
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias


class affine_word(nn.Module):
    def __init__(self, num_features):
        super(affine_word, self).__init__()
        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(512, 512)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(512, num_features)),
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(512, 512)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(512, num_features)),
        ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None, mask=None):
        cond = y
        weight = self.fc_gamma(cond)
        bias = self.fc_beta(cond)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = [cond.shape[0], cond.shape[1], x.shape[-3], x.shape[-2], x.shape[-1]]
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        x = x.unsqueeze(1).expand(size)

        x = weight * x + bias
        x = torch.sum(x, dim=1) / cond.shape[1]
        return x


if __name__ == '__main__':
    # model = CBN(5, 2, 2)
    # print(model)
    cond = torch.tensor([[1, 1, 1, 1, 1], [1., 1, 1, 1, 1], ])
    feature = torch.randn(2, 2, 8, 8)

    y = torch.ones((2, 4, 256))
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]])
    # y = torch.zeros((4, 256))
    # print(y.shape)
    # print(feature.shape)
    # normalised_feature, _ = model(feature, cond)
    # print(normalised_feature.shape)
    # print(normalised_feature)

    aff = affine_word(2)
    x = aff(feature, y, mask)
    # print(x)
