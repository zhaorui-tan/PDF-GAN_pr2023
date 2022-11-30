import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from affine import affine, affine_word
from miscc.config import cfg
from gem import *


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


class NetG(nn.Module):
    def __init__(self, ngf=64, nz=100, nef=256, n_tasks=2):
        super(NetG, self).__init__()
        self.ngf = ngf
        self.n_tasks = n_tasks

        self.fc = nn.Linear(nz, ngf * 8 * 4 * 4)
        self.block0 = G_Block(ngf * 8, ngf * 8, use_sent_affine=True, use_word_affine=False)  # 4x4
        self.block1 = G_Block(ngf * 8, ngf * 8, use_sent_affine=True, use_word_affine=False)  # 4x4

        self.block2_g = G_Block(ngf * 8, ngf * 8, use_sent_affine=True, use_word_affine=False)  # 8x8
        self.block2_l = G_Block(ngf * 8, ngf * 8, use_sent_affine=False, use_word_affine=True)  # 8x8
        self.conv_2_1 = nn.Conv2d(ngf * 8 * 2, ngf * 8, 3, 1, 1)
        self.conv_2_2 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)

        self.block3_g = G_Block(ngf * 8, ngf * 8, use_sent_affine=True, use_word_affine=False)  # 16x16
        self.block3_l = G_Block(ngf * 8, ngf * 8, use_sent_affine=False, use_word_affine=True)  # 16x16
        self.conv_3_1 = nn.Conv2d(ngf * 8 * 2, ngf * 8, 3, 1, 1)
        self.conv_3_2 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)

        self.block4 = G_Block(ngf * 8, ngf * 4, use_sent_affine=True, use_word_affine=False)  # 32x32
        self.block5 = G_Block(ngf * 4, ngf * 2, use_sent_affine=True, use_word_affine=False)  # 64x64
        self.block6 = G_Block(ngf * 2, ngf * 1, use_sent_affine=True, use_word_affine=False)  # 128x128

        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, 3, 3, 1, 1),
            nn.Tanh(),
        )

        # GEM settings
        self.margin = cfg.GEM.memory_strength
        self.ce = nn.CrossEntropyLoss()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), self.n_tasks)

    def forward(self, x, c, words_embs=None, mask=None, ):

        out = self.fc(x)
        out = out.view(x.size(0), 8 * self.ngf, 4, 4)
        out = self.block0(out, c, words_embs, mask)
        out = F.interpolate(out, scale_factor=2)
        out = self.block1(out, c, words_embs, mask)

        out = F.interpolate(out, scale_factor=2)
        out_l = self.block2_l(out, c, words_embs, mask)
        out_g = self.block2_g(out, c, words_embs, mask)
        out_c = torch.cat((out_l, out_g,), dim=1)
        out = self.conv_2_1(out_c)
        out = self.conv_2_2(out)

        out = F.interpolate(out, scale_factor=2)
        out_l = self.block3_l(out, c, words_embs, mask)
        out_g = self.block3_g(out, c, words_embs, mask)
        out_c = torch.cat((out_l, out_g,), dim=1)
        out = self.conv_3_1(out_c)
        out = self.conv_3_2(out)

        out = F.interpolate(out, scale_factor=2)
        out = self.block4(out, c, words_embs, mask)

        out = F.interpolate(out, scale_factor=2)
        out = self.block5(out, c, words_embs, mask)

        out = F.interpolate(out, scale_factor=2)
        out = self.block6(out, c, words_embs, mask)

        out = self.conv_img(out)

        return out

    def observe(self, t, proj=True):
        index = torch.LongTensor([1 - t])
        store_grad(self.parameters, self.grads, self.grad_dims, t)

        dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                        self.grads.index_select(1, index))
        if (dotp < 0).sum() != 0 and proj:
            print('different direction, apply GProj')
            try:
                project2cone2(self.grads[:, t].unsqueeze(1),
                              self.grads.index_select(1, index), self.margin)
                overwrite_grad(self.parameters, self.grads[:, t],
                               self.grad_dims)
            except Exception:
                overwrite_grad(self.parameters, self.grads[:, t] * 0,
                               self.grad_dims)
                print(f'GProj failed, set grads for task {t} to 0')


class G_Block(nn.Module):
    def __init__(self, in_ch, out_ch, use_sent_affine=True, use_word_affine=True):
        super(G_Block, self).__init__()
        self.in_ch, self.out_ch = in_ch, out_ch,
        self.learnable_sc = in_ch != out_ch
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.use_sent_affine = use_sent_affine
        self.use_word_affine = use_word_affine
        if self.use_sent_affine:
            self.affine_s0 = affine(in_ch)
            self.affine_s1 = affine(in_ch)
            self.affine_s2 = affine(out_ch)
            self.affine_s3 = affine(out_ch)
        if self.use_word_affine:
            self.affine_w0 = affine_word(in_ch)
            self.affine_w1 = affine_word(in_ch)
            self.affine_w2 = affine_word(out_ch)
            self.affine_w3 = affine_word(out_ch)

        self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def forward(self, x, sent_cond, word_cond, mask):
        return self.shortcut(x) + self.gamma * self.residual(x, sent_cond, word_cond, mask)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, sent_cond, word_cond, mask):
        h = x
        if self.use_sent_affine:
            h = self.affine_s0(h, sent_cond)
        if self.use_word_affine:
            h = self.affine_w0(h, word_cond, mask)
        h = nn.LeakyReLU(0.2, inplace=True)(h)

        if self.use_sent_affine:
            h = self.affine_s1(h, sent_cond)
        if self.use_word_affine:
            h = self.affine_w1(h, word_cond, mask)
        h = nn.LeakyReLU(0.2, inplace=True)(h)

        h = self.c1(h)
        if self.use_sent_affine:
            h = self.affine_s2(h, sent_cond)
        if self.use_word_affine:
            h = self.affine_w2(h, word_cond, mask)
        h = nn.LeakyReLU(0.2, inplace=True)(h)

        if self.use_sent_affine:
            h = self.affine_s3(h, sent_cond)
        if self.use_word_affine:
            h = self.affine_w3(h, word_cond, mask)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        return self.c2(h)


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, ncond=512):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.cond_dim = ncond
        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 16 + ncond, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )

    def forward(self, out, y):
        y = y.view(-1, self.cond_dim, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((out, y), 1)
        out = self.joint_conv(h_c_code)
        return out


# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf, ncond):
        super(NetD, self).__init__()

        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)  # 128
        self.block0 = resD(ndf * 1, ndf * 2)  # 64

        self.block1 = resD(ndf * 2, ndf * 4)  # 32
        self.block2 = resD(ndf * 4, ndf * 8)  # 16
        self.block3 = resD(ndf * 8, ndf * 16)  # 8
        self.block4 = resD(ndf * 16, ndf * 16)  # 4
        self.block5 = resD(ndf * 16, ndf * 16)  # 4
        self.COND_DNET = D_GET_LOGITS(ndf, ncond)

    def forward(self, x):
        out = self.conv_img(x)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        return out


class resD(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x) + self.gamma * self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)