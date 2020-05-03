import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math


class NCEAverageMultiview(nn.Module):

    def __init__(self, inputSize, outputSize, n_views, K, T=0.07, momentum=0.5, use_softmax=False):
        ## inputSize is set via commandline argument feat_dim (default int: 128)
        ## outputSize is set by the number of samples in the training set n_data (int)
        ## K is set via commandline argument nce_k (default int: 16384)
        ## T is set via commandline argument nce_t (default float: 0.07)
        ## momentum is set via commandline argument nce_m (default float: 0.5)

        super(NCEAverageMultiview, self).__init__()
        self.n_views = n_views
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, v1, v2, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out_v2 = torch.div(out_v2, T)
            out_v1 = torch.div(out_v1, T)
            out_v1 = out_v1.contiguous()
            out_v2 = out_v2.contiguous()
        else:
            out_v2 = torch.exp(torch.div(out_v2, T))
            out_v1 = torch.exp(torch.div(out_v1, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_v1 < 0:
                self.params[2] = out_v1.mean() * outputSize
                Z_v1 = self.params[2].clone().detach().item()
                print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
            if Z_v2 < 0:
                self.params[3] = out_v2.mean() * outputSize
                Z_v2 = self.params[3].clone().detach().item()
                print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))
            # compute out_v1, out_v2
            out_v1 = torch.div(out_v1, Z_v1).contiguous()
            out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # # update memory
        with torch.no_grad():
            v1_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            v1_pos.mul_(momentum)
            v1_pos.add_(torch.mul(v1, 1 - momentum))
            v1_norm = v1_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = v1_pos.div(v1_norm)
            self.memory_l.index_copy_(0, y, updated_v1)

            v2_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            v2_pos.mul_(momentum)
            v2_pos.add_(torch.mul(v2, 1 - momentum))
            v2_norm = v2_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = v2_pos.div(v2_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2


# =========================
# InsDis and MoCo
# =========================

class MemoryInsDis(nn.Module):
    """Memory bank with instance discrimination"""
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(MemoryInsDis, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z = self.params[2].item()
        momentum = self.params[3].item()

        batchSize = x.size(0)
        outputSize = self.memory.size(0)
        inputSize = self.memory.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        # sample
        weight = torch.index_select(self.memory, 0, idx.view(-1))
        weight = weight.view(batchSize, K + 1, inputSize)
        out = torch.bmm(weight, x.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out = torch.div(out, T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, T))
            if Z < 0:
                self.params[2] = out.mean() * outputSize
                Z = self.params[2].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            weight_pos = torch.index_select(self.memory, 0, y.view(-1))
            weight_pos.mul_(momentum)
            weight_pos.add_(torch.mul(x, 1 - momentum))
            weight_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(weight_norm)
            self.memory.index_copy_(0, y, updated_weight)

        return out


class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, inputSize))

    def forward(self, q, k):
        batchSize = q.shape[0]
        k = k.detach()

        Z = self.params[0].item()

        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        # neg logit
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)

        out = torch.cat((l_pos, l_neg), dim=1)

        if self.use_softmax:
            out = torch.div(out, self.T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, self.T))
            if Z < 0:
                self.params[0] = out.mean() * self.outputSize
                Z = self.params[0].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize

        return out
