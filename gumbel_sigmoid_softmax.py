# -*- coding: utf-8 -*- 
"""
a bunch of lasagne code implementing gumbel softmax
https://arxiv.org/abs/1611.01144
 
"""
import torch

# modified from https://github.com/yandexdataschool/gumbel_lstm/blob/master/gumbel_sigmoid.py
def gumbel_sigmoid(
    logits,
    temperature=1.0,
    hard=False,
    eps=1e-20
):
    """
    A gumbel-sigmoid nonlinearity with gumbel(0,1) noize
    In short, it's a function that mimics #[a>0] indicator where a is the logit

    Explaination and motivation: https://arxiv.org/abs/1611.01144

    Math:
    Sigmoid is a softmax of two logits: a and 0
    e^a / (e^a + e^0) = 1 / (1 + e^(0 - a)) = sigm(a)

    Gumbel-sigmoid is a gumbel-softmax for same logits:
    gumbel_sigm(a) = e^([a+gumbel1]/t) / [ e^([a+gumbel1]/t) + e^(gumbel2/t)]
    where t is temperature, gumbel1 and gumbel2 are two samples from gumbel noize: -log(-log(uniform(0,1)))
    gumbel_sigm(a) = 1 / ( 1 +  e^(gumbel2/t - [a+gumbel1]/t) = 1 / ( 1+ e^(-[a + gumbel1 - gumbel2]/t)
    gumbel_sigm(a) = sigm([a+gumbel1-gumbel2]/t)
    
    For computation reasons:
    gumbel1-gumbel2 = -log(-log(uniform1(0,1)) + log(-log(uniform2(0,1)) = log( log(uniform2(0,1)) / log(uniform1(0,1)) )
    gumbel_sigm(a) = sigm([ a - log(log(uniform2(0,1)) / log(uniform1(0,1))) ] / t)

    Args:
        logits: [batch_size, ] unnormalized log-probs
        temperature: temperature of sampling. Lower means more spike-like sampling. Can be symbolic.
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        eps: a small number used for numerical stability

    Returns:
        a callable that can (and should) be used as a nonlinearity
    
    """

    assert temperature != 0

    # computes a gumbel softmax sample

    #sample from Gumbel(0, 1)
    uniform1 = torch.rand(logits.shape, device=logits.device)
    uniform2 = torch.rand(logits.shape, device=logits.device)

    noise = torch.log(
        torch.log(uniform2 + eps) / torch.log(uniform1 + eps) + eps
    )

    #draw a sample from the Gumbel-Sigmoid distribution
    y = torch.sigmoid((logits + noise) / temperature)

    if not hard:
        return y

    y_hard = torch.zeros_like(y)  # [Batchsize,]
    y_hard[y>0] = 1
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard



# modified from https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f#file-st-gumbel-py
def _sample_gumbel(shape, eps=1e-20, device=None):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def _gumbel_softmax_sample(logits, temperature=1.0):
    y = logits + _sample_gumbel(logits.size(), device=logits.device)
    return torch.softmax(y / temperature, dim=-1)

# ST-gumple-softmax
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y

    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
    y = _gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

