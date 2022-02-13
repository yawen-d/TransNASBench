import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss


class MSELoss(nn.MSELoss):
    def __init__(self):
        super(MSELoss, self).__init__()


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()


class SoftmaxCrossEntropyWithLogits(_WeightedLoss):
    """Implementation of tf.nn.softmax_cross_entropy_with_logits in PyTorch
    Able to handle probabilistic target like [[0.7, 0.2, 0.1]]
    """
    def __init__(self, weight=None):
        super(SoftmaxCrossEntropyWithLogits, self).__init__(weight=None)
        self.weight = weight

    def forward(self, input, target):
        logits_scaled = torch.log(F.softmax(input, dim=-1) + 0.00001)
        if self.weight is not None:
            loss = - ((target * logits_scaled) * self.weight).sum(dim=-1)
        else:
            loss = - (target * logits_scaled).sum(dim=-1)
        return loss.mean()


class GANLoss(nn.Module):
    """
    Define different GAN objectives. The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    code from https://github.com/phillipi/pix2pix
    """

    def __init__(self, gan_mode):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan.
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def __call__(self, prediction, target):
        """Calculate loss given Discriminator's output and ground truth labels.
        Parameters:
            prediction (tensor) - - typically the prediction output from a discriminator
            target (tensor) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        assert self.gan_mode in ['lsgan', 'vanilla']
        return self.loss(prediction, target)
