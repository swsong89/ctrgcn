import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, device, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        print('focal loss cuda: ', device)
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1)).cuda(device)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha).cuda(device)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, -1)

        class_mask = inputs.data.new(N, C).fill_(0)  # [batch, class_num]比如120类，先让120为0
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)#  拉伸 [batch, class_num]
        class_mask.scatter_(1, ids.data, 1.)  # 让[batch, class_num]相应位置为1,其余还是0
        #print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            # print('loss forward: ', inputs.get_device())
            self.alpha = self.alpha.cuda(inputs.get_device())
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1) # 即，只计算了真实标签对应的预测概率的值，别的都是0

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


# HDGCN
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()



# a class-balanced focal loss DG-STGCN
# class BCELossWithLogits(BaseWeightedLoss):
#     """Binary Cross Entropy Loss with logits.
#     Args:
#         loss_weight (float): Factor scalar multiplied on the loss.
#             Default: 1.0.
#         class_weight (list[float] | None): Loss weight for each class. If set
#             as None, use the same weight 1 for all classes. Only applies
#             to CrossEntropyLoss and BCELossWithLogits (should not be set when
#             using other losses). Default: None.
#     """

#     def __init__(self, loss_weight=1.0, class_weight=None):
#         super().__init__(loss_weight=loss_weight)
#         self.class_weight = None
#         if class_weight is not None:
#             self.class_weight = torch.Tensor(class_weight)

#     def _forward(self, cls_score, label, **kwargs):
#         """Forward function.
#         Args:
#             cls_score (torch.Tensor): The class score.
#             label (torch.Tensor): The ground truth label.
#             kwargs: Any keyword argument to be used to calculate
#                 bce loss with logits.
#         Returns:
#             torch.Tensor: The returned bce loss with logits.
#         """
#         if self.class_weight is not None:
#             assert 'weight' not in kwargs, "The key 'weight' already exists."
#             kwargs['weight'] = self.class_weight.to(cls_score.device)
#         loss_cls = F.binary_cross_entropy_with_logits(cls_score, label,
#                                                       **kwargs)
#         return loss_cls