
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RecallLoss(nn.Module):
    """ 
        Recall weighted loss by Zhen Qi <qizhen816@163.com>
        Modified from:
            An unofficial implementation of
            <Recall Loss for Imbalanced Image Classification and Semantic Segmentation>
            Created by: Zhang Shuai
            Email: shuaizzz666@gmail.com
            recall = TP / (TP + FN)
    Args:
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """
    def __init__(self,reduction='mean', ignore_index=255,):
        super(RecallLoss, self).__init__()
        self.smooth = 1e-5
        self.nll_loss = nn.NLLLoss(weight=None, ignore_index=ignore_index, reduction=reduction)
        self.reduction = reduction

    def forward(self, input, target):
        N, C = input.size()[:2]
        _, predict = torch.max(input, 1)  # # (N, C, *) ==> (N, 1, *)

        predict = predict.view(N, 1, -1)  # (N, 1, *)
        target_ = target.view(N, 1, -1)  # (N, 1, *)
        last_size = target_.size(-1)

        ## convert predict & target (N, 1, *) into one hot vector (N, C, *)
        predict_onehot = torch.zeros((N, C, last_size)).cuda()  # (N, 1, *) ==> (N, C, *)
        predict_onehot.scatter_(1, predict, 1)  # (N, C, *)
        target_onehot = torch.zeros((N, C, last_size)).cuda()  # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target_, 1)  # (N, C, *)

        true_positive = torch.sum(predict_onehot * target_onehot, dim=2)  # (N, C)
        total_target = torch.sum(target_onehot, dim=2)  # (N, C)
        ## Recall = TP / (TP + FN)
        recall = (true_positive + self.smooth) / (total_target + self.smooth)  # (N, C)

        recall = (1 - recall)
        loss = []
        for i in range(0, input.shape[0]):
            self.nll_loss.weight = recall[i]
            loss.append(self.nll_loss(F.log_softmax(input[i].unsqueeze(0), dim=1), target[i].unsqueeze(0)))
        if self.reduction == 'none':
            loss = torch.cat(loss, dim=0)
        else:
            loss = torch.stack(loss, dim=0)
        return loss


if __name__ == '__main__':
    y_target = torch.Tensor([[0, 1], [1, 0]]).long().cuda(0)
    y_predict = torch.Tensor([[[1.5, 1.0], [2.8, 1.6]],
                           [[1.0, 1.0], [2.4, 0.3]]]
                          ).cuda(0)

    criterion = RecallLoss()
    loss = criterion(y_predict, y_target)
    print(loss)
