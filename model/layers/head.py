import torch
import torch.nn as nn

from model.builder import HEADS, LOSSES
from model.builder import build_loss
from utils.evaluation import top_k_accuracy

from pytorch_metric_learning.losses import TripletMarginLoss as triLoss
from pytorch_metric_learning.miners import MultiSimilarityMiner


@LOSSES.register_module
class TripletMarginLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.loss = nn.TripletMarginLoss
        self.loss = triLoss(margin=0.2)
        

    def forward(self, embeddings, labels=None, indices_tuple=None, ref_emb=None, ref_labels=None):
        return self.loss(embeddings, labels, indices_tuple, ref_emb, ref_labels)


@HEADS.register_module()
class metric_head(nn.Module):
    # 训练度量头
    def __init__(self,
                 num_classes,
                 dim=1024,
                 loss_cls=dict(type='CrossEntropyLoss'),
                #  metric_loss=dict(type='TripletMarginLoss'),
                 multi_class=False,
                 label_smooth_eps=0.0
                 ):
        super().__init__()
        
        self.num_classes = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps
        # self.metric_loss = build_loss(metric_loss)
        # self.softmax = nn.Softmax()
        self.miner = MultiSimilarityMiner(epsilon=0.1)
        self.metric_loss = triLoss(margin=0.2)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        # 1024 -> num_classes

    
    def forward(self, x):
        
        x = self.mlp_head(x)

        return x

    def init_weights(self):
        pass

    def loss(self, embedding, cls_score, label, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``label``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            label (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        losses = dict()
        if label.shape == torch.Size([]):
            label = label.unsqueeze(0)
        elif label.dim() == 1 and label.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            label = label.unsqueeze(0)

        if not self.multi_class and cls_score.size() != label.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       label.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(
                top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(
                top_k_acc[1], device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            label = ((1 - self.label_smooth_eps) * label + self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_score, label, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        indices_tuple = self.miner(embedding, label)
        losses['metric_loss'] = self.metric_loss(embedding, label, indices_tuple)

        return losses


@HEADS.register_module()
class infer_head(nn.Module):
    def __init__(self,
                 num_classes,
                 dim=1024,
                 loss_cls=dict(type='CrossEntropyLoss'),
                #  metric_loss=dict(type='TripletMarginLoss'),
                 multi_class=False,
                 label_smooth_eps=0.0
                 ):
        super().__init__()
        
        self.num_classes = num_classes
        self.loss_cls = build_loss(loss_cls)

    
    def forward(self, x):
        
        return x

    def init_weights(self):
        pass

    def loss(self, embedding, cls_score, label, **kwargs):
        pass

@HEADS.register_module()
class vivit_head(nn.Module):
    def __init__(self,
                 num_classes,
                 dim=192,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 multi_class=False,
                 label_smooth_eps=0.0
                 ):
        super().__init__()
        
        self.num_classes = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )
        
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.softmax(x)
        return x

    def init_weights(self):
        pass

    def loss(self, cls_score, label, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``label``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            label (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        losses = dict()
        if label.shape == torch.Size([]):
            label = label.unsqueeze(0)
        elif label.dim() == 1 and label.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            label = label.unsqueeze(0)

        if not self.multi_class and cls_score.size() != label.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       label.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(
                top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(
                top_k_acc[1], device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            label = ((1 - self.label_smooth_eps) * label + self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_score, label, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        # print('losses =', float(losses['loss_cls']), '; top1_acc =', float(losses['top1_acc']),
        #  '; top5_acc =', float(losses['top5_acc']))
        return losses

@HEADS.register_module()
class test_head_old(nn.Module):
    def __init__(self,
                 num_classes,
                 dim=1024,
                 loss_cls=dict(type='CrossEntropyLoss'),
                #  metric_loss=dict(type='TripletMarginLoss'),
                 multi_class=False,
                 label_smooth_eps=0.0
                 ):
        super().__init__()
        
        self.num_classes = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps
        # self.metric_loss = build_loss(metric_loss)
        # self.softmax = nn.Softmax()
        self.miner = MultiSimilarityMiner(epsilon=0.1)
        self.metric_loss = triLoss(margin=0.2)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    
    def forward(self, x):
        
        # x = self.mlp_head(x)

        return x

    def init_weights(self):
        pass

    def loss(self, embedding, cls_score, label, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``label``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            label (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        losses = dict()
        if label.shape == torch.Size([]):
            label = label.unsqueeze(0)
        elif label.dim() == 1 and label.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            label = label.unsqueeze(0)

        if not self.multi_class and cls_score.size() != label.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       label.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(
                top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(
                top_k_acc[1], device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            label = ((1 - self.label_smooth_eps) * label + self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_score, label, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        indices_tuple = self.miner(embedding, label)
        losses['metric_loss'] = self.metric_loss(embedding, label, indices_tuple)

        return losses