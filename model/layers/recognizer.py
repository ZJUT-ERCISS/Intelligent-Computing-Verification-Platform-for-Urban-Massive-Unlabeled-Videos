from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from model import builder
from model.builder import RECOGNIZERS


class BaseRecognizer(nn.Module, metaclass=ABCMeta):
    """Base class for recognizers.

    All recognizers should subclass it.
    All subclass should overwrite:

    - Methods:``forward_train``, supporting to forward when training.
    - Methods:``forward_test``, supporting to forward when testing.

    Args:
        backbone (dict): Backbone modules to extract feature.
        cls_head (dict | None): Classification head to process feature. Default: None.
        train_cfg (dict): Config for training. Default: {}.
        test_cfg (dict): Config for testing. Default: {}.
    """

    def __init__(self,
                 backbone,
                 cls_head=None,
                 train_cfg=dict(),
                 test_cfg=dict()):
        super().__init__()
        # record the source of the backbone
        self.backbone = builder.build_backbone(backbone)
        self.cls_head = builder.build_head(cls_head) if cls_head else None

        if train_cfg is None:
            train_cfg = dict()
        if test_cfg is None:
            test_cfg = dict()

        assert isinstance(train_cfg, dict)
        assert isinstance(test_cfg, dict)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # max_testing_views should be int
        self.max_testing_views = test_cfg.get('max_testing_views', None)
        self.init_weights()

    @property
    def with_cls_head(self):
        """bool: whether the recognizer has a cls_head"""
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def init_weights(self):
        """Initialize the model network weights."""
        self.backbone.init_weights()
        if self.with_cls_head:
            self.cls_head.init_weights()

    def extract_feat(self, imgs):
        """Extract features through a backbone.

        Args:
            imgs (torch.Tensor): The input images.

        Returns:
            torch.tensor: The extracted features.
        """
        x = self.backbone(imgs)
        return x

    def average_clip(self, cls_score):
        """Averaging class score over multiple clips.
        
        Using different averaging types ('score' or 'prob' or None, which defined in test_cfg) to computed the final
        averaged class score. Only called in test mode. By default, we use 'prob' mode.

        Args:
            cls_score (torch.Tensor): Class score to be averaged.

        Returns:
            torch.Tensor: Averaged class score.
        """
        assert len(cls_score.shape) == 3  # * (Batch, NumSegs, Dim)
        average_clips = self.test_cfg.get('average_clips', 'prob')
        if average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{average_clips} is not supported. Supported: ["score", "prob", None]')

        if average_clips is None:
            return cls_score

        if average_clips == 'prob':
            return F.softmax(cls_score, dim=2).mean(dim=1)
        elif average_clips == 'score':
            return cls_score.mean(dim=1)

    @abstractmethod
    def forward_train(self, imgs, label, **kwargs):
        """Defines the computation performed at every call when training."""

    @abstractmethod
    def forward_test(self, imgs, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""

    def _parse_losses(self, losses):
        """Parse the ra w outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward(self, imgs=None, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(imgs, label, **kwargs)

        return self.forward_test(imgs, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs
    

    def val_step(self, data_batch, optimizer = None, **kwargs):

        losses = self(**data_batch, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs


@RECOGNIZERS.register_module()
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def forward_train(self, imgs, label, **kwargs):
        """Defines the computation performed at every call when training."""

        # assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()

        x = self.extract_feat(imgs)

        cls_score = self.cls_head(x)
        gt_label = label.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_label, **kwargs)
        losses.update(loss_cls)

        return losses

    def forward_test(self, imgs, **kwargs):
        """Defines the computation performed at every call when evaluation, testing."""
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            feats = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                feats.append(x)
                view_ptr += self.max_testing_views
            # should consider the case that feat is a tuple
            if isinstance(feats[0], tuple):
                len_tuple = len(feats[0])
                feat = [
                    torch.cat([x[i] for x in feats]) for i in range(len_tuple)
                ]
                feat = tuple(feat)
            else:
                feat = torch.cat(feats)
        else:
            feat = self.extract_feat(imgs)

        if self.test_cfg.get('feat_ext', False):
            feat_dim = len(feat[0].size()) if isinstance(feat, tuple) else len(feat.size())
            assert feat_dim in [5, 2], (
                'Got feature of unknown architecture, '
                'only 3D-CNN-like ([N, in_channels, T, H, W]), and '
                'transformer-like ([N, in_channels]) features are supported.')
            if feat_dim == 5:  # 3D-CNN architecture
                # perform spatio-temporal pooling
                avg_pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(feat, tuple):
                    feat = [avg_pool(x) for x in feat]
                    # concat them
                    feat = torch.cat(feat, axis=1)
                else:
                    feat = avg_pool(feat)
                # squeeze dimensions
                feat = feat.reshape((batches, num_segs, -1))
                # temporal average pooling
                feat = feat.mean(axis=1)
            return feat.cpu().numpy()

        # should have cls_head if not extracting features
        assert self.with_cls_head
        cls_score = self.cls_head(feat)
        cls_score = cls_score.reshape(batches, num_segs, cls_score.shape[-1])
        cls_score = self.average_clip(cls_score)
        return cls_score.cpu().numpy()
