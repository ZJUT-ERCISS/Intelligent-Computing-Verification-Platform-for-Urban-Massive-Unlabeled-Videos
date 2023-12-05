model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='AAGCN_vivit',
        graph_cfg=dict(layout='coco', mode='spatial'),
        num_classes=144,
        dim=1024,
        spatial_depth=6,
        temporal_depth=6,
        heads=8,
        mlp_dim=2048,
    ),
    cls_head=dict(
        type='test_head_old',
        # type='test_head',
        num_classes=120,
    )
)

dataset_type = 'PoseDataset'
ann_file = '/home/code/video_model/new_results.pkl' # train部分保存的为sample数据样本,test部分保存测试数据


train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='UniformSample', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='UniformSample', clip_len=100, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xsub_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xsub_val'),)
    # sample=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xsub_sample'))
    # test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xsub_train'))

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005, nesterov=True)#weight_decay=0.0005
optimizer_config = dict(grad_clip=None)
# learning policy
steps_per_epoch = 380800 // 32
warmup_steps = int(2.5 * steps_per_epoch)
lr_config = dict(
    policy = 'CosineAnnealing',
    warmup = 'linear',
    warmup_iters = warmup_steps,
    warmup_ratio = 0.01,
    min_lr_ratio = 1e-5,
)
# lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 75
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=300, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = '/home/shr/code/gcn_vivit/.vscode/train/work'
