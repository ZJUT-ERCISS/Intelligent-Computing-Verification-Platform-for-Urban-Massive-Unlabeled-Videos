
ann_file = '/home/shihr/code/ntu_kin.pkl'
# ann_file = '/home/shy/code_hik/video_model/falldown_splited_w_20ntu_test_aagcn_vivit.pkl' # lr=0.01
# ann_file = '/home/shy/code_hik/video_model/falldown.list'
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='AAGCN_vivit',
        graph_cfg=dict(layout='coco', mode='spatial'),
        num_classes=120,
        dim=1024,
        spatial_depth=6,
        temporal_depth=6,
        heads=8,
        mlp_dim=2048,
    ),
    cls_head=dict(
        type='metric_head',
        num_classes=144,
        # num_classes = 120
    )
)

dataset_type = 'PoseDataset'

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
        dataset=dict(type=dataset_type, ann_file=ann_file, split='xsub_train', pipeline=train_pipeline)),
    val=dict(type=dataset_type, ann_file=ann_file, split='xsub_val', pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split='xsub_val', pipeline=test_pipeline)
)
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
total_epochs = 75
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=300, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = '/home/shy/code_hik/video_model/.vscode/train/work_dir'

mode = ''
