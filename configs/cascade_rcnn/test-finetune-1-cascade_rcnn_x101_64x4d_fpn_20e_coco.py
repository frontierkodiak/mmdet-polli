_base_ = './cascade_rcnn_r50_fpn_20e_coco.py'

data_root = None
dataset_type = 'CocoDataset'
classes = ('Insecta','Chromista','Animalia','Plantae','Aves','Mollusca','Protozoa','Mammalia','Amphibia','Fungi','Reptilia','Actinopterygii', 'Arachnida')
data = dict(
    samples_per_gpu=2,  # Batch size of a single GPU
    workers_per_gpu=8,  # Worker to pre-fetch data for each single GPU
    train = dict(
                img_prefix = None,
                classes = classes,
                ann_file = '/local-data/Polli/Datasets/inaturalist2017/train_val/train_2017_bboxes_super.json'),
    val = dict(
        img_prefix = None,
        classes = classes,
        ann_file = '/local-data/Polli/Datasets/inaturalist2017/train_val/val_2017_bboxes_super.json'),
    test = dict(
        img_prefix = None,
        classes = classes,
        ann_file = '/local-data/Polli/Datasets/inaturalist2017/train_val/val_2017_bboxes_super.json')
    )

model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'#,
        # init_cfg=dict(
        #     type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')
        ),
        roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(type='Shared2FCBBoxHead', num_classes=13),
            dict(type='Shared2FCBBoxHead', num_classes=13),
            dict(type='Shared2FCBBoxHead', num_classes=13)
        ]))

work_dir = '/models/work_dirs/DiNat17-super/test-finetune-1-cascade_rcnn_x101_64x4d_fpn_20e_coco/'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth'  # noqa

runner = dict(
    type='EpochBasedRunner', # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_epochs=8) # Runner that runs the workflow in total max_epochs. For IterBasedRunner use `max_iters`
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 12 epochs according to the total_epochs.

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7])

# lr is set for a batch size of 2 x 1GPU
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)