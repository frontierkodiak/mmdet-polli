_base_ = './cascade_rcnn_r50_fpn_20e_coco.py'

dataset_type = 'CocoDataset'
classes = ('Insecta','Chromista','Animalia','Plantae','Aves','Mollusca','Protozoa','Mammalia','Amphibia','Fungi','Reptilia','Actinopterygii', 'Arachnida')
data = dict(
    train = dict(
            dataset=dict(
                img_prefix = '',
                classes = classes,
                ann_file = '/local-data/Polli/Datasets/inaturalist2017/train_val/train_2017_bboxes_super.json')
    ),
    val = dict(
        img_prefix = '',
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
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')),
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

load_from = '/models/pretrained/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco_20200512_161033-bdb5126a.pth'