_base_ = './cascade_rcnn_r50_fpn_20e_coco.py'

dataset_type = 'CocoDataset'
classes = '/local-data/Polli/Datasets/inaturalist2017/train_val/super_classes.txt'
data = dict(
    train = dict(
        img_prefix = '',
        classes = classes,
        ann_file = '/local-data/Polli/Datasets/inaturalist2017/train_val/train_2017_bboxes_super.json'),
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
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))

# TODO: update config to reflect correct no. of classes