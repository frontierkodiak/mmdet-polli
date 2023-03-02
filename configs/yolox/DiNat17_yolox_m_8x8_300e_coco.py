_base_ = './yolox_s_8x8_300e_coco.py'

# model settings
model = dict(
    backbone=dict(deepen_factor=0.67, widen_factor=0.75),
    neck=dict(in_channels=[192, 384, 768], out_channels=192, num_csp_blocks=2),
    bbox_head=dict(num_classes=9, in_channels=192, feat_channels=192),
)

data_root = None
dataset_type = 'CocoDataset'
classes = ('Actinopterygii', 'Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Insecta', 'Mammalia', 'Mollusca', 'Reptilia')

data = dict(
    samples_per_gpu=1,  # Batch size of a single GPU
    workers_per_gpu=6,  # Worker to pre-fetch data for each single GPU
    train = dict(
                img_prefix = '/local-data/Polli/Datasets/inaturalist2017/exports/train/data/',
                classes = classes,
                ann_file = '/local-data/Polli/Datasets/inaturalist2017/exports/train/labels.json'),
    val = dict(
        img_prefix = '/local-data/Polli/Datasets/inaturalist2017/exports/val/data/',
        classes = classes,
        ann_file = '/local-data/Polli/Datasets/inaturalist2017/exports/val/labels.json'),
    test = dict(
        img_prefix = '/local-data/Polli/Datasets/inaturalist2017/exports/val/data/',
        classes = classes,
        ann_file = '/local-data/Polli/Datasets/inaturalist2017/exports/val/labels.json')
    )