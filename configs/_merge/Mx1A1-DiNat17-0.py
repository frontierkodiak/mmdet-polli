### Merge this with any mmdet config to easily use that configuration with my allBees detection dataset.
### Specify the merge using param hyperparameters:merge_config: within the determined experiment config yaml.

data_root = None
dataset_type = 'CocoDataset'
classes = ('Insecta','Chromista','Animalia','Plantae','Aves','Mollusca','Protozoa','Mammalia','Amphibia','Fungi','Reptilia','Actinopterygii', 'Arachnida')

train_pipeline = [
    dict(type='Resize', img_scale=(800, 492), keep_ratio=True),
]
test_pipeline = [
    dict(
        img_scale=(800, 492),)
]

data = dict(
    samples_per_gpu=6,  # Batch size of a single GPU
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
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=13),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=13),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=13)
        ])
)