_base_ = './Mx1A1.DiNat17-super.tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py'

work_dir = 'det_ckpts/Mx1A1/E9.0/'

dataset_type = 'CocoDataset'
data_root = 'data/coco/'
classes = ('Actinopterygii', 'Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Insecta', 'Mammalia', 'Mollusca', 'Reptilia')

data = dict(
    samples_per_gpu=1,  # Batch size of a single GPU
    workers_per_gpu=4,  # Worker to pre-fetch data for each single GPU
    train = dict(
                img_prefix = '/local-data/Polli/Datasets/inaturalist2017/exports/train/data/',
                classes = classes,
                ann_file = '/local-data/Polli/Datasets/inaturalist2017/exports/train/labels.json'),
    val = dict(
        img_prefix = '/local-data/Polli/Datasets/inaturalist2017/exports/val-15k/data/',
        classes = classes,
        ann_file = '/local-data/Polli/Datasets/inaturalist2017/exports/val-15k/labels.json'),
    test = dict(
        img_prefix = '/local-data/Polli/Datasets/inaturalist2017/exports/val/data/',
        classes = classes,
        ann_file = '/local-data/Polli/Datasets/inaturalist2017/exports/val/labels.json')
    )


evaluation = dict(interval=1, metric='bbox')
