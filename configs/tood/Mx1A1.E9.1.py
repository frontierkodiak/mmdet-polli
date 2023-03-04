_base_ = './Mx1A1.DiNat17-super.tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py'

work_dir = 'det_ckpts/Mx1A1/E9.1/'

dataset_type = 'CocoDataset'
data_root = 'data/coco/'
classes = ('Actinopterygii', 'Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Insecta', 'Mammalia', 'Mollusca', 'Reptilia')

load_from ='/root/.cache/torch/hub/checkpoints/resnet101-63fe2227.pth' # this doesn't work within determined

data = dict(
    samples_per_gpu=1,  # Batch size of a single GPU
    workers_per_gpu=4,  # Worker to pre-fetch data for each single GPU
    train = dict(
                img_prefix = '/local-data/Polli/Datasets/inaturalist2017/exports/train/data/',
                classes = classes,
                ann_file = '/local-data/Polli/Datasets/inaturalist2017/exports/train/labels.json'),
    val = dict(
        img_prefix = '/local-data/Polli/Datasets/inaturalist2017/exports/val-mini/data/',
        classes = classes,
        ann_file = '/local-data/Polli/Datasets/inaturalist2017/exports/val-mini/labels.json'),
    test = dict(
        img_prefix = '/local-data/Polli/Datasets/inaturalist2017/exports/val/data/',
        classes = classes,
        ann_file = '/local-data/Polli/Datasets/inaturalist2017/exports/val/labels.json')
    )

fp16 = dict(loss_scale=512.)

evaluation = dict(interval=1, metric='bbox')
