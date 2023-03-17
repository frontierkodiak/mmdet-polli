_base_ = './naBirds_yolox_tiny_8x8_300e_coco.py'


data = dict(
    samples_per_gpu=44,  # Batch size of a single GPU
    workers_per_gpu=8  # Worker to pre-fetch data for each single GPU
    )

auto_scale_lr = dict(base_batch_size=88) # NOTE: Adding this, as I suspect there is version mismatch b/w mmdet-polli configs and mmdet 2.21.0 (the version used in determined image).


load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
