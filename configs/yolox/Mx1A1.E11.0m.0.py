_base_ = './DiNat17_yolox_m_8x8_300e_coco.py'


data = dict(
    samples_per_gpu=1,  # Batch size of a single GPU
    workers_per_gpu=6  # Worker to pre-fetch data for each single GPU
    )