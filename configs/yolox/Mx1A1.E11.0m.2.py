_base_ = './DiNat17_yolox_m_8x8_300e_coco.py'


data = dict(
    samples_per_gpu=12,  # Batch size of a single GPU
    workers_per_gpu=4  # Worker to pre-fetch data for each single GPU
    )

max_epochs = 60 #300
num_last_epochs = 5 #15
resume_from = None
load_from = '/pretrained/yolox_m.pth'
interval = 1 #10

lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=True,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=1,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)