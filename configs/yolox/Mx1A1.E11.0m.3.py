_base_ = './DiNat17_yolox_m_8x8_300e_coco.py'


data = dict(
    samples_per_gpu=12,  # Batch size of a single GPU
    workers_per_gpu=4  # Worker to pre-fetch data for each single GPU
    )

max_epochs = 60 #300 # NOTE: These don't appear to work. Adjust this in yolox_s...
num_last_epochs = 3 #15
#resume_from = '/'
load_from = '/pretrained/yolox_m.pth'
interval = 2 #10

optimizer = dict(
    type='SGD',
    lr=0.01/2.66, # 64/24 = 2.66 -> 8gpus * 8imgs/gpu = 64imgs / 2gpus * 12imgs/gpu = 24imgs
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

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

checkpoint_config = dict(interval=interval)
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='bbox')
log_config = dict(interval=50)