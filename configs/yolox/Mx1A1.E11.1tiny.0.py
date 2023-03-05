_base_ = './DiNat17_yolox_tiny_8x8_300e_coco.py'



### yolox_s_8x8_300e_coco.py

# optimizer = dict(
#     type='SGD',
#     lr=0.01,
#     momentum=0.9,
#     weight_decay=5e-4,
#     nesterov=True,
#     paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
# optimizer_config = dict(grad_clip=None)

# max_epochs = 60 #300
# num_last_epochs = 3 #15
# resume_from = None
# interval = 2

# # learning policy
# lr_config = dict(
#     _delete_=True,
#     policy='YOLOX',
#     warmup='exp',
#     by_epoch=False,
#     warmup_by_epoch=True,
#     warmup_ratio=1,
#     warmup_iters=5,  # 5 epoch
#     num_last_epochs=num_last_epochs,
#     min_lr_ratio=0.05)