_base_ = './DiNat17_yolox_m_8x8_300e_coco.py'



data = dict(
    samples_per_gpu=22,
    workers_per_gpu=8)



auto_scale_lr = dict(base_batch_size=44)


load_from = '/lake/mmdet-polli/pretrained/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'