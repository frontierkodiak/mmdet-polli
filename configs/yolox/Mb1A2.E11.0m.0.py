_base_ = './naBirds_yolox_m_8x8_300e_coco.py'


data = dict(
    samples_per_gpu=12,  # Batch size of a single GPU
    workers_per_gpu=4  # Worker to pre-fetch data for each single GPU
    )

max_epochs = 300 #300 # NOTE: These don't appear to work. Adjust this in yolox_s...
num_last_epochs = 15 #15
#resume_from = '/'
load_from = '/pretrained/yolox_m.pth'
interval = 1 #10

