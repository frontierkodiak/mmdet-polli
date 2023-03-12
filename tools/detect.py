from PIL import Image
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from pathlib import Path
import numpy as np
from tqdm import tqdm
from detectTools import imshow_det_bboxes

input_dir = '/local-data/Polli/Datasets/backyard/birdfeeders/images' # ~3.5it/s from lake. maybe ~4 from local-data. high variance on local-data.
recursive = True
ckpt_path = '/mmdetection/work_dirs/Mx1A1.E11.1tiny.6/epoch_182.pth'
cfg_path = '/mmdetection/work_dirs/Mx1A1.E11.1tiny.6/Mx1A1.E11.1tiny.6.py'

output_root = '/local-data/Polli/Outputs/Mx1A1/E11.1tiny.6/backyard/birdfeeders/images' # ~3.5it/s. possibly bad comparison b/c different threshold
#output_root = '/local-data/Polli/Outputs/Mx1A1/E11.1tiny.6/imago_RMBL_2020'

model = init_detector(cfg_path, ckpt_path)#, device='cuda')
score_thr = 0.8 # NOTE: 0.8 is best for backyard birdfeeders
save_labels_conf = True
classes = model.CLASSES


def infer(imagePath, score_thr=0.5, classes=model.CLASSES): # TODO: Convert to async detector, or bump up batch size.
    bbox_result = inference_detector(model, imagePath)
    return bbox_result


if not Path(output_root).exists():
    Path(output_root).mkdir(parents=True)

# Find all images in the input directory.
### Images can have extensions: '*.jpg','*.png','*.jpeg','*.JPG','*.PNG','*.JPEG'
if recursive:
    images = [f for f in Path(input_dir).rglob('*.jpg')]
    images.extend([f for f in Path(input_dir).rglob('*.png')])
    images.extend([f for f in Path(input_dir).rglob('*.jpeg')])
    images.extend([f for f in Path(input_dir).rglob('*.JPG')])
    images.extend([f for f in Path(input_dir).rglob('*.PNG')])
    images.extend([f for f in Path(input_dir).rglob('*.JPEG')])
else:
    images = [f for f in Path(input_dir).glob('*.jpg')]
    images.extend([f for f in Path(input_dir).glob('*.png')])
    images.extend([f for f in Path(input_dir).glob('*.jpeg')])
    images.extend([f for f in Path(input_dir).glob('*.JPG')])
    images.extend([f for f in Path(input_dir).glob('*.PNG')])
    images.extend([f for f in Path(input_dir).glob('*.JPEG')])
images = sorted(images)

for imagePath in tqdm(images):
    bbox_result = infer(imagePath, score_thr, classes)
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
        ]
    labels = np.concatenate(labels)
    imshow_det_bboxes(imagePath, bboxes=bboxes, labels=labels, class_names=classes, score_thr=score_thr, show=False, outRoot=output_root, out_file=None)













# def cropDetection(imagePath, bboxes):
#     im = Image.open(imagePath)
#     left = int(bboxes[0])
#     top = int(bboxes[1])
#     right = int(bboxes[2])
#     bottom = int(bboxes[3])
#     im1 = im.crop((left, top, right, bottom))
#     return im1    
    
# def postProcessResult(bbox_result, score_thr=0.5, classes=model.CLASSES):
#     labels = [
#     np.full(bbox.shape[0], i, dtype=np.int32)\
#     for i, bbox in enumerate(bbox_result)
#     ]
#     labels = np.concatenate(labels)
#     bboxes = np.vstack(bbox_result)
#     labels_impt = np.where(bboxes[:, -1] > score_thr)[0]
#     labels_conf = bboxes[labels_impt, -1]
#     labels_impt_list = [labels[i] for i in labels_impt]
#     labels_class = [classes[i] for i in labels_impt_list]
#     return labels_impt, labels_class, bboxes, labels_conf

# def makeOutputPath(imagePath, bboxes, output_root, labels_class, labels_conf, save_labels_conf):
#     "Make output path for the image. Each image will be saved in a folder named after the class. If the class folder doesn't exist, it will be created."
#     # Get the image name
#     image_name = imagePath.name
#     # Prefix image name with bbox coordinates and class
#     if save_labels_conf:
#         image_name = f'{int(bboxes[0])}_{int(bboxes[1])}_{int(bboxes[2])}_{int(bboxes[3])}_{labels_class}_{round(float(labels_conf),2)}_{image_name}'
#     else:
#         image_name = f'{int(bboxes[0])}_{int(bboxes[1])}_{int(bboxes[2])}_{int(bboxes[3])}_{labels_class}_{image_name}'
#     # Make output path
#     output_dir = Path(output_root) / labels_class
#     if not output_dir.exists():
#         output_dir.mkdir(parents=True, exist_ok=True)
#     output_path = output_dir / image_name
#     return output_path

# def infer(imagePath, score_thr=0.5, classes=model.CLASSES): # TODO: Convert to async detector, or bump up batch size.
#     bbox_result = inference_detector(model, imagePath) # TODO: Implement modified imshow_det_bboxes from detectionTools. Instead of drawing boxes, crop the image and save it.
#     # NOTE: It may ultimately be easier to modify the test.py script to do this to take advantage of the batch processing.
#     labels_impt, labels_class, bboxes, labels_conf = postProcessResult(bbox_result, score_thr=score_thr, classes=classes)
#     return labels_impt, labels_class, bboxes, labels_conf