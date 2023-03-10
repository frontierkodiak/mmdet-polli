from PIL import Image
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from pathlib import Path
import numpy as np


input_dir = '/local-data/Polli/Datasets/backyard/birdfeeders/images'
ckpt_path = '/mmdetection/work_dirs/Mx1A1.E11.1tiny.6/epoch_104.pth'
cfg_path = '/mmdetection/work_dirs/Mx1A1.E11.1tiny.6/Mx1A1.E11.1tiny.6.py'

output_root = '/lake/Polli/Outputs/Mx1A1/E11.1tiny.6/backyard/birdfeeders/images'

model = init_detector(cfg_path, ckpt_path, device='cuda:0')
score_thr = 0.7
classes = model.CLASSES

# Find all images in the input directory. Don't search recursively.
images = list(Path(input_dir).glob('*.jpg','*.png','*.jpeg','*.JPG','*.PNG','*.JPEG'))


def cropDetection(imagePath, bboxes, labels_impt):
    im = Image.open(imagePath)
    left = bboxes[labels_impt][0][0]
    top = bboxes[labels_impt][0][1]
    right = bboxes[labels_impt][0][2]
    bottom = bboxes[labels_impt][0][3]
    im1 = im.crop((left, top, right, bottom))
    return im1    
    
def postProcessResult(bbox_result, score_thr=0.5, classes=model.CLASSES):
    labels = [
    np.full(bbox.shape[0], i, dtype=np.int32)\
    for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(bbox_result)
    labels_impt = np.where(bboxes[:, -1] > 0.3)[0]

    labels_impt_list = [labels[i] for i in labels_impt]
    labels_class = [classes[i] for i in labels_impt_list]
    
    return labels_impt, labels_class, bboxes

def infer(imagePath, score_thr=0.5, classes=model.CLASSES):
    bbox_result = inference_detector(model, imagePath)
    labels_impt, labels_class, bboxes = postProcessResult(bbox_result, score_thr=score_thr, classes=classes)
    return labels_impt, labels_class, bboxes

def makeOutputPath(imagePath, bboxes, output_root, labels_impt, labels_class):
    "Make output path for the image. Each image will be saved in a folder named after the class. If the class folder doesn't exist, it will be created."
    # Get the image name
    image_name = imagePath.name
    # Prefix image name with bbox coordinates and class
    image_name = f'{bboxes[labels_impt][0][0]}_{bboxes[labels_impt][0][1]}_{bboxes[labels_impt][0][2]}_{bboxes[labels_impt][0][3]}_{labels_class[0]}_{image_name}'
    # Make output path
    output_path = Path(output_root) / labels_class[0] / image_name
    return output_path

for imagePath in images:
    labels_impt, labels_class, bboxes = infer(imagePath, score_thr=score_thr, classes=classes)
    if len(labels_impt) > 0:
        k = 0
        for i in range(len(labels_impt)):
            output_path = makeOutputPath(imagePath, bboxes[k], output_root, labels_impt[k], labels_class[k])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            im1 = cropDetection(imagePath, bboxes[k], labels_impt[k])
            im1.save(output_path)
            k += 1