from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import os
import os.path as osp
import numpy as np
import mmcv
 
def draw_bboxes(ax, bboxes, alpha=0.8, thickness=2):
    """Draw bounding boxes on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        bboxes (ndarray): The input bounding boxes with the shape
            of (n, 4).
        color (list[tuple] | matplotlib.color): the colors for each
            bounding boxes.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
        thickness (int): Thickness of lines. Default: 2.

    Returns:
        matplotlib.Axes: The result axes.
    """
    color = 'g'
    polygons = []
    for i, bbox in enumerate(bboxes):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
    p = PatchCollection(
        polygons,
        facecolor='none',
        edgecolors=color,
        linewidths=thickness,
        alpha=alpha)
    ax.add_collection(p)

    return ax

def makeCropPath(filename, outRoot, x1, y1, x2, y2, score, class_name, suffix= '.jpg'):
    """Make a filename for cropped image.

    Args:
        filename (str): The original filename.
        x1 (int): The x coordinate of the top left corner of the crop box.
        y1 (int): The y coordinate of the top left corner of the crop box.
        x2 (int): The x coordinate of the bottom right corner of the crop box.
        y2 (int): The y coordinate of the bottom right corner of the crop box.
        suffix (str): The suffix of the cropped image. Default: '.jpg'.

    Returns:
        str: The filename of the cropped image.
    """
    outDir = osp.join(outRoot, class_name)
    if not osp.exists(outDir):
        os.makedirs(outDir)
    filename = osp.basename(filename) # import os.path as osp
    filename = osp.splitext(filename)[0]
    score = str(int(round(float(score), 2)*100))
    filename = f'{x1}_{y1}_{x2}_{y2}_{score}_{filename}{suffix}'
    outPath = osp.join(outDir, filename)
    return outPath
    
    

def imshow_det_bboxes(img,
                      bboxes=None,
                      labels=None,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_size=8,
                      win_name='',
                      show=True,
                      wait_time=0,
                      crop_box=True,
                      outRoot=None,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str | ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray | None): Masks, shaped (n,h,w) or None.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown. Default: 0.
        bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        text_color (list[tuple] | tuple | str | None): Colors of texts.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        mask_color (list[tuple] | tuple | str | None, optional): Colors of
           masks. If a single color is given, it will be applied to all
           classes. The tuple of color should be in RGB order.
           Default: None.
        thickness (int): Thickness of lines. Default: 2.
        font_size (int): Font size of texts. Default: 13.
        show (bool): Whether to show the image. Default: True.
        win_name (str): The window name. Default: ''.
        wait_time (float): Value of waitKey param. Default: 0.
        crop_box (bool): Whether to crop the image according to the bbox, instead of drawing the bbox. Default: True.
        outRoot (str, optional): The root directory to save the cropped image. Subdirs will be created per-class. Default: None.
        out_file (str, optional): The filename to write the image.
            Default: None.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    # assert bboxes is None or bboxes.ndim == 2, \
    #     f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    # assert labels.ndim == 1, \
    #     f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    # assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
    #     f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    # assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
    #     'labels.shape[0] should not be less than bboxes.shape[0].'
    # assert segms is None or segms.shape[0] == labels.shape[0], \
    #     'segms.shape[0] and labels.shape[0] should have the same length.'
    # assert segms is not None or bboxes is not None, \
    #     'segms and bboxes should not be None at the same time.'

    inFilename = img
    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        # assert bboxes is not None and bboxes.shape[1] == 5 # NOTE: Why 5? Ans: (x1, y1, x2, y2, score)?
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        #labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    # fig = plt.figure(win_name, frameon=False)
    # plt.title(win_name)
    # canvas = fig.canvas
    # dpi = fig.get_dpi()
    # # add a small EPS to avoid precision lost due to matplotlib's truncation. value: 1e-5
    # # (https://github.com/matplotlib/matplotlib/issues/15363)
    # EPS = 1e-5
    # fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # # remove white edges by set subplot margin
    # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # ax = plt.gca()
    # ax.axis('off')


    # if bboxes is not None and crop_box is False:
    #     draw_bboxes(ax, bboxes, alpha=0.8, thickness=thickness)
    #     horizontal_alignment = 'left'
    #     positions = bboxes[:, :2].astype(np.int32) + thickness
    #     areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
    # if bboxes is not None and crop_box is True:
    #     (ax, bboxes, alpha=0.8, thickness=thickness)
    #     horizontal_alignment = 'left'
    #     positions = bboxes[:, :2].astype(np.int32) + thickness
    #     areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
    if crop_box is True and len(bboxes) > 0:
        # crop the image according to the bboxes
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox[:4]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            img_crop = img[y1:y2, x1:x2]
            try:
                img_crop = mmcv.bgr2rgb(img_crop)
                cropPath = makeCropPath(inFilename, outRoot, x1,y1,x2,y2, score=bbox[-1], class_name= class_names[labels[i]]) # Expects class_names to be a list of strings
                mmcv.imwrite(img_crop, cropPath) # NOTE: If these don't look right, check mmcv docs
            except:
                print("Error cropping image {} with bbox {}".format(osp.basename(inFilename), bbox))
    return None
 
    # plt.imshow(img)

    # stream, _ = canvas.print_to_buffer()
    # buffer = np.frombuffer(stream, dtype='uint8')
    # img_rgba = buffer.reshape(height, width, 4)
    # rgb, alpha = np.split(img_rgba, [3], axis=2)
    # img = rgb.astype('uint8')
    # img = mmcv.rgb2bgr(img)

    # if show:
    #     # We do not use cv2 for display because in some cases, opencv will
    #     # conflict with Qt, it will output a warning: Current thread
    #     # is not the object's thread. You can refer to
    #     # https://github.com/opencv/opencv-python/issues/46 for details
    #     if wait_time == 0:
    #         plt.show()
    #     else:
    #         plt.show(block=False)
    #         plt.pause(wait_time)
    # if out_file is not None:
    #     mmcv.imwrite(img, out_file)

    # plt.close()

    # return img