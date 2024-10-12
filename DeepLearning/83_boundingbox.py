#目标检测数据集，每行表示一个物体包含图片文件名、物体类别、边缘框。
#边缘框(bounding box)可以由 1.左上角和右下角xy坐标 2.中心坐标xy和宽度高度 确定
#图像中坐标的原点是图像的左上角，向右的方向为x轴的正方向，向下的方向为y轴的正方向

import torch
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.plt.imread('./pictures/girl2.png')
d2l.plt.imshow(img)
#d2l.plt.show()

#对两种边缘框表示方法进行互相转换
def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

#将边界框画出来
def bbox_to_rect(bbox, color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # ((左上x,左上y),宽,高)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

# bbox是边界框的英文缩写
dog_bbox, cat_bbox = [227.0, 132.0, 880.0, 1070.0], [1050.0, 135.0, 1650.0, 1070.0]

fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
d2l.plt.show()