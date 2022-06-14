import mss
import numpy as np
import cv2
import torch
import config

from krunker_aimbot.model_loader import get_model
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import Annotator, colors

sct = mss.mss()
screen_width = 2560
screen_height = 1440
GAME_LEFT, GAME_TOP, GAME_WIDTH, GAME_HEIGHT = screen_width // 4, screen_height // 4, screen_width // 2, screen_height // 2
# GAME_LEFT, GAME_TOP, GAME_WIDTH, GAME_HEIGHT = 0, 0, screen_width, screen_height # original screen size
RESIZE_WIDTH, RESIZE_HEIGHT = screen_width // 2, screen_height // 2  # 視窗顯示大小
monitor = {
    'left': GAME_LEFT,
    'top': GAME_TOP,
    'width': GAME_WIDTH,
    'height': GAME_HEIGHT
}
window_name = 'test'
# get model
model, device, stride, names = get_model()
imgsz = check_img_size(config.IMGSZ, s=stride)


@torch.no_grad()  # 推理不需要反向傳播，所以不需要紀錄梯度，節省內存，提高推理速度
def pred_img(img0):
    # Padded resize
    img = letterbox(img0, imgsz, stride=stride, auto=True)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    # Run inference
#    model.warmup(imgsz=1)  # warmup
    img = torch.from_numpy(img).to(device)
    img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
    img /= 255  # 歸一化
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    # Inference
    pred = model(img, augment=False, visualize=False)

    # NMS
    pred = non_max_suppression(pred, config.CONF_THRES, config.IOU_THRES, None, False, max_det=config.MAX_DET)

    # Process predictions
    det = pred[0]
    im0 = img0.copy()
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    annotator = Annotator(im0, line_width=config.LINE_THICKNESS, example=str(names))
    xywh_list = []
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        # Write results
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            xywh_list.append(xywh)

            c = int(cls)  # integer class
            label = None if config.HIDE_LABELS else (names[c] if config.HIDE_CONFIGS else f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))

    # Stream results
    im0 = annotator.result()
    return im0, xywh_list


while True:
    grab_img = sct.grab(monitor=monitor)
    grab_img = np.array(grab_img)
    grab_img = cv2.cvtColor(grab_img, cv2.COLOR_BGRA2BGR)
    grab_img, aims = pred_img(grab_img)  # aims = xywh_list

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, RESIZE_WIDTH, RESIZE_HEIGHT)
    cv2.imshow(window_name, grab_img)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC
        cv2.destroyAllWindows()
        exit('ESC...')
