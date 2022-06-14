import mss
import numpy as np
import cv2
import torch
from pynput import mouse
import time
import config
import pynput
import pyautogui
from krunker_aimbot.model_loader import get_model
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import Annotator, colors

sct = mss.mss()
screen_width = 2560
screen_height = 1440
RESIZE_WIDTH, RESIZE_HEIGHT = screen_width // 4, screen_height // 4  # 視窗顯示大小
monitor = {
    'left': screen_width // 5,
    'top': screen_height // 5,
    'width': (screen_width // 5) * 3,
    'height': (screen_height // 5) * 3
}
window_name = 'detect'
# get model
model, device, stride, names = get_model()
imgsz = check_img_size(config.IMGSZ, s=stride)

# mouse control
mouse_controller = pynput.mouse.Controller()


@torch.no_grad()  # 推理不需要反向傳播，所以不需要紀錄梯度，節省內存，提高推理速度
def pred_img(img0): # img0 = grab_img from 'while loop'
    # Padded resize
    img = letterbox(img0, imgsz, stride=stride, auto=True)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img) # 將array在記憶體存放的位址變成連續

    # Run inference
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
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    annotator = Annotator(img0, line_width=config.LINE_THICKNESS, example=str(names))
    xywh_list = []
    if len(det):
        # Rescale boxes from img_size to img0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        # Write results
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            xywh_list.append(xywh)

            c = int(cls)  # integer class
            label = None if config.HIDE_LABELS else (names[c] if config.HIDE_CONFIS else f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))

    # Stream results
    img0 = annotator.result()
    return img0, xywh_list


def mouse_aim_controller(xywh_list, left, top, width,
                         height):  # xywh:predict position # left,top,width,height:monitor position
    # mouse absolute position
    mouse_x, mouse_y = pyautogui.position()
    # monitor area position
    best_xy = None
    for xywh in xywh_list:
        x, y, _, _ = xywh
        # 歸依化還原
        x *= width
        y *= height
        # transform to original screen coordinate
        x += left
        y += top
        dist = (x - mouse_x) ** 2 + (y - mouse_y) ** 2  # the distance to the center of the screen
        if not best_xy:
            best_xy = ((x, y), dist)
        else:
            _, old_dist = best_xy
            if dist < old_dist:
                best_xy = ((x, y), dist)

    x, y = best_xy[0]
    pyautogui.move(x, y)


LOCK_AIM = False


def on_click(x, y, button, pressed):
    global LOCK_AIM
    if button == button.x1:
        if pressed:
            LOCK_AIM = not LOCK_AIM
            print('自瞄:'f"{LOCK_AIM and '開' or '關'}")


listener = mouse.Listener(on_click=on_click)
listener.start()
pTime = 0
while True:
    grab_img = sct.grab(monitor=monitor) # much faster than using opencv
    grab_img = np.array(grab_img)
    grab_img = cv2.cvtColor(grab_img, cv2.COLOR_BGRA2BGR)
    grab_img, aims = pred_img(grab_img)  # aims = xywh_list
    if aims and LOCK_AIM:
        mouse_aim_controller(aims, **monitor)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, RESIZE_WIDTH, RESIZE_HEIGHT)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(grab_img, f'FPS: {int(fps+5)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 255), 6)
    cv2.putText(grab_img, 'auto_aim:OFF' if LOCK_AIM else 'auto_aim:ON', (20, 140), cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 255), 6)

    cv2.imshow(window_name, grab_img)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC
        cv2.destroyAllWindows()
        exit('ESC...')
