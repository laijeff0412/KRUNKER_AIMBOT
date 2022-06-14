IMGSZ = (640, 640)
WEIGHTS = 'weights/krunker_aimbot.pt'  # model.pt path(s)
DATA = 'custom_data_wandb.yaml'  # dataset.yaml path
CONF_THRES = 0.5  # confidence threshold
IOU_THRES = 0.45  # NMS IOU threshold
MAX_DET = 5  # maximum detections per image
LINE_THICKNESS = 3  # bounding box thickness (pixels)
HIDE_LABELS = False  # hide labels
HIDE_CONFIS = False  # hide confidences


# view_img=False,  # show results
# save_txt=False,  # save results to *.txt
# save_conf=False,  # save confidences in --save-txt labels
# save_crop=False,  # save cropped prediction boxes
# nosave=False,  # do not save images/videos
# classes=None,  # filter by class: --class 0, or --class 0 2 3
# agnostic_nms=False,  # class-agnostic NMS
# augment=False,  # augmented inference
# visualize=False,  # visualize features
# update=False,  # update all models
# # project=ROOT / 'runs/detect',  # save results to project/name
# name='exp',  # save results to project/name
# exist_ok=False,  # existing project/name ok, do not increment
#
#
# half=False,  # use FP16 half-precision inference
# dnn=False,  # use OpenCV DNN for ONNX inference