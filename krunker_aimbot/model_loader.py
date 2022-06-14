from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import check_img_size, increment_path, non_max_suppression, scale_coords, xyxy2xywh
import config


def get_model():
    # Load model
    device = select_device('')
    model = DetectMultiBackend(config.WEIGHTS, device=device, dnn=False, data=config.DATA, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(config.IMGSZ, s=stride)  # check image size

    return model, device, stride, names
