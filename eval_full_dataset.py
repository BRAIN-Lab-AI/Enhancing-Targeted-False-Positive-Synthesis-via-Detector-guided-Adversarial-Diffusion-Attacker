import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.utils import save_image
from Diffusion import DADA
from Diffusion.Model import UNet

from utils.metrics import ap_per_class, box_iou

# -----------------------------------------
# CONFIG
# -----------------------------------------
device = "cuda"
img_dir = "./dataset/images/"
mask_dir = "./dataset/masks/"
save_gen = "./FullEval/"
os.makedirs(save_gen, exist_ok=True)

detection_weight = "./Detection_model/yolov5l.pt"
bgde_weight = "./BG-De_model/kvasir.pt"

# -----------------------------------------
# Load BG-De Model
# -----------------------------------------
model = UNet(T=1000, ch=64, ch_mult=[1,2,3,4,4], attn=[2],
             num_res_blocks=2, dropout=0.)
model = torch.nn.DataParallel(model).to(device)
model.load_state_dict(torch.load(bgde_weight, map_location=device))
model.eval()

# -----------------------------------------
# Load YOLO Detector
# -----------------------------------------
detector = torch.hub.load(
    'ultralytics/yolov5', 'custom',
    path=detection_weight, force_reload=False
).to(device)
detector.conf = 0.25

# -----------------------------------------
# Image & Mask transforms
# -----------------------------------------
tf_img = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
tf_mask = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

# Accumulators for metrics
all_tp = []
all_fp = []
all_conf = []
all_pred_cls = []
all_target_cls = []

# -----------------------------------------
# START LOOP
# -----------------------------------------
names = sorted(os.listdir(img_dir))
for name in names:

    img_path = os.path.join(img_dir, name)
    mask_path = os.path.join(mask_dir, name)

    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    mask = mask.filter(ImageFilter.GaussianBlur(5))

    img_t = tf_img(image).unsqueeze(0).to(device)
    mask_t = tf_mask(mask).unsqueeze(0).to(device)

    # DADA Sampler
    sampler = DADA(model, img_t, mask_t, 1e-4, 0.02, 1000, detection_weight).to(device)
    noise = torch.randn([1, 3, 256, 256], device=device)

    generated = sampler(noise)
    generated = generated * 0.5 + 0.5
    save_image(generated, os.path.join(save_gen, f"dada_{name}"))

    # YOLO Detection
    pil_img = transforms.ToPILImage()(generated[0].cpu())
    results = detector(pil_img)
    det = results.xyxy[0].cpu().numpy()

    # No prediction case
    if len(det) == 0:
        # 0 TP, 0 FP → store dummy zeros
        all_tp.append(0)
        all_fp.append(0)
        continue

    boxes = det[:, :4]
    conf = det[:, 4]
    cls = det[:, 5].astype(int)

    # Compute GT box from mask
    mask_np = mask_t.cpu().numpy().squeeze()
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        continue
    gt_box = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]])
    gt_cls = np.array([0])

    ious = box_iou(torch.tensor(boxes), torch.tensor(gt_box)).numpy().squeeze()

    # Ensure ious is 1D array
    ious = np.atleast_1d(ious)

    # True/False positives
    tp_flags = (ious > 0.5).astype(int)
    tp_flags = np.atleast_1d(tp_flags)

    all_tp.extend(tp_flags.tolist())
    all_fp.extend((1 - tp_flags).tolist())

    all_conf.extend(conf.tolist())
    all_pred_cls.extend(cls.tolist())
    all_target_cls.extend(gt_cls.tolist() * len(tp_flags))

print("\nComputing final metrics...")

tp, fp, p, r, f1, ap, classes = ap_per_class(
    np.array(all_tp).reshape(-1,1),
    np.array(all_conf),
    np.array(all_pred_cls),
    np.array(all_target_cls)
)

print("Precision:", p.mean())
print("Recall:", r.mean())
print("F1:", f1.mean())
print("mAP@0.5:", ap[:,0].mean())
print("mAP@0.5:0.95:", ap.mean())
