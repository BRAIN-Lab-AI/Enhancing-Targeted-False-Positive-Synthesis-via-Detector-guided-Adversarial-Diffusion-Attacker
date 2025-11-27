import sys
import os
import torch
from PIL import Image, ImageFilter

# Fix paths
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "Diffusion"))
sys.path.append(os.path.join(ROOT, "utils"))

from torchvision import transforms          # ← ADD THIS
from Diffusion.Model import UNet
from Diffusion import DADA


device = "cuda"

# --- Directories ---
real_img_dir  = "./dataset/images/"
real_mask_dir = "./dataset/masks/"
save_dir      = "./data_aug/images/"

os.makedirs(save_dir, exist_ok=True)

# --- Load BG-De model ---
model = UNet(T=1000, ch=64, ch_mult=[1, 2, 3, 4, 4], attn=[2], num_res_blocks=2, dropout=0)
model = torch.nn.DataParallel(model).to(device)
model.load_state_dict(torch.load("./models/ckpt_7999_.pt", map_location=device))
model.eval()

# --- Loop over dataset ---
image_files = sorted(os.listdir(real_img_dir))

transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_mask = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

for fname in image_files:
    img_path = os.path.join(real_img_dir, fname)
    mask_path = os.path.join(real_mask_dir, fname.replace(".jpg", ".png"))

    # Load image
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").filter(ImageFilter.GaussianBlur(4))

    img_t  = transform_img(img).unsqueeze(0).to(device)
    mask_t = transform_mask(mask).unsqueeze(0).to(device)

    # Build sampler
    sampler = DADA(
    model,
    img_t,
    mask_t,
    1e-4,
    0.02,
    1000,
    "./Detection_model/yolo.pt")

    # Generate 1 false positive
    noise = torch.randn([1,3,256,256], device=device)


    gen = sampler(noise)
    gen = gen * 0.5 + 0.5  # Unnormalize

    # Save image
    save_path = os.path.join(save_dir, fname)
    transforms.ToPILImage()(gen.squeeze().cpu()).save(save_path)

    print("Saved:", save_path)
