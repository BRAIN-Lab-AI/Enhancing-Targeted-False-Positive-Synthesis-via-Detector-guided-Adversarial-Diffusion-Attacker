import os
from typing import Dict
from PIL import Image, ImageFilter

import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from Diffusion import DADA, Trainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler


# ===========================
# Dataset
# ===========================
class CustomDataset(Dataset):
    def __init__(self, image_root, mask_root, transform=None):
        self.image_root = image_root
        self.mask_root = mask_root
        self.transform = transform
        self.images = sorted(os.listdir(image_root))
        self.masks = sorted(os.listdir(mask_root))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_root, self.images[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_root, self.masks[idx])).convert("L")

        mask = transforms.ToTensor()(mask)

        if self.transform:
            img = self.transform(img)

        return img, mask


# ===========================
# TRAIN
# ===========================
def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    
    # dataset + loader
    dataset = CustomDataset(
        image_root=modelConfig["train_image_dir"],
        mask_root=modelConfig["train_mask_dir"],
        transform=transforms.Compose([
            transforms.Resize((modelConfig["img_size"], modelConfig["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=True,
                            num_workers=4, drop_last=True, pin_memory=True)

    # model
    net_model = UNet(
        T=modelConfig["T"],
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        attn=modelConfig["attn"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"]
    )
    net_model = nn.DataParallel(net_model).to(device)

    # load pretrained BG-De?
    if modelConfig["training_load_weight"] is not None:
        ckpt = torch.load(modelConfig["training_load_weight"], map_location=device)
        net_model.load_state_dict(ckpt)

    # optimizer + scheduler
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=modelConfig["epoch"],
        eta_min=0
    )
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=modelConfig["multiplier"],
        warm_epoch=modelConfig["epoch"] // 10,
        after_scheduler=cosineScheduler
    )
    # ===== OneCycleLR Scheduler (recommended for stable DDPM training) =====
    steps_per_epoch = len(dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=modelConfig["lr"],      # peak learning rate
        total_steps=modelConfig["epoch"] * steps_per_epoch,
        pct_start=0.1,                 # 10% warm-up
        anneal_strategy='cos',         # cosine annealing
        div_factor=10.0,               # initial LR = max_lr/10
        final_div_factor=100.0         # final LR = max_lr/100
    )

    # trainer
    trainer = Trainer(
        ddpm_model=net_model,
        beta_1=modelConfig["beta_1"],
        beta_T=modelConfig["beta_T"],
        T=modelConfig["T"]
    ).to(device)

    # training loop
    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for batch_idx, (images, masks) in enumerate(tqdmDataLoader):

                optimizer.zero_grad()

                x_0 = images.to(device)
                masks = masks.to(device)

                loss = trainer(x_0, masks).mean()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(net_model.parameters(),
                                              modelConfig["grad_clip"])

                optimizer.step()
                scheduler.step()   # <<< IMPORTANT

                tqdmDataLoader.set_postfix({
                    "epoch": e,
                    "loss": f"{loss.item():.6f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "step": f"{batch_idx+1}/{len(dataloader)}"
                })

        # save last 5 epochs
        if e >= modelConfig["epoch"] - 5:
            torch.save(
                net_model.state_dict(),
                os.path.join(modelConfig["save_weight_dir"],
                            f"ckpt_{e}_.pt")
            )



# ===========================
# EVAL
# ===========================
def eval(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    model = UNet(
        T=modelConfig["T"],
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        attn=modelConfig["attn"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=0
    )
    model = nn.DataParallel(model).to(device)

    ckpt = torch.load(modelConfig["BG-De_weight"], map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    print("Model loaded.")

    base_img = Image.open(modelConfig["inpaint_image"]).convert("RGB")
    mask_img = Image.open(modelConfig["inpaint_mask"]).convert("L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(5))

    base_tensor = transforms.ToTensor()(base_img).unsqueeze(0).to(device)
    mask_tensor = transforms.ToTensor()(mask_img).unsqueeze(0).to(device)

    sampler = DADA(
        model,
        base_tensor,
        mask_tensor,
        modelConfig["beta_1"],
        modelConfig["beta_T"],
        modelConfig["T"],
        modelConfig["Detection_weight"]
    ).to(device)

    noise = torch.randn([1, 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
    sampled = sampler(noise)
    sampled = sampled * 0.5 + 0.5

    save_image(sampled, os.path.join(modelConfig["sampled_dir"], "DADA.png"))

'''
import os
from typing import Dict
from PIL import Image, ImageDraw
from PIL import ImageFilter

import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image

from Diffusion import DADA, Trainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler
# ===== FIXED: Add missing imports =====
from utils.loss import DADALoss
from Diffusion import VGGPerceptualStyleLoss  
# ===== END FIXED =====

class CustomDataset(Dataset):
    def __init__(self, image_root, mask_root, transform=None):
        self.image_root = image_root
        self.mask_root = mask_root
        self.transform = transform
        self.images = sorted(os.listdir(image_root))
        self.masks = sorted(os.listdir(mask_root))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_root, self.images[idx])
        mask_path = os.path.join(self.mask_root, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  

        to_tensor = transforms.ToTensor()
        mask = to_tensor(mask)

        if self.transform:
            image = self.transform(image)

        return image, mask

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    
    dataset = CustomDataset(
        image_root = modelConfig["train_image_dir"],
        mask_root = modelConfig["train_mask_dir"],
        transform=transforms.Compose([
            transforms.Resize((modelConfig["img_size"], modelConfig["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"], num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"])
    
    net_model = nn.DataParallel(net_model).to(device)
    #================= my enhance =======================
    # ---- NEW: create perceptual+style loss module ----
    perc_style_loss = VGGPerceptualStyleLoss(
        lambda_perc=modelConfig.get("lambda_perc", 0.1),
        lambda_style=modelConfig.get("lambda_style", 0.05)
    ).to(device)

    # ---- Pass it into Trainer ----
    trainer = Trainer(
        ddpm_model=net_model,
        dadaloss=None,                      # or your existing DADALoss if you use it
        perc_style_loss=perc_style_loss,    # <-- our new regularizer
        beta_1=modelConfig["beta_1"],
        beta_T=modelConfig["beta_T"],
        T=modelConfig["T"]
    ).to(device)

#================= end enhance ========================
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(modelConfig["BG-De_weight"]), map_location=device))
    
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    # ===== FIXED: OneCycleLR Scheduler for better training stability =====
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=modelConfig["lr"],
        total_steps=modelConfig["epoch"] * len(dataloader),
        pct_start=0.1,  # 10% of training for warmup
        div_factor=10.0,  # Initial LR = max_lr/10
        final_div_factor=100.0,  # Final LR = max_lr/100
        anneal_strategy='cos'
    )
    # ===== END FIXED =====
    
    # ===== FIXED: Create proper dummy model with parameters =====
    class DummyModel(nn.Module):
        def __init__(self, device):
            super().__init__()
            # Add a real parameter to make the model device-aware
            self.register_parameter("dummy_param", nn.Parameter(torch.zeros(1, device=device)))
            
            self.hyp = {
                "box": 0.05,
                "obj": 1.0, 
                "cls": 0.5,
                "anchor_t": 4.0,
                "fl_gamma": 0.0,
                "epochs": modelConfig["epoch"],
                "label_smoothing": 0.0
            }
            # Create a proper module list
            self.model = nn.ModuleList([nn.Sequential(
                nn.Conv2d(3, 1, 1),
                nn.Sigmoid()
            )])
            # Register anchors as a buffer
            self.register_buffer("anchors", torch.tensor([[[10,13], [16,30], [33,23]]], dtype=torch.float32))

    # Create dummy model with device parameter
    dummy_model = DummyModel(device)
    dadaloss = DADALoss(dummy_model, autobalance=True, region_adaptive=True)
    # ===== END FIXED =====

    trainer = Trainer(net_model, dadaloss, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            # ===== FIXED: Define batch index variable =====
            for i, (images, masks) in enumerate(tqdmDataLoader):
            # ===== END FIXED =====
                optimizer.zero_grad()

                x_0 = images.to(device)
                masks = masks.to(device)

                # ===== FIXED: Remove commented code and fix triple quotes =====
                #loss = trainer(x_0, masks, epoch=e).sum() / 1000.
                loss = trainer(x_0, masks, epoch=e)  # returns scalar

                # ===== END FIXED =====
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()

                # Update the enhanced scheduler
                scheduler.step()

                # Enhanced progress display with region information
                current_lr = optimizer.param_groups[0]['lr']
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e, 
                    "loss": f"{loss.item():.4f}",
                    "img_shape": f"{x_0.shape[-2:]}",
                    "LR": f"{current_lr:.2e}",
                    "step": f"{i+1}/{len(dataloader)}"
                })
        
        # Scheduler is stepped inside the loop, no need to step here
        pass

        if e >= modelConfig["epoch"] - 5:
            torch.save(net_model.state_dict(), os.path.join(modelConfig["save_weight_dir"], f'ckpt_{e}_.pt'))


def eval(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"], num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)

    model = nn.DataParallel(model).to(device)

    ckpt = torch.load(os.path.join(modelConfig["BG-De_weight"]), map_location=device)
    model.load_state_dict(ckpt)
    print("model load weight done.")
    model.eval()

    base_image_path = modelConfig["inpaint_image"]
    base_image = Image.open(base_image_path).convert("RGB")
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    base_tensor = base_transform(base_image).unsqueeze(0) 

    mask_image_path = modelConfig["inpaint_mask"]
    mask_image = Image.open(mask_image_path).convert("L")
    blur_radius = 5  
    mask_image = mask_image.filter(ImageFilter.GaussianBlur(blur_radius))
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    mask_tensor = mask_transform(mask_image).unsqueeze(0)  

    sampler = DADA(model, base_tensor, mask_tensor, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], modelConfig["Detection_weight"]).to(device)
    noisyImage = torch.randn([1, 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)

    sampledImgs = sampler(noisyImage)
    sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
    save_image(sampledImgs, os.path.join(modelConfig["sampled_dir"], f"DADA.png"))
    '''