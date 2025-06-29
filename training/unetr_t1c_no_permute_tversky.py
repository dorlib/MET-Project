#!/usr/bin/env python3
# unetr_t1ce_training_with_augmentation.py - 3D UNETR training on T1CE-only .npy volumes

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from self_attention_cv import UNETR
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
NUMBER_OF_LAYERS = 19
BASE_DIR         = "../Data/"
TRAIN_IMG_DIR    = os.path.join(BASE_DIR, "input_data_128/train/images/")
TRAIN_MASK_DIR   = os.path.join(BASE_DIR, "input_data_128/train/masks/")
VAL_IMG_DIR      = os.path.join(BASE_DIR, "input_data_128/val/images/")
VAL_MASK_DIR     = os.path.join(BASE_DIR, "input_data_128/val/masks/")
TEST_IMG_DIR     = os.path.join(BASE_DIR, "input_data_128/test/images/")
TEST_MASK_DIR    = os.path.join(BASE_DIR, "input_data_128/test/masks/")
SAVED_MODEL_PATH = os.path.join(BASE_DIR, "saved_models/brats_t1ce-tversky.pth")
BATCH_VIS_DIR    = os.path.join(BASE_DIR, "predictions_visualizations")
GETITEM_VIS_DIR  = os.path.join(BASE_DIR, "train_samples_visualizations")
DEVICE           = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE       = 2
NUM_CLASSES      = 4
NUM_EPOCHS       = 300
LR               = 1e-4

# ──────────────────────────────────────────────────────────────
# Utility: numeric sort key
# ──────────────────────────────────────────────────────────────
def numeric_sort_key(fname: str) -> int:
    try:
        return int(os.path.basename(fname).split('_')[1].split('.')[0])
    except:
        return 0

# ──────────────────────────────────────────────────────────────
# Dataset with augmentation, normalization, and per-axis visualization
# ──────────────────────────────────────────────────────────────
class NpyT1cImageDataset(Dataset):
    def __init__(self, img_dir, img_list, mask_dir, mask_list, augment=False):
        self.img_list  = sorted(img_list, key=numeric_sort_key)
        self.mask_list = sorted(mask_list, key=numeric_sort_key)
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.augment   = augment

    def __len__(self):
        return len(self.img_list)

    def augment_data(self, img, mask):
        # 1) intensity normalization (z-score)
        img = (img - img.mean()) / (img.std() + 1e-5)
        augment_probability = 0.25
        # 2) random flips on each axis (D, H, W)
        for ax in (0, 1, 2):
            if np.random.rand() < augment_probability:
                img  = np.flip(img,  axis=ax)
                mask = np.flip(mask, axis=ax)

        # 3) random 90° rotate in a random plane
        if np.random.rand() < augment_probability:
            k = np.random.randint(1, 4)
            plane = np.random.choice([(0,1), (0,2), (1,2)])
            
            img  = np.rot90(img,  k, axes=plane)
            mask = np.rot90(mask, k, axes=plane)

        return img, mask

    def __getitem__(self, idx):
        img_name  = self.img_list[idx]
        mask_name = self.mask_list[idx]
        img = np.load(os.path.join(self.img_dir, img_name))
        if img.ndim == 4 and img.shape[-1] > 1:
            img = img[..., 1]

        mask = np.load(os.path.join(self.mask_dir, mask_name))
        if mask.ndim == 4 and mask.shape[-1] > 1:
            mask = np.argmax(mask, axis=-1)

        # Apply augmentation & normalization
        if self.augment:
            img, mask = self.augment_data(img, mask)
        else:
            img = (img - img.mean()) / (img.std() + 1e-5)

        # Convert to tensors
        #img = img[..., np.newaxis].astype(np.float32)
         #print(img.shape)
        #img_tensor  = torch.tensor(img).permute(3,2,0,1)
        #mask_tensor = torch.tensor(mask).permute(2,0,1)
        img = np.ascontiguousarray(img.astype(np.float32))     # shape(D,H,W)
        mask = np.ascontiguousarray(mask.astype(np.uint8))     # shape (D,H,W)
        img_tensor  = torch.from_numpy(img).unsqueeze(0)        # → [1,D,H,W]
        mask_tensor = torch.from_numpy(mask).long()             # → [D,H,W]

        # Visualization
        os.makedirs(GETITEM_VIS_DIR, exist_ok=True)
        C, D, H, W = img_tensor.shape
        mids = {'axial': D//2, 'coronal': H//2, 'sagittal': W//2}
        slices_img = {
            'axial':   img_tensor[0, mids['axial'], :, :].cpu().numpy(),
            'coronal': img_tensor[0, :, mids['coronal'], :].cpu().numpy(),
            'sagittal':img_tensor[0, :, :, mids['sagittal']].cpu().numpy()
        }
        slices_mask = {
            'axial':   mask_tensor[mids['axial'], :, :].cpu().numpy(),
            'coronal': mask_tensor[:, mids['coronal'], :].cpu().numpy(),
            'sagittal':mask_tensor[:, :, mids['sagittal']].cpu().numpy()
        }
        fig, axs = plt.subplots(3, 2, figsize=(6,9), dpi=150)
        for i, axis in enumerate(['axial','coronal','sagittal']):
            axs[i,0].imshow(slices_img[axis], interpolation='nearest')
            axs[i,0].set_title(f"Img {axis}");    axs[i,0].axis('off')
            axs[i,1].imshow(slices_mask[axis], interpolation='nearest')
            axs[i,1].set_title(f"Mask {axis}");   axs[i,1].axis('off')
        plt.tight_layout()
        fig.savefig(os.path.join(GETITEM_VIS_DIR, f"{numeric_sort_key(img_name)}_getitem.png"))
        plt.close(fig)

        return img_tensor, mask_tensor, img_name, mask_name

# ──────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────
def build_unetr():
    model = UNETR(
        img_shape=(128,128,128),
        input_dim=1,
        output_dim=NUM_CLASSES,
        embed_dim=128,
        patch_size=16,
        num_heads=8,
	ext_layers= [3, 6, 9, 12, 15, 18],
        norm='instance',
        dropout=0.2,
        base_filters=16,
        dim_linear_block=1024
    ).to(DEVICE)
    print(f"UNETR has {sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")
    return model

# ──────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────
def weighted_ce_loss():
    cols = [str(i) for i in range(NUM_CLASSES)]
    df = pd.DataFrame(columns=cols)
    for mask_file in sorted(glob.glob(os.path.join(TRAIN_MASK_DIR, "*.npy")), key=numeric_sort_key):
        m = np.load(mask_file)
        if m.ndim == 4:
            m = np.argmax(m, axis=-1)
        vals, cnt = np.unique(m, return_counts=True)
        row = dict.fromkeys(cols, 0)
        row.update(dict(zip(vals.astype(str), cnt)))
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    label_sum = df.sum(); total = label_sum.sum()
    weights = [total/(NUM_CLASSES*label_sum[str(i)]) for i in range(NUM_CLASSES)]
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))


def dice_loss(smooth: float = 1e-5):
    def _dice_loss_fn(logits, y):
        y_onehot = F.one_hot(y, num_classes=NUM_CLASSES) \
                       .permute(0, 4, 1, 2, 3).float()
        probs = F.softmax(logits, dim=1)
        intersection = torch.sum(probs * y_onehot, (2,3,4))
        cardinality  = torch.sum(probs + y_onehot, (2,3,4))
        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth)
        return 1.0 - dice_score.mean()
    return _dice_loss_fn
    



def tversky_loss(alpha: float = 0.7, smooth: float = 1e-5):
    """
    Multiclass Tversky loss.
      logits: (N, C, D, H, W)
      y:      (N, D, H, W) with 0..C-1 ints
    """
    def _loss(logits, y):
        N, C, D, H, W = logits.shape
        probs = F.softmax(logits, dim=1)  # (N,C,D,H,W)
        y_onehot = F.one_hot(y, num_classes=C) \
                       .permute(0,4,1,2,3).float()
        
        tp = (probs * y_onehot).sum((2,3,4))
        fp = (probs * (1 - y_onehot)).sum((2,3,4))
        fn = ((1 - probs) * y_onehot).sum((2,3,4))

        tversky = (tp + smooth) / (tp + alpha*fn + (1-alpha)*fp + smooth)
        return 1.0 - tversky.mean()  # average over batch & classes

    return _loss

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval(); loss_sum=acc_sum=0.0
    for x,y,_,_ in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss_sum += criterion(logits,y).item()
        preds = torch.argmax(logits,dim=1)
        acc_sum += (preds==y).float().mean().item()
    return loss_sum/len(loader), acc_sum/len(loader)

# ──────────────────────────────────────────────────────────────
# Training and plotting
# ──────────────────────────────────────────────────────────────
def plot_training_curves(train_losses,val_losses,val_accs,output_path=f'training_curves_num heads=8_tversky.png'):
    epochs = range(1,len(train_losses)+1)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
    ax1.plot(epochs,train_losses,'-',label='Train Loss')
    ax1.plot(epochs,val_losses,'-',label='Val Loss')
    ax1.set(title='Loss',xlabel='Epoch',ylabel='Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(epochs,val_accs,'-',label='Val Acc')
    ax2.set(title='Accuracy',xlabel='Epoch',ylabel='Acc'); ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.savefig(output_path); plt.close(fig)


def train(model,train_loader,val_loader,criterion):
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    os.makedirs(BATCH_VIS_DIR, exist_ok=True)
    train_losses, val_losses, val_accs = [], [], []
    for epoch in range(NUM_EPOCHS):
        model.train(); epoch_loss = 0.0
        for b, (x,y,_,_) in enumerate(train_loader):
            x,y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        t_loss = epoch_loss / len(train_loader)
        v_loss, v_acc = validate(model, val_loader, criterion)
        train_losses.append(t_loss); val_losses.append(v_loss); val_accs.append(v_acc)
        print(f"Tversky Epoch {epoch+1}/{NUM_EPOCHS} Train: {t_loss:.4f} Val: {v_loss:.4f} Acc: {v_acc:.4f}")
    torch.save(model.state_dict(), SAVED_MODEL_PATH)
    plot_training_curves(train_losses, val_losses, val_accs)

@torch.no_grad()

def test_and_plot(model, loader, device, output_dir="test_results_t1ce_Tversky"):
    """
    Run inference on the test loader, save predicted mask arrays (.npy) named by the original image,
    and generate mid-slice visualizations.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    for b, (x, y, img_names, mask_names) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        vols = x.cpu().numpy()
        masks = y.cpu().numpy()
        for i in range(vols.shape[0]):
            vol = vols[i, 0]
            pred = preds[i]
            gt = masks[i]
            # Derive prediction filename from original image name
            orig_name = img_names[i]
            base, _ = os.path.splitext(orig_name)
            # Expect base like 'image_6'; extract the numeric suffix
            suffix = base.split('_')[-1]
            pred_filename = f"prediction_{suffix}.npy"
            pred_path = os.path.join(output_dir, pred_filename)
            # Save predicted mask
            np.save(pred_path, pred)

            # Prepare mid-slices
            D, H, W = vol.shape
            mids = {'axial': D//2, 'coronal': H//2, 'sagittal': W//2}
            slices_vol  = {k: vol.take(v, axis=idx) for idx, (k, v) in enumerate(mids.items())}
            slices_pred = {k: pred.take(v, axis=idx) for idx, (k, v) in enumerate(mids.items())}
            slices_gt   = {k: gt.take(v, axis=idx)   for idx, (k, v) in enumerate(mids.items())}

            # Plot and save figure
            fig, axs = plt.subplots(3, 3, figsize=(9, 9), dpi=150)
            for r, axis in enumerate(['axial', 'coronal', 'sagittal']):
                axs[r, 0].imshow(slices_vol[axis], interpolation='nearest')
                axs[r, 0].set_title(f"Orig {axis}"); axs[r, 0].axis('off')
                axs[r, 1].imshow(slices_pred[axis], interpolation='nearest')
                axs[r, 1].set_title(f"Pred {axis}"); axs[r, 1].axis('off')
                axs[r, 2].imshow(slices_gt[axis], interpolation='nearest')
                axs[r, 2].set_title(f"GT {axis}"); axs[r, 2].axis('off')
            plt.tight_layout()
            fig_filename = f"{base}_prediction_{suffix}_axes-tversky0.8.png"
            fig_path = os.path.join(output_dir, fig_filename)
            fig.savefig(fig_path)
            plt.close(fig)
    print(f"Saved test results and .npy masks in {output_dir}")

# ──────────────────────────────────────────────────────────────
# Entry point: prepare loaders, model, run train/test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_imgs=[f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith('.npy')]
    train_masks=[f for f in os.listdir(TRAIN_MASK_DIR) if f.endswith('.npy')]
    val_imgs=[f for f in os.listdir(VAL_IMG_DIR) if f.endswith('.npy')]
    val_masks=[f for f in os.listdir(VAL_MASK_DIR) if f.endswith('.npy')]
    train_ds=NpyT1cImageDataset(TRAIN_IMG_DIR,train_imgs,TRAIN_MASK_DIR,train_masks)
    val_ds=NpyT1cImageDataset(VAL_IMG_DIR,val_imgs,VAL_MASK_DIR,val_masks)
    train_ld=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,pin_memory=True)
    val_ld=DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=4,pin_memory=True) 
    ce = weighted_ce_loss()
    dice = dice_loss()
    alpha =0.0
    combined_loss = lambda logits, y: alpha*ce(logits, y)+(1-alpha)*dice(logits,y) 
    tv = tversky_loss(alpha=0.8)
    ce_tv_loss = lambda logits, y: 0.5*ce(logits,y.long()) + 0.5*tv(logits,y.long())
    model=build_unetr(); criterion=ce_tv_loss
    #if os.path.exists(SAVED_MODEL_PATH):
    #   model.load_state_dict(torch.load(SAVED_MODEL_PATH,map_location=DEVICE))
    #   print("Loaded existing model, skipping training.")
    #else:
    train(model,train_ld,val_ld,criterion)

    test_imgs=[f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.npy')]
    test_masks=[f for f in os.listdir(TEST_MASK_DIR) if f.endswith('.npy')]
    test_ds=NpyT1cImageDataset(TEST_IMG_DIR,test_imgs,TEST_MASK_DIR,test_masks)
    test_ld=DataLoader(test_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=4,pin_memory=True)
    test_and_plot(model,test_ld,DEVICE)

