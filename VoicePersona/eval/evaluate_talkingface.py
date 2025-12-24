import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import scipy.ndimage

# ========================
# 工具函数 & 数据集
# ========================

class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.image_paths = [
            os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


# ========================
# NIQE 计算
# ========================

def niqe(image, C=1, avg_window=None, extend_mode='constant'):
    # 确保图像是灰度图像
    if image.ndim == 3:  # 如果是RGB图像，将其转换为灰度图像
        image = np.mean(image, axis=-1)
    
    # 计算 MS-CN 变换
    def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
        if avg_window is None:
            avg_window = gen_gauss_window(3, 7.0/6.0)
        h, w = image.shape
        mu_image = np.zeros((h, w), dtype=np.float32)
        var_image = np.zeros((h, w), dtype=np.float32)
        scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
        scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
        scipy.ndimage.correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode)
        scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
        var_image = np.sqrt(np.abs(var_image - mu_image**2))
        return (image - mu_image) / (var_image + C)
    
    # 生成高斯窗口
    def gen_gauss_window(lw, sigma):
        sd = np.float32(sigma)
        lw = int(lw)
        weights = [0.0] * (2 * lw + 1)
        weights[lw] = 1.0
        sum_weights = 1.0
        sd *= sd
        for ii in range(1, lw + 1):
            tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
            weights[lw + ii] = tmp
            weights[lw - ii] = tmp
            sum_weights += 2.0 * tmp
        for ii in range(2 * lw + 1):
            weights[ii] /= sum_weights
        return weights

    # 应用 MS-CN 变换
    mscn = compute_image_mscn_transform(image)

    # 特征提取（简单版，实际应提取更多特征）
    mean = np.mean(mscn)
    std_dev = np.std(mscn)

    # 假设自然图像的均值和标准差（应通过大数据集训练得到）
    natural_mean = np.array([0.0, 0.0])
    natural_std = np.array([1.0, 1.0])

    # 计算欧几里得距离
    diff = np.array([mean, std_dev]) - natural_mean
    distance = np.linalg.norm(diff / natural_std)  # 标准化后的距离

    return distance


# ========================
# PSNR / SSIM / NIQE 计算
# ========================

def compute_psnr_ssim_niqe(real_folder, fake_folder):
    """逐对计算 PSNR / SSIM / NIQE（按文件名顺序配对）"""
    real_files = sorted([f for f in os.listdir(real_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    fake_files = sorted([f for f in os.listdir(fake_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    assert len(real_files) == len(fake_files), "Real and fake folders must have same number of images!"

    psnr_vals, ssim_vals, niqe_vals_fake = [], [], []

    for rf, ff in zip(real_files, fake_files):
        real_img = np.array(Image.open(os.path.join(real_folder, rf)).convert('RGB')).astype(np.float32)
        fake_img = np.array(Image.open(os.path.join(fake_folder, ff)).convert('RGB')).astype(np.float32)

        # PSNR & SSIM (0~255 scale)
        psnr_vals.append(psnr(real_img, fake_img, data_range=255))
        ssim_vals.append(ssim(real_img, fake_img, channel_axis=-1, data_range=255))

        # NIQE (only on fake images)
        niqe_vals_fake.append(niqe(fake_img))

    return {
        'PSNR': np.mean(psnr_vals),
        'SSIM': np.mean(ssim_vals),
        'NIQE': np.mean(niqe_vals_fake)
    }


# ========================
# FID 计算（Inception-v3）
# ========================

def get_inception_features(loader, device='cuda'):
    inception = models.inception_v3(pretrained=True, transform_input=False).to(device).eval()
    feats = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # Resize to 299x299 if needed
            if batch.shape[-1] != 299:
                batch = torch.nn.functional.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            pred = inception(batch)
            feats.append(pred.cpu().numpy())
    return np.concatenate(feats, axis=0)


def calculate_fid(real_feats, fake_feats):
    mu1, sigma1 = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# ========================
# LPIPS-like (LSE-C/D proxy)
# ========================

def compute_lpips(real_folder, fake_folder, device='cuda'):
    vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    real_files = sorted([f for f in os.listdir(real_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    fake_files = sorted([f for f in os.listdir(fake_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    lpips_vals = []
    with torch.no_grad():
        for rf, ff in zip(real_files, fake_files):
            real_img = transform(Image.open(os.path.join(real_folder, rf)).convert('RGB')).to(device)
            fake_img = transform(Image.open(os.path.join(fake_folder, ff)).convert('RGB')).to(device)
            feat_real = vgg(real_img.unsqueeze(0))
            feat_fake = vgg(fake_img.unsqueeze(0))
            loss = torch.nn.functional.mse_loss(feat_real, feat_fake).item()
            lpips_vals.append(loss)
    return np.mean(lpips_vals)
# ========================
# 主函数
# ========================

def evaluate_all_metrics(real_dir, fake_dir, batch_size=32, device='cuda'):
    print("Computing PSNR, SSIM, NIQE...")
    basic_metrics = compute_psnr_ssim_niqe(real_dir, fake_dir)

    print("Computing FID...")
    transform_fid = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    real_dataset = ImageFolderDataset(real_dir, transform=transform_fid)
    fake_dataset = ImageFolderDataset(fake_dir, transform=transform_fid)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    real_feats = get_inception_features(real_loader, device=device)
    fake_feats = get_inception_features(fake_loader, device=device)
    fid_score = calculate_fid(real_feats, fake_feats)

    print("Computing LPIPS (as LSE proxy)...")
    lpips_score = compute_lpips(real_dir, fake_dir, device=device)

    results = {
        **basic_metrics,
        'FID': fid_score,
        'LPIPS': lpips_score  # treat as LSE-C/LSE-D approximation
    }

    return results


# ========================
# 使用示例
# ========================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=str, required=True, help="Path to real images folder")
    parser.add_argument("--fake", type=str, required=True, help="Path to generated images folder")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    metrics = evaluate_all_metrics(args.real, args.fake, batch_size=args.batch_size, device=args.device)

    print("\n" + "="*40)
    print("Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("="*40)