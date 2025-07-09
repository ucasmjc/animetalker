from PIL import Image
import torch
import numpy as np
import os
from wan.modules.vae import WanVAE
from diffusers.models import AutoencoderKLWan
# 加载VAE模型 (注意：原代码路径可能有误，已修正)
vae = AutoencoderKLWan.from_pretrained(
    "/mnt/data/checkpoints/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers/", 
    subfolder="vae",
    torch_dtype=torch.float32
).eval().to("cuda")  # 添加eval模式和GPU支持
device="cuda"
vae1 = WanVAE(vae_pth=os.path.join("/mnt/data/mjc/Index-anisora/", 'Wan2.1_VAE.pth'),device=device)
# 1. 加载并预处理图像
image = Image.open("/mnt/data/mjc/Index-anisora/anisoraV2_gpu/image.png").convert("RGB")

# 转换为tensor并添加需要的维度
image_tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255.0 * 2.0 - 1.0  # 归一化到[-1, 1]
image_tensor = image_tensor.permute(2, 0, 1)        # 从 (H, W, C) -> (C, H, W)
image_tensor = image_tensor.unsqueeze(0).unsqueeze(2)  # 添加批次和时间维度 [1, C, 1, H, W]
image_tensor = image_tensor.to("cuda")
tensorr = vae1.encode([
                torch.concat([
                    image_tensor[0],
                    torch.zeros(3, 81-1, 1024,1024).to("cuda")
                ], dim=1)
            ])[0].cpu()
import pdb
pdb.set_trace()
# image_tensor = image_tensor.to("cuda")
# import pdb
# pdb.set_trace() 
# 2. 编码图像到隐空间
z = vae1.encode(image_tensor)
with torch.no_grad():
    latent_dist = vae.encode(image_tensor)
    latent = latent_dist.latent_dist.sample()  # 得到 [1, 16, 1, 32, 32]

import pdb
pdb.set_trace()
    
# 3. 从隐空间重建图像
with torch.no_grad():
    reconstructed = vae.decode(latent).sample  # 得到 [1, 3, 1, 256, 256]

# 4. 后处理重建结果
reconstructed = reconstructed[0]                 # 移除批次维度 [3, 1, 256, 256]
reconstructed = reconstructed[:, 0]              # 移除时间维度 [3, 256, 256]
reconstructed = reconstructed.permute(1, 2, 0)   # [256, 256, 3]
reconstructed = ((reconstructed + 1.0) * 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

# 保存重建图像
Image.fromarray(reconstructed).save("reconstructed.png")