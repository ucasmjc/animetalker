from PIL import Image

# 输入和输出路径
input_path = "/mnt/data/mjc/Index-anisora/anisoraV2_gpu/image.png"
output_path = "/mnt/data/mjc/Index-anisora/anisoraV2_gpu/image_resized_256x256.png"

# 打开图片并调整尺寸
try:
    # 读取原始图片
    img = Image.open(input_path)
    
    # 使用LANCZOS算法（高质量下采样）调整尺寸
    img_resized = img.resize((256, 256), Image.BILINEAR)
    
    # 保存结果（保留原始格式）
    img_resized.save(output_path)
    print(f"图片已成功调整为256x256，保存至: {output_path}")

except FileNotFoundError:
    print(f"错误：文件不存在 - {input_path}")
except Exception as e:
    print(f"处理过程中发生错误: {str(e)}")