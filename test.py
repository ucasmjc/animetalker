import numpy as np
from PIL import Image, ImageDraw

# 创建512×512的黑色背景图像
image_size = (512, 512)
background = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
image = Image.fromarray(background, mode='L')

# 假设给定的坐标是边界框的左上角和右下角坐标: [x1, y1, x2, y2]
# 这里将输入的数组转换为边界框坐标
# 原始数组: [[[114, 7], [137, 243]]]
# 转换为边界框坐标: (114, 7, 137, 243)
bbox = (114, 7, 114+137, 7+243)

# 确保边界框坐标在图像范围内
x1, y1, x2, y2 = bbox
x1 = max(0, min(x1, image_size[0]))
y1 = max(0, min(y1, image_size[1]))
x2 = max(0, min(x2, image_size[0]))
y2 = max(0, min(y2, image_size[1]))

# 创建绘图对象
draw = ImageDraw.Draw(image)

# 绘制白色边界框（线宽为2像素）
draw.rectangle([x1, y1, x2, y2], outline=255, width=2)

# 保存图像
output_path = 'bbox_visualization.png'
image.save(output_path)
print(f"边界框已可视化到512×512图像，保存为 {output_path}")

# 尝试显示图像
try:
    image.show()
except Exception as e:
    print(f"无法显示图像: {e}。请手动打开 {output_path} 查看结果。")