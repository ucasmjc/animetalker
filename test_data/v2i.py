import cv2
import os

# 视频文件路径
video_folder = '/mnt/data/mjc/test_data/mp4s'
# 保存帧的目录
output_folder = '/mnt/data/mjc/test_data/frames'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历视频文件夹中的所有mp4文件
for video_file in [f for f in os.listdir(video_folder) if f.endswith('.mp4')]:
    video_path = os.path.join(video_folder, video_file)
    
    # 获取视频文件名（不带扩展名）
    video_name = os.path.splitext(video_file)[0]
    
    # 创建对应的子文件夹
    video_output_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 保存帧为图像文件
        frame_filename = os.path.join(video_output_folder, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames from {video_file}")