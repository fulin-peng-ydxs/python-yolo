# OpenCV 库，用于视频处理
import cv2
# os 库，用于文件和目录操作
import os


# 抽取视频帧方法
def extract_frames(video_path, output_dir, frame_interval):
    # 配置参数
    video_path = video_path  # 视频文件路径
    output_dir = output_dir  # 输出帧保存目录
    frame_interval = frame_interval  # 抽帧间隔（每隔 frame_interval 帧保存一帧）

    # 创建输出目录: 用于递归创建目录。如果父目录不存在，它会先创建父目录，然后再创建子目录
    os.makedirs(output_dir, exist_ok=True)  # exist_ok=True 如果目录已经存在，不要抛出异常

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        exit()

    frame_count = 0  # 帧计数器
    saved_count = 0  # 保存帧计数器

    while True:
        ret, frame = cap.read()
        if not ret:  # 视频读取结束
            break

        # 判断是否为需要保存的帧
        if frame_count % frame_interval == 0:
            output_file = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(output_file, frame)
            saved_count += 1
            print(f"保存帧: {output_file}")

        frame_count += 1

    # 释放资源
    cap.release()
    print(f"抽帧完成，共保存了 {saved_count} 帧到 {output_dir}")


# 调用函数
extract_frames('./BVN.mp4', './output_frames', 30)
