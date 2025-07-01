# @Function  :可以从视频当中按照指定的间隔提取帧画面

import cv2
import os
import argparse


def extract_frame(video_path: str, save_path: str, interval: int = 1):
    """
    从视频中按照指定的间隔提取帧画面并保存为单张图像
    :param video_path: 视频的路径
    :param save_path: 存储路径
    :param interval: 提取间隔
    :return:
    """
    if not  os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频总帧数: {total_frames}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            save_file = os.path.join(save_path, f"frame_{saved_count:04d}.png")
            cv2.imwrite(save_file, frame)
            saved_count += 1
            print(f"已保存帧: {saved_count}")
        frame_count += 1
    cap.release()
    print(f"已保存总帧数: {saved_count},到{save_path}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="从视频中提取帧画面")
    parser.add_argument("--video_path", type=str, required=True, help="视频路径")
    parser.add_argument("--save_path", type=str, required=True, help="保存路径")
    parser.add_argument("--interval", type=int, default=1, help="提取间隔")
    args = parser.parse_args()
    extract_frame(args.video_path, args.save_path, args.interval)