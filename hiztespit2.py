#ekrana sığmayan resim boyutunu resize ile 0.3 oranında küçültme
#halj_frame=cv2.resize(frame,(0,0),fx=0.3,fy=0.3)

import supervision as sv
import cv2
from ultralytics import YOLO

if __name__=="__main__":
    video_info=sv.VideoInfo.from_video_path(video_path="vehicle.mp4")
    model=YOLO("yolov8n.pt")

    bounding_box_annotator=sv.BoundingBoxAnnotator(thickness=4)
    frame_generator=sv.get_video_frames_generator(source_path="vehicle.mp4")

    for frame in frame_generator:
        half_frame=cv2.resize(frame,(0,0),fx=0.3,fy=0.3)
        result=model(half_frame)[0]
        