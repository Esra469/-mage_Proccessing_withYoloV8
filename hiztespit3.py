#yolo class ID leri ekranda göster
#thickness=3
#text_scale=1

import supervision as sv
import cv2
from ultralytics import YOLO

if __name__=="__main__":
    video_info=sv.VideoInfo.from_video_path(video_path="vehicle.mp4")
    model=YOLO("yolov8n.pt")
    thickness=3
    text_scale=1

    bounding_box_annotator=sv.BoundingBoxAnnotator(thickness=thickness)
    label_annotator=sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTON_CENTER)
    
    frame_generator=sv.get_video_frame_generator(source_path="vehicle.mp4")

    for frame in frame_generator:
        #burası doldurulacak

