#tracking (araç takbi) için byte_track(supervision) izleyici aktifleştir,ve ekranda göster
#byte_track=sv.ByteTrack(frame_rate=video_info.fps)
#detection=byte_track.update_with_detections(detections=detections)

import supervision as sv
import cv2
from ultralytics import YOLO
import numpy as np

if __name__=="__main__":
    video_info=sv.VideoInfo.from_video_path(video_path="vehicle.mp4")
    model=YOLO("yolov8n.pt")

    byte_track=sv.ByteTrack(frame_rate=video_info.fps,track_thresh=0.5)
    thickness=3
    text_scale=1

    bounding_box_annotator=sv.BoundingBoxAnnotator(thickness=thickness)
    label_annotator=sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_positions=sv.Position.BOTTOM_CENTER)
    frame_generator=sv.get_video_frames_generator(source_path="vehicle.mp4")

    for frame in frame_generator:
        half_frame=cv2.resize(frame,(0,0),fx=0.3,fy=0.3)
        result=model(half_frame)[0]
        detections=sv.Detections.from_ultralystics(result)
        detections=byte_track.update_with_detections(detections=detections)

        labels=[]
        for tracker_id in np.array(detections.tracker_id):
            labels.append(f"#{tracker_id}")
        annotated_frame=half_frame.copy()
        annotated_frame=bounding_box_annotator.annotate(
            scene=annotated_frame,detections=detections)
        
        annotated_frame=label_annotator.annotate(
            scene=annotated_frame,detections=detections,labels=labels)
        
        #eksik bir şey var mı bak