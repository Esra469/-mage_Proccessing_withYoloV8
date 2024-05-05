#poligon zon bölgesi tanımla ve sadece o bölgedeki araçları algılayıp göster
#SOURCE=np.array([])->nasıl bir alan oluşturduğumuzu görürüz
#polygon_zone
#detection
#annotated_frame

import supervision as sv
import cv2
from ultralytics import YOLO
import numpy as np

SOURCE=np.array([[int(1252*0.3),int(787*0.3)],[int(2298*0.3),int(803*0.3)],
                 [int(5039*0.3),int(2159*0.3)],[int(-550*0.3),int(2159*0.3)]])

if __name__=="__main__":
     video_info=sv.VideoInfo.from_video_path(video_path="vehicle.mp4")
     model=YOLO("yolov8n.pt")

     byte_track=sv.ByteTrack(frame_rate=video_info.fps,track_thresh=0.5)

     thickness=3
     text_scale=1

     bounding_box_annotator=sv.BoundingBoxAnnotator(thickness=thickness)
     label_annotator=sv.labelAnnotator(
          text_scale=text_scale,
          text_thickness=thickness,
          text_position=sv.Position.BOTTOM_CENTER
     )
     frame_generator=sv.get_video_frames_generator(source_path="vehicle.mp4")
     
     polygoon_zone=sv.PolygoneZone(polygon=SOURCE,frame_resolution_wh=video_info.resolution_wh)

     for frame in frame_generator:
          half_frame=cv2.resize(frame,(0,0),fx=0.3,fy=0.3)
          result=model(half_frame)[0]
          detections=sv.Detections.from_ultralystics(result)
          detections=detections[polygoon_zone.trigger(detections)]
          detections=byte_track.update_with_detections(detections=detections)

          labels=[]
          for tracker_id in np.array(detections.tracker.id):
               labels.append(f"#(tracker_id)")
          annotated_frame=half_frame.copy()
          annotated_frame=sv.draw_polygon(annotated_frame,polygon=SOURCE,color=sv.Color.red())
          annotated_frame=bounding_box_annotator.annotate(
               scene=annotated_frame,detections=detections
          )
          annotated_frame=label_annotator.annotate(
               scene=annotated_frame,detections=detections,labels=labels
          )

          cv2.imshow("frame",annotated_frame)
          if cv2.waitKey(1) & 0xFF==ord("q"):
               break
          cv2.destroyAllWindows()
               
