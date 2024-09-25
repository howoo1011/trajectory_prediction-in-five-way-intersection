import json
import cv2
from ultralytics import YOLO
import numpy as np
import os
import sys

# Load the ROI data from JSON
with open('C:/Drone/ultralytics/roi/roi.json', 'r') as file:
    roi_data = json.load(file)

def create_masks(frame_size, rois):
    masks = []
    for roi in rois:
        mask = np.zeros(frame_size, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(roi, dtype=np.int32)], 255)
        masks.append(mask)
    return masks

def draw_lanes(frame, rois):
    for i, roi in enumerate(rois):
        for j in range(len(roi)):
            next_j = (j + 1) % len(roi)
            cv2.line(frame, roi[j], roi[next_j], (0, 0, 255), 2)

def process_frame(frame, model, timestamp, previous_positions, lane_changes, output_directory, rois):
    result = model.track(frame, conf=0.5, persist=True, save_txt=True, tracker='bytetrack.yaml', line_width = 1, show=True)
    if len(result[0].boxes.xywh) > 0:
        for xywh, cls, confidence, obj_id in zip(result[0].boxes.xywh, result[0].boxes.cls, result[0].boxes.conf, result[0].boxes.id):
            center_x, center_y, width, height = [float(coord) for coord in xywh]
            x1, y1 = float(center_x - width / 2), float(center_y - height / 2)
            x2, y2 = float(center_x + width / 2), float(center_y + height / 2)
            # Determine which lane the object is in
            current_lane = None
            for i, roi in enumerate(rois):
                if cv2.pointPolygonTest(np.array(roi, dtype=np.int32), (center_x, center_y), False) >= 0:
                    current_lane = f"lane_{i+1}"
                    break
            if current_lane:
                # Check for lane change
                if obj_id in previous_positions and previous_positions[obj_id] != current_lane:
                    lane_changes.append((timestamp, obj_id, previous_positions[obj_id], current_lane))
                    print(f"Lane change detected for ID: {obj_id} from {previous_positions[obj_id]} to {current_lane} at timestamp: {timestamp}")
                previous_positions[obj_id] = current_lane
                # Save the output in lane specific file
                with open(f"{output_directory}/output_{current_lane}.txt", "a") as file:
                    file.write(f"{timestamp:.2f}, {int(obj_id)}, {result[0].names[int(cls)]}, {x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}, {center_x:.2f}, {center_y:.2f}, {x1:.2f}, {center_y:.2f}, {x2:.2f}, {center_y:.2f}, {width:.2f}, {height:.2f}\n")
    draw_lanes(frame, rois)

def run_yolo_tracking(model_path, source_video, rois, save=True, conf=0.5, show=True, line_width=3, output_video_path='output.avi'):
    current_directory = os.getcwd()
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source_video)
    desired_width = 1920
    desired_height = 1080
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (desired_width, desired_height))
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Unable to read video source")
        sys.exit(1)
    first_frame = cv2.resize(first_frame, (desired_width, desired_height))
    frame_number = 0
    previous_positions = {}
    lane_changes = []
    # Create files for output
    for i in range(len(rois)):
        open(f"{current_directory}/output_lane_{i+1}.txt", "w").close()
    open(f"{current_directory}/lane_changes.txt", "w").close()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (desired_width, desired_height))
        timestamp = frame_number / fps
        process_frame(frame, model, timestamp, previous_positions, lane_changes, current_directory, rois)
        out.write(frame)
        if show:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_number += 1
    # Output lane changes
    print("Lane changes:")
    with open(f"{current_directory}/lane_changes.txt", "w") as file:
        for change in lane_changes:
            print(change)
            file.write(f"Timestamp: {change[0]:.2f}, ID: {change[1]}, From: {change[2]}, To: {change[3]}\n")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "C:/Users/user/Desktop/drone_vision_local/best.pt"
    video_number = '2'  
    formatted_video_number = str(int(video_number)).zfill(3)  
    source_video = f"C:/Drone/ultralytics/clip/SiteC_{formatted_video_number}.Mp4"
    roi_key = str(int(video_number))  
    rois = roi_data.get(roi_key, [])
    run_yolo_tracking(model_path, source_video, rois, output_video_path='output.avi')

