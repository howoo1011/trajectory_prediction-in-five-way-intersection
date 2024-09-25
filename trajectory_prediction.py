import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys

def draw_custom_roi_on_first_frame(video_path, roi_coordinates, output_width=1100, output_height=620):
    # 영상에서 첫 번째 프레임 읽기
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Unable to read video at {video_path}")
        cap.release()
        return

    # 이미지 크기 조정 (output_width와 output_height에 따라 축소)
    original_height, original_width = frame.shape[:2]
    frame = cv2.resize(frame, (output_width, output_height))
    
    # 크기 비율 계산 (원본 대비 축소된 이미지 비율)
    width_ratio = output_width / original_width
    height_ratio = output_height / original_height

    # ROI를 프레임 위에 그리기
    for roi in roi_coordinates:
        points = np.array(roi, dtype=np.int32)
        # 각 좌표를 새로운 비율로 변환
        points = np.array([[int(x * width_ratio), int(y * height_ratio)] for x, y in points])
        cv2.polylines(frame, [points], isClosed=True, color=(0, 0, 255), thickness=2)  # 빨간색 선으로 ROI 그리기

    # 이미지 보여주기
    cv2.imshow("ROI Visualization", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 영상 해제
    cap.release()

def predict_turn_path(vehicle_position, rois):
    # rois[0]: 출발 갈래, rois[1]: 첫 번째 좌회전 경로, rois[2]: 두 번째 좌회전 경로
    if cv2.pointPolygonTest(np.array(rois[0], dtype=np.int32), vehicle_position, False) >= 0:
        # 차량이 출발 갈래에 있는 경우에만 예측 진행
        for idx, roi in enumerate(rois[1:], start=1):
            # 좌회전 경로 ROI 검사
            if cv2.pointPolygonTest(np.array(roi, dtype=np.int32), vehicle_position, False) >= 0:
                return f"Turn Path {idx}"  # 예: "Turn Path 1" 또는 "Turn Path 2"
    return None  # 예측 대상이 아닌 차량

def process_frame(frame, model, timestamp, previous_positions, output_directory, rois):
    result = model.track(frame, conf=0.5, persist=True, save_txt=True, tracker='bytetrack.yaml', line_width=1, show=True)
    if len(result[0].boxes.xywh) > 0:
        for xywh, cls, confidence, obj_id in zip(result[0].boxes.xywh, result[0].boxes.cls, result[0].boxes.conf, result[0].boxes.id):
            center_x, center_y, width, height = [float(coord) for coord in xywh]
            # 차량 중심 좌표
            vehicle_position = (center_x, center_y)

            # 좌회전 경로 예측
            predicted_path = predict_turn_path(vehicle_position, rois)
            if predicted_path:
                # Save the output for the turn path
                print(f"Vehicle {obj_id} predicted to take {predicted_path} at timestamp: {timestamp}")
                with open(f"{output_directory}/predicted_turn_paths.txt", "a") as file:
                    file.write(f"{timestamp:.2f}, ID: {obj_id}, {predicted_path}\n")
    
    draw_custom_roi_on_first_frame(frame, rois)

def run_yolo_tracking(model_path, source_video, roi_coordinates, output_video_path='output.avi'):
    current_directory = os.getcwd()
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    frame_number = 0
    previous_positions = {}

    # Create the predicted turn paths file
    open(f"{current_directory}/predicted_turn_paths.txt", "w").close()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_number / fps
        process_frame(frame, model, timestamp, previous_positions, current_directory, roi_coordinates)
        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "C:/Users/a0109/OneDrive/바탕 화면/Drone_data/best.pt"  # 모델 파일 경로
    source_video = 'C:/Users/a0109/OneDrive/바탕 화면/Drone_data/DJI_0002.MP4'  # 영상 파일 경로
    
    # 직접 입력한 ROI 좌표 설정
    roi_coordinates = [
        [[0, 1300], [0, 1400], [1660, 1400], [1660, 1300]],  # 첫 번째 다각형, 여기있는 차량만 디텍션
        [[1600, 1000], [1670, 1030], [1680, 880], [1630, 860]],  # 두 번째 다각형, 첫 번째 좌회전 경로
        [[1890, 900], [1890, 800], [2020, 800], [2020, 900]]  # 세 번째 다각형, 두 번째 좌회전 경로
    ]

    # Run YOLO tracking with the custom ROIs
    run_yolo_tracking(model_path, source_video, roi_coordinates, output_video_path='output.avi')
