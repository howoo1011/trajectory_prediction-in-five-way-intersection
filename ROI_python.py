import cv2
import numpy as np

# 전역 변수 설정
roi_points = []  # 다각형 좌표를 저장할 리스트
drawing = False  # 드로잉 상태를 표시

# 마우스 콜백 함수
def draw_polygon(event, x, y, flags, param):
    global roi_points, drawing
    
    # 마우스 왼쪽 버튼을 누를 때
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # 좌표 저장
        roi_points.append((x, y))
        print(f"좌표 추가: {x, y}")
    
    # 마우스 오른쪽 버튼을 누를 때 (다각형을 닫고 그리기)
    elif event == cv2.EVENT_RBUTTONDOWN and len(roi_points) > 2:
        drawing = False
        # 다각형 그리기
        cv2.polylines(frame_resized, [np.array(roi_points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        print("다각형 완료")

# 비디오 파일 경로 설정 (웹캠 사용 시 0)
video_path = "C:\\Users\\a0109\\OneDrive\\바탕 화면\\trajectory_prediction\\좌회전\\edit20.mp4"  # 비디오 파일 경로
cap = cv2.VideoCapture(video_path)

# 첫 번째 프레임 읽기
ret, frame = cap.read()

if ret:
    # 원본 영상 크기 가져오기
    original_height, original_width = frame.shape[:2]
    
    resize_width = 1920
    resize_height = 1080

    # 프레임 크기를 조정
    frame_resized = cv2.resize(frame, (resize_width, resize_height))

    # 창 생성 및 마우스 콜백 설정
    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', draw_polygon)

    while True:
        # 프레임을 복사하여 마우스 클릭 이벤트마다 프레임 새로 그림
        display_frame = frame_resized.copy()
        
        # 다각형을 그리기 전에 선택된 좌표들을 연결하는 선을 그리기
        if len(roi_points) > 1:
            for i in range(1, len(roi_points)):
                cv2.line(display_frame, roi_points[i-1], roi_points[i], (255, 0, 0), 2)

        # 첫 번째 프레임 출력
        cv2.imshow('Frame', display_frame)

        # ESC를 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 선택한 다각형 좌표 출력
    print("선택된 다각형 좌표들:", roi_points)

    # 모든 창 닫기
    cv2.destroyAllWindows()
else:
    print("비디오를 불러올 수 없습니다.")

cap.release()
