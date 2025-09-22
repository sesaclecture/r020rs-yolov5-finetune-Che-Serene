import torch
import cv2
import os
import json
import glob

# TODO: 커스텀 모델 로드
path = os.path.abspath('./nums_model.onnx')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)

# TODO: Label 로드
label_path = os.path.abspath('./nums_model.names.json')
with open(label_path, 'r') as f:
    label_names = json.load(f)

img_path = os.path.abspath('./data')
images = glob.glob(f'{img_path}/*.png')
print(f"총 {len(images)}개의 이미지를 찾았습니다.")

# TODO: 모델의 입력 크기
input_h, input_w = 640, 640

# 이미지 처리 루프
for idx, img_file in enumerate(images):
    print(f"\n처리 중: {os.path.basename(img_file)} ({idx+1}/{len(images)})")
    
    # 이미지 로드
    frame = cv2.imread(img_file)
    if frame is None:
        print(f"이미지 로드 실패: {img_file}")
        continue
    
    # 원본 이미지 크기
    frame_h, frame_w = frame.shape[:2]
    
    # 스케일 계산 (원본 크기 기준)
    scale_x = frame_w / input_w
    scale_y = frame_h / input_h
    
    # 추론 실행 (BGR -> RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # TODO: 추론 전 입력 크기 보정 (640x640)
    rgb_frame_resized = cv2.resize(rgb_frame, (640, 640))
    results = model(rgb_frame_resized)
    
    # Bounding box 그리기
    if len(results.xyxy[0]) > 0:
        for i, obj in enumerate(results.xyxy[0]):
            # 인식결과를 표시하기 위한 좌표를 얻음
            x1, y1, x2, y2, conf, cls = obj.tolist()
            cls = int(cls)
            
            # TODO: 인식된 정확도(confidence)와 클래스를 label로 구성
            label = f"{conf:.2f}, {label_names[str(cls)]}"
            
            # TODO: 출력 바운딩박스 크기 조절 (원본 이미지 크기로 변환)
            x1, x2 = map(lambda x: int(x * scale_x), (x1, x2))
            y1, y2 = map(lambda x: int(x * scale_y), (y1, y2))
            
            # OpenCV를 이용해서 해당 좌표에 사각형과 text를 출력
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"Object {i}: {label} at [{x1}, {y1}, {x2}, {y2}]")
    else:
        print("검출된 객체가 없습니다.")
    
    # 윈도우 이름에 파일명 포함
    window_name = f"YOLOv5 Detection - {os.path.basename(img_file)}"
    
    # 화면 크기에 맞춰 리사이즈 (선택적)
    display_frame = frame.copy()
    if frame_w > 1200 or frame_h > 800:  # 이미지가 너무 크면 리사이즈
        scale_display = min(1200/frame_w, 800/frame_h)
        new_w = int(frame_w * scale_display)
        new_h = int(frame_h * scale_display)
        display_frame = cv2.resize(display_frame, (new_w, new_h))
    
    # 화면 표시
    cv2.imshow(window_name, display_frame)
    
    # 키 입력 대기
    print(f"이미지 표시 중... (ESC: 종료, Space/Enter: 다음 이미지, S: 저장)")
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27:  # ESC - 전체 종료
            print("프로그램을 종료합니다.")
            cv2.destroyAllWindows()
            exit()
        elif key == 32 or key == 13:  # Space 또는 Enter - 다음 이미지
            cv2.destroyWindow(window_name)
            break
        elif key == ord('s') or key == ord('S'):  # S - 결과 저장
            save_path = f"result_{os.path.basename(img_file)}"
            cv2.imwrite(save_path, frame)
            print(f"결과 저장: {save_path}")

print("모든 이미지 처리 완료!")
cv2.destroyAllWindows()
