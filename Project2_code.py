import cv2
import numpy as np
import time
from pymycobot.mycobot import MyCobot
from ultralytics import YOLO

# MyCobot 연결
mc = MyCobot('COM3', 115200)
mc.send_coords([219.3, 17.5, 339, -176.39, -2.67, -83.56], 40)
time.sleep(3)

mc.set_gripper_mode(0)
mc.init_eletric_gripper()

# 카메라 보정 매트릭스와 왜곡 계수 로드
camera_matrix = np.load('Image/Image/camera_matrix.npy')
dist_coeffs = np.load('Image/Image/dist_coeffs.npy')

# 비디오 캡처 시작
cap = cv2.VideoCapture(1)
model = YOLO(r'best.pt')

# 색상 범위 정의 (HSV)
color_ranges = {
    'red': [(np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([160, 100, 100]), np.array([180, 255, 255]))],
    'yellow': (np.array([20, 100, 100]), np.array([35, 255, 255])),
    'purple': (np.array([130, 100, 100]), np.array([160, 255, 255])),
    'blue': (np.array([100, 100, 100]), np.array([120, 255, 255])),
    'green': (np.array([35, 100, 100]), np.array([85, 255, 255]))
}

block_counts = {'red': 1, 'yellow': 0, 'purple': 0, 'blue': 0, 'green': 0}
layer_height = 25  # 각 층의 높이

def move_to_original_position(mc):
    print("로봇암 초기 위치로 이동")
    mc.send_coords([219.3, 17.5, 339, -176.39, -2.67, -83.56], 80)
    time.sleep(3)

def move_to_circle_position(mc):
    print("원형검출 위치로 이동")    
    mc.send_coord(3, 330, 80)
    time.sleep(1)
    mc.send_coords([70.4, -187, 304.9, 179.97, -4.16, -84.44], 80) 
    time.sleep(2)
    
def stack_block(mc, color, layer):
    # 쌓을 층 계산 (각 색깔별로 1층부터 시작)
    if layer==0:
        z_position = 170
    else:
        z_position = 169 + layer * layer_height
    print(f"{color} 블록 {layer+1}층 쌓기, {z_position}")
    mc.send_coord(3, z_position, 60)
    time.sleep(2)
  
    mc.set_eletric_gripper(0)
    mc.set_gripper_value(50, 60, 1)
    time.sleep(1)
    
    mc.send_coord(3, 250, 80)
    time.sleep(1)

    mc.set_eletric_gripper(0)
    mc.set_gripper_value(0,80,1)
    time.sleep(1)

    # 해당 색깔 블록 카운트 증가
    block_counts[color] += 1
    
def calculate_block_to_robotarmCoords(cx, cy):
    # 오프셋 보정
    cx_offset = cx + 190  # x좌표 보정
    cy_offset = cy + 80    # y좌표 보정
    
    x = round(-0.39 * cx_offset + -0.03 * cy_offset + 338, 1)
    y = round(-0.04 * cx_offset + 0.44 * cy_offset + -173, 1)
    z = 280
    return x, y, z

def calculate_rectangle_to_robotarmCoords(cx, cy):
    x = round(-0.4564 * cx + -0.0164 * cy + 218.6309, 1)
    y = round(0.0438 * cx + 0.3773 * cy + -331.9630, 1)
    z = round(0.0157 * cx + -0.0216 * cy + 216.5304, 1)
    rx = round(-0.0060 * cx + 0.0057 * cy + 174.6645, 1)
    ry = round(0.0021 * cx + -0.0057 * cy + -4.8904, 1)
    rz = round(0.0097 * cx + -0.0008 * cy + -86.1346, 1)
    return [x, y, z, rx, ry, rz]
    
def calculate_circle_to_robotarmCoords(cx, cy):
    # 오프셋 보정
    # cx = cx + 80  # x좌표 보정
    # cy = cy + 60    # y좌표 보정
    
    x = round(-0.4575 * cx + -0.0529 * cy + 231.7269 + 14, 1)
    y = round(-0.0438 * cx + 0.3818 * cy + -308.1350 -27, 1)
    z = round(0.0140 * cx + 0.0237 * cy + 232.6226, 1)
    rx = round(-0.0180 * cx + 0.0109 * cy + 177.3429, 1)
    ry = round(0.0003 * cx + 0.0072 * cy + -7.4247, 1)
    rz = round(-0.0025 * cx + 0.0276 * cy + -90.7202, 1)
    return [x, y, z, rx, ry, rz]

def gripper_angle_adjust(mc, angle):
    if angle < -45:
        angle += 90  # 각도가 -45도 이하일 때 보정
    
    # 0도에서 45도 사이로 각도 제한
    if 45 < angle <= 90:
        angle = angle - 90    #블록이 왼쪽으로 회전하는 게 (-)방향, 오른쪽이 (+)

    if -10 < angle < 10:
        angle = 0
        
    print(f"블록 각도: {angle:.2f}도")
    # # 그리퍼 각도 이동
    current_angles = mc.get_angles()  # 모든 관절의 각도 가져오기
    current_j6_angle = current_angles[5]  # J6 관절의 각도 (리스트의 6번째 요소)
    # 목표 J6 각도 계산
    target_j6_angle = current_j6_angle + angle -90
    # myCobot의 J6 각도를 설정합니다.
    mc.send_angle(6, target_j6_angle, 80)
    time.sleep(1) 

def draw_circle(circles, frame):
    # 원이 검출되었을 때 원 그리기
    for circle in circles:
        center = (circle[0], circle[1])  # 원의 중심 좌표
        radius = circle[2]  # 원의 반지름

        # 원의 중심에 작은 원 그리기
        cv2.circle(frame, center, 1, (0, 100, 100), 3)
        # 원의 윤곽선 그리기
        cv2.circle(frame, center, radius, (255, 0, 255), 3)

def grab_block(mc,x,y,z):
    mc.send_coord(1, x, 80)
    time.sleep(1)
    mc.send_coord(2, y, 80)
    time.sleep(1)
    mc.set_eletric_gripper(0)
    mc.set_gripper_value(70, 80, 1)
    time.sleep(1)

    mc.send_coord(3, z, 80)
    time.sleep(1)

    print("그리퍼_블록 잡기")
    mc.set_eletric_gripper(0)
    mc.set_gripper_value(25, 80, 1)
    time.sleep(1)
    
def throw_away_nonpass(mc):
    mc.send_coord(3, 325, 80)
    time.sleep(1)

    # 버리는 위치로 이동
    mc.send_coords([54.9, 211.6, 315.1, 175.26, 4.64, 68.63], 80)
    time.sleep(2)

    mc.set_eletric_gripper(0)
    mc.set_gripper_value(50, 80, 1)
    time.sleep(1)

    print("그리퍼_블록 잡기")
    mc.set_eletric_gripper(0)
    mc.set_gripper_value(0, 80, 1)
    time.sleep(1)

def yolo_detect(frame, results):
    class_name = ""  # 기본값 설정
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = box.conf[0]
            xyxy = box.xyxy[0].cpu().numpy()

            class_name = class_names[cls_id]
            color = class_colors.get(class_name, (255, 255, 255))

            x1, y1, x2, y2 = [int(coord) for coord in xyxy]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{class_name} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return class_name
            
def detect_color(hsv):
    # HSV 이미지에서 색상을 탐지하는 함수
    detected_color = None
    mask = None
    for color, range_values in color_ranges.items():
        if color == 'red':
            mask1 = cv2.inRange(hsv, range_values[0][0], range_values[0][1])
            mask2 = cv2.inRange(hsv, range_values[1][0], range_values[1][1])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, range_values[0], range_values[1])

        if cv2.countNonZero(mask) > 500:
            detected_color = color
            break
    return detected_color, mask

def detect_center_color(hsv, x, y):
    detected_color = None
    for color, ranges in color_ranges.items():
        if color == 'red':
            mask1 = cv2.inRange(hsv, ranges[0][0], ranges[0][1])
            mask2 = cv2.inRange(hsv, ranges[1][0], ranges[1][1])
            mask = cv2.bitwise_or(mask1, mask2)
        # elif color == 'yellow':
        #     mask = cv2.inRange(hsv, ranges[0], ranges[1])
        #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # 모폴로지 연산
            
        else:
            mask = cv2.inRange(hsv, ranges[0], ranges[1])

        if mask[y, x] > 0:
            detected_color = color
            break
    return detected_color

def search_block_rectangle_coord(mc, contour, frame):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 블록 중심점 계산
    cx, cy = int(rect[0][0]), int(rect[0][1])
    print(f"블록의 중심점: ({cx}, {cy})")

    angle = rect[2]
    gripper_angle_adjust(mc,angle)

    # 사각형 그리기 및 그리퍼 조작
    cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
    
    return x, y, z


def search_target_rectangle_coord(mc, target_color, layer):
    while True:
        ret, frame = cap.read()
        if ret:
            # BGR에서 HSV로 변환
            hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv = cv2.GaussianBlur(hsv2, (5, 5), 0)

            # target_color에 맞는 색상 필터링
            if target_color == "red":
                # 빨간색은 두 개의 범위로 나뉘어 있으므로, 두 개의 마스크를 생성하고 결합
                lower1, upper1 = color_ranges['red'][0]
                lower2, upper2 = color_ranges['red'][1]
                mask1 = cv2.inRange(hsv, lower1, upper1)
                mask2 = cv2.inRange(hsv, lower2, upper2)
                mask = cv2.bitwise_or(mask1, mask2)  # 두 마스크 결합
            else:
                # 다른 색상은 단일 범위만 사용
                lower, upper = color_ranges[target_color]
                mask = cv2.inRange(hsv, lower, upper)

            # 윤곽선 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            target_found = False  # target_color 사각형이 발견되었는지 확인
            
            if contours:
                for contour in contours:
                    if cv2.contourArea(contour) > 500:
                        rect = cv2.minAreaRect(contour)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)

                        # 블록 중심점 계산
                        cx, cy = int(rect[0][0]), int(rect[0][1])
                        print(f"블록의 중심점: ({cx}, {cy})")
                        
                        # 사각형 그리기 및 중심점 표시
                        cv2.drawContours(frame, [box], 0, (0, 255, 0), 3)
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                        cv2.putText(frame, f"Center: ({cx}, {cy})", (cx - 40, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        print(f"Target Rectangle Center: ({cx}, {cy})")
                        
                        cv2.imshow("circle", frame)
                        
                        print(calculate_rectangle_to_robotarmCoords(cx, cy))
                        mc.send_coords(calculate_rectangle_to_robotarmCoords(cx, cy),60)
                        time.sleep(3)
                        stack_block(mc, target_color, layer)
                        move_to_original_position(mc)
                        print("원위치 이동 완료")
                        
                        target_found = True  # target_color 사각형이 발견됨을 표시
                        break  # 사각형을 찾았으므로 반복 종료
                
                if target_found:  # 타겟 원을 찾았을 경우
                        break  # 원을 찾았으므로 외부 루프 종료

                        
            else:
                print("사각형 검출 실패, 다시 시도 중...")
            
            time.sleep(1)  # 재시도 간격 조정
 
        else:
            print("카메라에서 프레임을 읽지 못했습니다. 다시 시도 중...")
    cv2.imshow("circle", frame)




def search_target_circle_coord(mc, target_color, layer):
    while True:  # circle_color가 target_color와 일치할 때까지 반복
        ret3, frame3 = cap.read()
        if ret3:
            #frame3 = frame3[60:420, 80:500]
            hsv = cv2.cvtColor(frame3, cv2.COLOR_BGR2HSV)
            # hsv = cv2.GaussianBlur(hsv3, (5, 5), 0)

            # gray = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환
            blurred = cv2.GaussianBlur(frame3, (15, 15), 0)
            edges = cv2.Canny(blurred, 50, 150)

            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                                        param1=50, param2=30, minRadius=10, maxRadius=80)

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                draw_circle(circles, frame3)

                found_target_circle = False  # target_color와 일치하는 원을 찾았는지 여부

                for (x, y, r) in circles:
                    center_hsv = hsv[y, x]
                    circle_color = detect_center_color(hsv, x, y)
                    if circle_color == target_color:
                        # 오프셋 보정
                        #cx2, cy2 = x+80, y+60
                        cx2, cy2 = x, y
            
                        print(f"{circle_color} 원 검출 성공!")
                        print(f"    원의 중심 : ({cx2}, {cy2})")

                        mc.send_coords(calculate_circle_to_robotarmCoords(cx2, cy2),60)
                        time.sleep(2)
                        stack_block(mc, block_color, layer)
                        move_to_original_position(mc)
                        print("원위치 이동 완료")
                        found_target_circle = True  # target_color와 일치하는 원을 찾음
                        break
                if found_target_circle:  # 타겟 원을 찾았을 경우
                        break  # 원을 찾았으므로 외부 루프 종료

            else:
                print("원 검출 실패, 다시 시도 중...")

            time.sleep(1)  # 재시도 간격 조정
        else:
            print("카메라에서 프레임을 읽지 못했습니다. 다시 시도 중...")
    cv2.imshow("circle", frame3)




# 클래스 이름과 색상 매핑
class_colors = {'pass': (255, 0, 0), 'nonpass': (0, 0, 255)}
class_names = model.names

# 메인 루프
while True:
    # class_name 변수 초기화
    class_name = None

    ret, frame = cap.read()
    if not ret:
        break

    # ROI 적용
    frame = frame[80:480, 190:480]
    results = model(frame)

    # 객체 탐지 후 처리
    class_name = yolo_detect(frame, results)
    
    # 블록 상태가 'pass'일 경우
    if class_name == 'pass':
        time.sleep(2)
        ret2, frame2 = cap.read()
        if not ret2:
            break
        
        frame2 = frame2[80:480, 190:480]
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv2, (5, 5), 0)
        block_color, mask = detect_color(hsv)
        print(f"블록 상태: pass, 색상: {block_color}")
        
        target_color = block_color
        layer = block_counts[target_color]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            for contour in contours:
                if cv2.contourArea(contour) > 500:
                    x, y, z = search_block_rectangle_coord(mc, contour, frame2)
                    grab_block(mc,x,y,z)

                    # 블록 적재 위치 검출하는 곳으로 이동
                    move_to_circle_position(mc)
                    
                    # 블록이 쌓여있지 않으면 원형(스티커) 검출
                    if layer==0:
                        search_target_circle_coord(mc, target_color, layer)
                    
                    # 블록이 쌓여있으면 사각형(쌓여 있는 블록) 검출
                    else:
                        search_target_rectangle_coord(mc, target_color, layer)   
                        
        cv2.imshow("block", frame2)
        
        
    # 블록 상태가 'nonpass'일 경우
    elif class_name == 'nonpass':
        time.sleep(2)
        ret2, frame2 = cap.read()
        if not ret2:
            break
        
        frame2 = frame2[80:480, 190:480]
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv2, (5, 5), 0)
        block_color, mask = detect_color(hsv)
        print(f"불량품이 검출되었습니다.")

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            for contour in contours:
                if cv2.contourArea(contour) > 500:
                    x, y, z = search_block_rectangle_coord(mc, contour, frame2)

                    grab_block(mc,x,y,z)

                    throw_away_nonpass(mc)
                    
                    move_to_original_position(mc)
                    print("원위치 이동 완료")
    
    else : print("no detect")

    cv2.imshow("PASS_NONPASS", frame)
    # 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
