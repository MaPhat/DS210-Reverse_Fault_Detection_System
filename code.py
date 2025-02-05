import cv2
import torch
import math
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import numpy as np
from ultralytics import YOLO
import pandas as pd

# Tải mô hình YOLOv5
model = YOLO(r'final.pt')
# Khởi tạo tracker
deepsort = DeepSort(max_age=5)

# Khởi tạo vị trí trước đó
trajectories = {}

#Khởi tạo vector chiều
vector_track = {}

#Trạng thái hiện tại
object_status = {}

count = 0

# Biến lưu trạng thái vẽ zone
# zones = []
# drawing = False
# current_zone = []

# def draw_zone(event, x, y, flags, param):
#     global drawing, current_zone, zones

#     # Khi nhấn chuột trái, bắt đầu vẽ
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         current_zone = [(x, y)]

#     # Khi di chuyển chuột, vẽ đường viền
#     elif event == cv2.EVENT_MOUSEMOVE and drawing:
#         if len(current_zone) == 1:
#             current_zone.append((x, y))
#         else:
#             current_zone[1] = (x, y)

#     # Khi nhả chuột, hoàn thành vùng
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         if len(current_zone) == 2:
#             zones.append(tuple(current_zone))
#             current_zone = []
# Mở video
cap = cv2.VideoCapture(r'C:\Users\Phat Ma\Desktop\ds201\40s_fullhdkoche.asf')

# Tạo cửa sổ và gán callback để vẽ
# cv2.namedWindow("Draw Zones")
# cv2.setMouseCallback("Draw Zones", draw_zone)

# Giai đoạn vẽ zone
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Không thể tải khung hình.")
#         break

#     # Hiển thị vùng đã vẽ
#     for zone in zones:
#         cv2.rectangle(frame, zone[0], zone[1], (0, 255, 0), 2)

#     # Hiển thị vùng đang vẽ
#     if drawing and len(current_zone) == 2:
#         cv2.rectangle(frame, current_zone[0], current_zone[1], (255, 0, 0), 2)

#     cv2.imshow("Draw Zones", frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):  # Nhấn 'q' để thoát
#         break
#     elif key == ord('s') and zones:  # Nhấn 's' để lưu vùng và bắt đầu detect
#         print("Vùng đã vẽ:", zones)
#         break

# cv2.destroyWindow("Draw Zones")

# zone_center = {
#     'Thay_duoi_xe' : ((zones[0][1][0] - zones[0][0][0]) / 2, (zones[0][1][1] - zones[0][0][1]) / 2),
#     'Thay_dau_xe' : ((zones[1][1][0] - zones[1][0][0]) / 2, (zones[1][1][1] - zones[1][0][1]) / 2)
# }
# trungdiem_canhtren = {
#     'Thay_duoi_xe' : ((zones[0][1][0] - zones[0][0][0]) / 2, (zones[0][0][1] - zones[0][0][1]) / 2),
#     'Thay_dau_xe' : ((zones[1][1][0] - zones[1][0][0]) / 2, (zones[1][0][1] - zones[1][0][1]) / 2)
# }

# vector_chieu = {
#     'Thay_duoi_xe' : (trungdiem_canhtren['Thay_duoi_xe'][0] - zone_center['Thay_duoi_xe'][0], trungdiem_canhtren['Thay_duoi_xe'][1] - zone_center['Thay_duoi_xe'][1]),
#     'Thay_dau_xe' : (trungdiem_canhtren['Thay_dau_xe'][0] - zone_center['Thay_dau_xe'][0], trungdiem_canhtren['Thay_dau_xe'][1] - zone_center['Thay_dau_xe'][1])
# }

# Tính góc 2 vector
def calculate_angle(vector1, vector2):
    # Lấy thành phần của vector
    x1, y1 = vector1
    x2, y2 = vector2

    # Tính tích vô hướng
    dot_product = x1 * x2 + y1 * y2

    # Tính độ dài các vector
    magnitude1 = math.sqrt(x1**2 + y1**2)
    magnitude2 = math.sqrt(x2**2 + y2**2)

    # Tính cos(theta)
    if magnitude1 == 0 or magnitude2 == 0:
        cos_theta = dot_product
    else:
        cos_theta = dot_product / (magnitude1 * magnitude2)

    # Tính góc (rad)
    angle_rad = math.acos(max(-1, min(1, cos_theta)))  # Giới hạn giá trị trong [-1, 1]

    # Đổi góc từ radian sang độ (nếu cần)
    angle_deg = math.degrees(angle_rad)

    return angle_rad, angle_deg

# def check_vector_in_zone(vector : tuple, zone_name: str, A : tuple, B : tuple):
#     #print(vector)
#     if(zone_name == 'Thay_duoi_xe'):
#         location = zones[0]
#     else:
#         location = zones[1]
#     x_min, y_min = location[0]
#     x_max, y_max = location[1]

#   return x_min <= A[0] <= x_max and y_min <= A[1] <= y_max and x_min <= B[0] <= x_max and y_min <= B[1] <= y_max

def compute_distance(vector_x, vector_y):
    distance = math.sqrt(pow(vector_y[0] - vector_x[0], 2) + pow(vector_y[1] - vector_x[1], 2))

    return distance

def check_vector_nguocchieu(vector_x, vector_y):
    rad, deg = calculate_angle(vector_x, vector_y)

    if(deg >=100):
        return True
    else:
        return False

prev_time = time.time()
while cap.isOpened():

    #Lưu khoảng cách của từng vector với các vector khác
    distance_vector = {}

    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.resize(frame, (960, 540))
    # Chạy YOLOv5 để phát hiện đối tượng
    results = model(frame)

    # # Kiểm tra cấu trúc đầu ra
    # df = results.pandas().xyxy[0]

    # # Lọc ra các xe máy 
    # df_motorbike = df[df['class'] == 0]
    boxes = results[0].boxes  # Lấy các hộp dự đoán từ kết quả đầu ra

    #print(boxes)
    # Chuyển đổi kết quả thành DataFrame (nếu cần)
    df = pd.DataFrame(boxes.data.cpu().numpy(), columns=["xmin", "ymin", "xmax", "ymax","confidence", "class"])

    class_names = ['xe_may', 'o_to', 'nguoi_di_bo']  # Thay bằng danh sách lớp thực tế

    # Thêm cột 'name' bằng cách ánh xạ từ cột 'class'
    df['name'] = df['class'].apply(lambda x: class_names[int(x)])
    # Lọc ra các xe máy dựa trên lớp (class)
    df_motorbike = df[df['class'] == 0]
    detections = []


    # Lặp qua các dòng trong kết quả trả về từ yolo (kết quả được ép thành DataFrame)
    for _, row in df_motorbike.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        name = row['name']
        class_id = row['class']
        detections.append(([[x1, y1, x2 - x1, y2 - y1], confidence, class_id]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(frame, f"{name}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    # Dùng deepsort theo dấu đối tượng
    trackers = deepsort.update_tracks(detections, frame=frame)

    # Bỏ qua nếu track không được xác nhận
    for track in trackers:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        # Lấy bounding box và ID
        bbox = track.to_tlbr()
        track_id = track.track_id

        # Tính trung tâm của bounding box
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        # Lưu đường vết
        if track_id not in trajectories:
            trajectories[track_id] = []
        trajectories[track_id].append(((center_x), (center_y)))

        # Giới hạn độ dài đường vết
        if len(trajectories[track_id]) > 2:
            trajectories[track_id].pop(0)

        #Vector chiều di chuyển
        #print(trajectories[track_id])
        if(len(trajectories[track_id]) >= 2):
            begin = trajectories[track_id][0]
            end = trajectories[track_id][-1]

            vector_track[track_id] = (end[0]-begin[0], end[1]-begin[1])
        else:
            vector_track[track_id] = (0, 0)
        #print("Id {}, vector {}".format(track_id, vector_track[track_id]))

        #Kiểm tra góc chiều giữa 2 vector
        #Thấy đuôi xe nghĩa là bên phải
        # if(len(vector_track[track_id]) == 0):
        #     continue

        # begin_point = trajectories[track_id][0]
        # end_point = trajectories[track_id][-1]

        # if(check_vector_in_zone(vector_track[track_id], 'Thay_duoi_xe', begin_point, end_point)):
        #     #print('Thay duoi xe')
        #     angle_rad, angle_deg = calculate_angle(vector_track[track_id], vector_chieu['Thay_duoi_xe'])
        #     if(angle_deg > 91):
        #         #Cập nhật object_status đi ngược chiều
        #         object_status[track_id] = ['Di_nguoc_chieu']
        #     else:
        #         object_status[track_id] = ['Di_dung_chieu']
        # #Thấy đầu xe nghĩa là góc bên trái
        # else:
        #     angle_rad, angle_deg = calculate_angle(vector_track[track_id], vector_chieu['Thay_dau_xe'])
        #     if(angle_deg < 91):
        #         #Cập nhật object_status đi ngược chiều
        #         object_status[track_id] = ['Di_nguoc_chieu']
        #     else:
        #         object_status[track_id] = ['Di_dung_chieu']

        for id in vector_track.keys():
            #Quy định check 1 vector với n vector khác
            distance_vector[id] = {}
            n = 5
            id_list_check = [id_temp for id_temp in vector_track.keys() if id_temp != id]
            for id_check in id_list_check:
                distance = compute_distance(vector_track[id], vector_track[id_check])
                distance_vector[id][id_check] = distance

            distance_vector[id] = dict(sorted(distance_vector[id].items(), key=lambda item: item[1], reverse=False))


        #print(distance_vector)

        for id in distance_vector.keys():
            #kiem lai check distance_vector[id] khong chua it nhat 5 phan tu de check
            if(len(distance_vector[id].keys()) < 5):
                object_status[id] = ['NULL']
                continue
            list_vector_check = [id_check for id_check in list(distance_vector[id].keys())[:10]]

            #Đếm số vector ngược chiều so với vector id đang check
            count_True = 0
            count_False = 0
            for id_check in list_vector_check:
                if(check_vector_nguocchieu(vector_track[id], vector_track[id_check]) == True):
                    count_True += 1
                else:
                    count_False += 1
            if(count_True > count_False):
                object_status[id] = ['Di_nguoc_chieu']
            else:
                object_status[id] = ['Di_dung_chieu']

        # Vẽ đường vết
        # trajectory_points = trajectories[track_id]
        # if len(trajectory_points) > 5:
        #     cv2.polylines(frame, [np.array(trajectory_points, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=2)
        
        # Hiện status
        status = object_status[track_id][0]  # Lấy trạng thái hiện tại của object
        if status == 'Di_nguoc_chieu':
            color = (0, 0, 255)  # Màu đỏ (BGR)
        else:
            color = (0, 255, 0)  # Màu xanh lá (BGR)

        cv2.putText(frame, f"Status: {status}", 
                    (int(bbox[2]), int(bbox[3]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.39, 
                    color, 1)

    # tính fps = 1/thời gian xử lý
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Hiển thị video
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Detected Motorbikes', frame)
    data = {'ID': list(object_status.keys()), 'Status': [status[0] for status in object_status.values()]}
    df_status = pd.DataFrame(data)
    print(df_status)
    count = df_status['ID'].loc[df_status['Status'] == 'Di_nguoc_chieu'].count()
    print("="*8)
    print("So object di nguoc chieu: ", count)
    print("="*8)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()