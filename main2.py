import cv2
import json
import os
import numpy as np
from ultralytics import YOLO
import cvzone
# Load YOLOv8 model
model = YOLO('best.pt')
names = model.names

cap = cv2.VideoCapture("vid1.mp4")
frame_count = 0

# Polygon drawing variables
polygon_points = []
polygons = []  # List of polygons (each is a list of 4 points)
polygon_file = "polygons.json"

# Load saved polygons safely
if os.path.exists(polygon_file):
    try:
        with open(polygon_file, 'r') as f:
            polygons = json.load(f)
    except (json.JSONDecodeError, ValueError):
        print("Warning: polygons.json is empty or corrupted. Resetting.")
        polygons = []
        with open(polygon_file, 'w') as f:
            json.dump(polygons, f)

# Save polygons to JSON
def save_polygons():
    with open(polygon_file, 'w') as f:
        json.dump(polygons, f)

# Mouse callback to add polygon points
def RGB(event, x, y, flags, param):
    global polygon_points, polygons
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        if len(polygon_points) == 4:
            polygons.append(polygon_points.copy())
            save_polygons()
            polygon_points.clear()

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    results = model.track(frame, persist=True)

    # Draw saved polygons
    for poly in polygons:
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Track how many zones are occupied
    occupied_zones = 0
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            for poly in polygons:
                pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
                    occupied_zones += 1
                    break

    total_zones = len(polygons)
    free_zones = total_zones - occupied_zones
    print("Free zones:", free_zones)
    cvzone.putTextRect(frame,f'FREEZONE:{free_zones}',(30,40),2,2)
    cvzone.putTextRect(frame,f'OCC:{occupied_zones}',(30,140),2,2)

    # Draw in-progress polygon points
    for pt in polygon_points:
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)

    cv2.imshow("RGB", frame)

    key = cv2.waitKey(0) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key == ord('r') and polygons:
        polygons.pop()
        save_polygons()

cap.release()
cv2.destroyAllWindows()
