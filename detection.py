import math
import numpy as np
from ultralytics import YOLO
import cv2


def process_hand_frame(hand):
    hand = cv2.resize(hand, (640, 640))
    # increase the contrast of the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hand = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
    hand = hand.astype(np.float32)
    hand = hand / 255.0
    hand = np.array(hand).reshape((-1, 640, 640, 1))

    return hand


def compute_iou(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2, _ = box1
    box2_x1, box2_y1, box2_x2, box2_y2, _ = box2

    dx = min(box1_x2, box2_x2) - max(box1_x1, box2_x1) + 1
    dy = min(box1_y2, box2_y2) - max(box1_y1, box2_y1) + 1

    intersection_area = max(0, dx * dy)

    box1_area = (box1_x2 - box1_x1 + 1) * (box1_y2 - box1_y1 + 1)
    box2_area = (box2_x2 - box2_x1 + 1) * (box2_y2 - box2_y1 + 1)

    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area


def nms(boxes, overlap_threshold=0.5):
    if len(boxes) == 0:
        return []

    boxes.sort(key=lambda x: x[4], reverse=True)
    selected_boxes = []

    while len(boxes) > 0:
        best_box = boxes[0]
        selected_boxes.append(best_box)

        remaining_boxes = boxes[1:]
        iou_scores = [compute_iou(best_box, remaining_box) for remaining_box in remaining_boxes]

        boxes = [box for i, box in enumerate(remaining_boxes) if iou_scores[i] < overlap_threshold]

    return selected_boxes


class Detection:
    model = None
    confidence_threshold = 0.293

    def load_detector(self):
        self.model = YOLO("./models/detection_model.pt")

    def detect(self, image):
        return self.model(image, verbose=False)

    def detect_hands(self, camera_frame):
        original_frame = camera_frame
        results = self.detect(camera_frame)
        class_names = ["Human-hand", "Human-hands"]

        hands = []
        for r in results:
            boxes = r.boxes

            for box in boxes:
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = class_names[cls]

                if confidence > self.confidence_threshold and class_name in class_names:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    hands.append([x1, y1, x2, y2, confidence])

        nms_hands = nms(hands, 0.2)
        hands = []

        for hand in nms_hands:
            x1, y1, x2, y2, confidence = hand
            hand_frame = original_frame[y1:y2, x1:x2]
            hand_frame = process_hand_frame(hand_frame)
            hands.append([hand_frame, [x1, y1, x2, y2]])

        return hands
