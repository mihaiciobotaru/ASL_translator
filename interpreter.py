import cv2
from classifier import Classifier
from detection import Detection


def highlight_hand(hand_dimensions, main_frame):
    alpha = 0.3

    overlay = main_frame.copy()
    cv2.rectangle(overlay, (hand_dimensions[0], hand_dimensions[1]), (hand_dimensions[2], hand_dimensions[3]),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, main_frame, 1 - alpha, 0, main_frame)
    cv2.rectangle(main_frame, (hand_dimensions[0], hand_dimensions[1]), (hand_dimensions[2], hand_dimensions[3]),
                  (210, 210, 210), 2)


def write_probability_classes(probability_classes, hand_dimensions, main_frame):

    probabilities_rectangle_width = 0.50 * (hand_dimensions[2] - hand_dimensions[0])
    probabilities_rectangle_width = max(probabilities_rectangle_width, 60)
    probabilities_rectangle_width = min(probabilities_rectangle_width, 120)
    probabilities_rectangle_height = hand_dimensions[3] - hand_dimensions[1]
    probabilities_rectangle_x1 = int(hand_dimensions[0] - probabilities_rectangle_width)
    probabilities_rectangle_y1 = hand_dimensions[1]
    probabilities_rectangle_x2 = hand_dimensions[0]
    probabilities_rectangle_y2 = hand_dimensions[3]

    cv2.rectangle(main_frame, (probabilities_rectangle_x1, probabilities_rectangle_y1),
                  (probabilities_rectangle_x2, probabilities_rectangle_y2), (210, 210, 210), -1)
    cv2.rectangle(main_frame, (probabilities_rectangle_x1, probabilities_rectangle_y1),
                  (probabilities_rectangle_x2, probabilities_rectangle_y2), (210, 210, 210), 2)

    if probabilities_rectangle_height < 200:
        dy = (200 - probabilities_rectangle_height) // 20
        probability_classes = probability_classes[:-dy]

    for i in range(len(probability_classes)):
        class_name = probability_classes[i][0]
        if class_name == "Hand":
            class_name = "Other"
        class_probability = probability_classes[i][1]
        text = class_name + " " + str(class_probability) + "%"

        font_scale = 0.4
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX

        (text_width, text_height) = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        text_x = probabilities_rectangle_x1 + (probabilities_rectangle_width - text_width) * 0.1
        text_y = probabilities_rectangle_y1 + (i + 1) * ((probabilities_rectangle_height * 0.9) /
                                                         len(probability_classes))

        cv2.putText(main_frame, text, (int(text_x), int(text_y)), font, font_scale, (0, 0, 0), font_thickness)


class Interpreter:
    classifier = None
    detection = None
    hand_index = 0

    def __init__(self):
        self.classifier = Classifier()
        self.detection = Detection()

        self.classifier.load_classifier()
        self.detection.load_detector()

    def interpret_image(self, image):
        hands = self.detection.detect_hands(image)
        if hands is None or len(hands) == 0:
            return image

        hand_index = 0
        for hand in hands:
            self.interpret_gesture(hand, image, hand_index)
            hand_index += 1

    def interpret_gesture(self, hand, main_frame, hand_index):
        hand_frame = hand[0]
        hand_dimensions = hand[1]

        if hand_index < 3:
            probability_classes = self.classifier.classify(hand_frame)
        else:
            return None

        if probability_classes is None or probability_classes[0][0] == "None":
            return

        hand_index += 1
        highlight_hand(hand_dimensions, main_frame)
        if probability_classes[0][0] != "Hand":
            write_probability_classes(probability_classes, hand_dimensions, main_frame)

