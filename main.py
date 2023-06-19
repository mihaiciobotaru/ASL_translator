import cv2
from interpreter import Interpreter


class AslTranslator:
    interpreter = None

    def __init__(self):
        self.interpreter = Interpreter()

    def run(self):
        webcam = cv2.VideoCapture(0)
        while True:

            _, frame = webcam.read()
            self.interpreter.interpret_image(frame)
            cv2.imshow("American Sign Language Alphabet Translator", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        webcam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    asl_translator = AslTranslator()
    asl_translator.run()
