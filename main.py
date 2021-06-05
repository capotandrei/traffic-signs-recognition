import warnings

warnings.filterwarnings('ignore')

import numpy as np
import cv2
from tensorflow import keras
import json

from utils import get_detection_model

DETECTION_PATH = './detection/'
CLASSIFICATION_PATH = './classification/'

IMG_HEIGHT = 30
IMG_WIDTH = 30

f = open(CLASSIFICATION_PATH + 'labels_map.txt', 'r')
classes = {int(key): value for key, value in json.loads(f.read()).items()}

tfnet = get_detection_model(model_path=DETECTION_PATH, model_name='yolo_v2')
class_model = keras.models.load_model(CLASSIFICATION_PATH + 'trained_model/classification_model.h5')


def main(min_score_thresh=0.5, pixel_offset=10):
    video_file = 'test1.mp4'
    vidcap = cv2.VideoCapture(video_file)

    vidcap.set(cv2.CAP_PROP_FPS, 44100)

    frame_count = 0

    while True:
        ret, frame = vidcap.read()
        frame_count += 1
        if frame_count % 5 == 0:
            frame_count = 0

            # Sign detection
            pred_results = tfnet.return_predict(frame)

            processed_frame = frame

            for idx, det in enumerate(pred_results):
                score = det['confidence']
                if score > min_score_thresh:
                    processed_frame = cv2.rectangle(frame,
                                                    (det['topleft']['x'], det['topleft']['y']),
                                                    (det['bottomright']['x'], det['bottomright']['y']),
                                                    color=(255, 0, 0),
                                                    thickness=2
                                                    )

                    # Sign classification
                    try:
                        detected_sign = frame[det['topleft']['y'] - pixel_offset:det['bottomright']['y'] + pixel_offset,
                                              det['topleft']['x'] - pixel_offset:det['bottomright']['x'] + pixel_offset]

                        data = []
                        resize_image = cv2.resize(detected_sign, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
                        data.append(resize_image)

                        X_test = np.array(data)
                        X_test = X_test / 255

                        pred = class_model.predict_classes(X_test)
                        scaled_score = str(score * 100)
                        sign_class = f"{classes[pred[0]]} ({scaled_score[0:2]}%)"

                        cv2.putText(processed_frame, sign_class, (det['topleft']['x'], det['topleft']['y'] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    except Exception as e:
                        print('Error: ', str(e))

            cv2.imshow('frame', processed_frame)

            if cv2.waitKey(1) == ord('q') or frame is None:
                break

    vidcap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
