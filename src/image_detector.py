import cv2
from keras.models import load_model
import numpy as np
import sys
import argparse
import dlib

from utils.datasets import get_labels
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from wide_resnet import WideResNet

def detector(image_input, image_output):
    # parameters for loading data and images
    detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
    emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    emotion_labels = get_labels('fer2013')
    gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
    gender_labels = get_labels('imdb')
    font = cv2.FONT_HERSHEY_SIMPLEX

    gender_to_cnt = {}
    emotion_to_cnt = {}
    age_to_cnt = {}

    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (20, 40)
    gender_offsets = (10, 10)

    # loading models
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    gender_classifier = load_model(gender_model_path, compile=False)
    gender_target_size = gender_classifier.input_shape[1:3]

    depth = 16
    k = 8
    weight_file = "weights.18-4.06.hdf5"

    # load model and weights
    gender_age_prediction_img_size = 64
    model = WideResNet(gender_age_prediction_img_size, depth=depth, k=k)()
    model.load_weights(weight_file)

    cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

    def pipeline(bgr_image):
        #bgr_image = cv2.resize(bgr_image, (640, 360))
        faces = cnn_face_detector(bgr_image, 1)
        global total_faces

        total_faces = total_faces + len(faces)
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        for face in faces:
            x1, y1, x2, y2, w, h = face.rect.left(), face.rect.top(), face.rect.right() + 1, face.rect.bottom() + 1, face.rect.width(), face.rect.height()
            xw1 = max(int(x1 - 0.4 * w), 0)
            yw1 = max(int(y1 - 0.4 * h), 0)
            xw2 = min(int(x2 + 0.4 * w), bgr_image.shape[1] - 1)
            yw2 = min(int(y2 + 0.4 * h), bgr_image.shape[0] - 1)
            gray_face = gray_image[yw1:yw2 + 1, xw1:xw2 + 1]

            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, False)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
            emotion_text = emotion_labels[emotion_label_arg]

            if emotion_text not in emotion_to_cnt:
                emotion_to_cnt[emotion_text] = 0
            emotion_to_cnt[emotion_text] = emotion_to_cnt[emotion_text] + 1

            color = (0, 0, 0)

            cv2.putText(rgb_image, emotion_text, (face.rect.left(), face.rect.top() - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 1, cv2.LINE_AA)

        face_list = np.empty((len(faces), gender_age_prediction_img_size, gender_age_prediction_img_size, 3))

        for i in range(0, len(faces)):
            face = faces[i]
            x1, y1, x2, y2, w, h = face.rect.left(), face.rect.top(), face.rect.right() + 1, face.rect.bottom() + 1, face.rect.width(), face.rect.height()
            xw1 = max(int(x1 - 0.4 * w), 0)
            yw1 = max(int(y1 - 0.4 * h), 0)
            xw2 = min(int(x2 + 0.4 * w), bgr_image.shape[1] - 1)
            yw2 = min(int(y2 + 0.4 * h), bgr_image.shape[0] - 1)
            rgb_face = rgb_image[yw1:yw2 + 1, xw1:xw2 + 1, :]

            try:
                face_list[i, :, :, :] = cv2.resize(rgb_face,
                                                   (gender_age_prediction_img_size, gender_age_prediction_img_size))
            except:
                continue

        gender_age_prediction = model.predict(face_list)
        for i in range(0, len(faces)):
            face = faces[i]
            predicted_genders = gender_age_prediction[0]
            gender_text = "FEMALE" if predicted_genders[i][0] > 0.5 else "MALE"

            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = gender_age_prediction[1].dot(ages).flatten()
            age_text = str(predicted_ages[i])

            if gender_text not in gender_to_cnt:
                gender_to_cnt[gender_text] = 0
            gender_to_cnt[gender_text] = gender_to_cnt[gender_text] + 1

            if age_text not in age_to_cnt:
                age_to_cnt[age_text] = 0
            age_to_cnt[age_text] = age_to_cnt[age_text] + 1

            gender_color = (255, 0, 0) if gender_text == "MALE" else (0, 0, 255)
            cv2.rectangle(rgb_image, (face.rect.left(), face.rect.top()), (face.rect.right(), face.rect.bottom()),
                          gender_color, 1)

            color = (0, 0, 0)
            cv2.putText(rgb_image, gender_text, (face.rect.left(), face.rect.top() - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, gender_color, 1, cv2.LINE_AA)
            cv2.putText(rgb_image, age_text, (face.rect.left(), face.rect.top() - 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 1, cv2.LINE_AA)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        return bgr_image

    image = cv2.imread(image_input)
    cv2.imwrite(image_output, pipeline(image))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_input")
    parser.add_argument("--video_output")
    # parser.add_help

    args = parser.parse_args()

    detector(args.image_input, args.image_output)

if __name__ == "__main__":
    main()




