import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np

# Constants
FACE_DETECTION_CONFIDENCE_THRESHOLD = 0.1355
VIDEO_FACE_DETECTION_CONFIDENCE_THRESHOLD = 0.2
RESIZE_DIMENSION = 300
MEAN_VALUES = (104.0, 177.0, 123.0)


def choose_file():
    """Open a file dialog to choose an image or video"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Choose an Image or Video",
        filetypes=[
            ("Image files", "*.jpg;*.jpeg;*.png"),
            ("Video files", "*.mp4;*.avi;*.mov"),
        ],
    )
    return file_path


def load_face_detection_model():
    """Load the pre-trained face detection model"""
    try:
        model_prototxt = "deploy.prototxt"
        model_weights = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        model = cv2.dnn.readNetFromCaffe(model_prototxt, model_weights)
        return model
    except IOError as e:
        print(f"Failed to load the model: {e}")
        return None


def detect_faces(model, image):
    """Find faces in the provided image"""
    (height, width) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (RESIZE_DIMENSION, RESIZE_DIMENSION)),
        1.0,
        (RESIZE_DIMENSION, RESIZE_DIMENSION),
        MEAN_VALUES,
    )
    model.setInput(blob)
    detections = model.forward()

    detected_faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > FACE_DETECTION_CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            detected_faces.append((start_x, start_y, end_x, end_y))

    return detected_faces


def draw_faces_on_image(image, faces):
    """Draw rectangles around detected faces in the image"""
    for start_x, start_y, end_x, end_y in faces:
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)


def process_image(image_path):
    """Detect faces in an image and show the result"""
    model = load_face_detection_model()
    if model is None:
        return

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load the image.")
        return

    faces = detect_faces(model, image)
    draw_faces_on_image(image, faces)

    output_file = "detected_faces.jpg"
    cv2.imwrite(output_file, image)

    cv2.namedWindow("Faces Detected", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Faces Detected", image.shape[1], image.shape[0])
    cv2.imshow("Faces Detected", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(video_path):
    """Detect faces in a video and save the result"""
    model = load_face_detection_model()
    if model is None:
        return

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Unable to open the video file.")
        return

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_file = "detected_faces_video.avi"
    video_writer = cv2.VideoWriter(
        output_file, fourcc, fps, (frame_width, frame_height)
    )

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        faces = detect_faces(model, frame)
        draw_faces_on_image(frame, faces)

        video_writer.write(frame)

    video_capture.release()
    video_writer.release()

    processed_video = cv2.VideoCapture(output_file)
    while processed_video.isOpened():
        ret, frame = processed_video.read()
        if not ret:
            break
        cv2.namedWindow("Faces Detected Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Faces Detected Video", frame.shape[1], frame.shape[0])
        cv2.imshow("Faces Detected Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    processed_video.release()
    cv2.destroyAllWindows()


def main():
    """Main function to handle user input and process files"""
    file_path = choose_file()

    if file_path:
        if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
            process_image(file_path)
        elif file_path.lower().endswith((".mp4", ".avi", ".mov")):
            process_video(file_path)
        else:
            print("Selected file type is not supported.")
    else:
        print("No file selected.")


if __name__ == "__main__":
    main()
