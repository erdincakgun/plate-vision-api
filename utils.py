import cv2
import numpy as np
from ultralytics import YOLO
from easyocr import Reader
from base64 import b64encode

license_plate_detector = YOLO('license_plate_detector.pt')
reader = Reader(['en'])


def convert(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not convert the file to an OpenCV image.")
    return img


def detect(frame: np.ndarray) -> list:
    results = license_plate_detector(frame)
    plate_images = []
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            plate_img = frame[y1:y2, x1:x2]
            plate_images.append(plate_img)
    return plate_images


def read(plate_images: list) -> list:
    plate_numbers = []
    for img in plate_images:
        result = reader.readtext(img)
        plate_text = " ".join([text for (_, text, _) in result])
        plate_numbers.append(plate_text)
    return plate_numbers


def encode(frames: list) -> list:
    encoded_frames = []
    for frame in frames:
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = b64encode(buffer).decode('utf-8')
        encoded_frames.append(img_base64)
    return encoded_frames
