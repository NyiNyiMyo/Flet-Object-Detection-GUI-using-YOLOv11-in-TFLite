from pathlib import Path

import cv2
import numpy as np
try:
    # This will be used when the app is bundled on Android/iOS via Flet
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # This acts as fallback for local Windows desktop testing
    from ai_edge_litert.interpreter import Interpreter


COCO_CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


class YoloTFLiteDetector:

    def __init__(self, model_path: Path, confidence: float = 0.25, iou: float = 0.55):
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou

        self.interpreter = None
        self.input_index = None
        self.output_index = None

        self.input_width = 320
        self.input_height = 320

    def _load(self) -> None:
        if self.interpreter is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.interpreter = Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.input_index = input_details[0]["index"]
        self.output_index = output_details[0]["index"]

        shape = input_details[0]["shape"]

        if len(shape) == 4:
            self.input_height = int(shape[1])
            self.input_width = int(shape[2])

    def annotate(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        self._load()

        original_height, original_width = frame.shape[:2]

        resized = cv2.resize(
            frame,
            (self.input_width, self.input_height)
        )

        tensor = resized.astype(np.float32) / 255.0
        tensor = tensor[np.newaxis, :, :, :]

        self.interpreter.set_tensor(self.input_index, tensor)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_index)

        detections = self._filter(np.squeeze(output))

        boxes, scores, class_ids = self._rescale_and_nms(
            detections,
            original_width,
            original_height,
        )

        annotated = frame.copy()

        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = [int(value) for value in box]

            label = (
                COCO_CLASSES[class_id]
                if class_id < len(COCO_CLASSES)
                else f"class {class_id}"
            )

            color = self._class_color(class_id)

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            cv2.putText(
                annotated,
                f"{label} {score:.2f}",
                (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return annotated, len(boxes)

    def _filter(
        self,
        output: np.ndarray,
    ) -> list[tuple[float, float, float, float, int, float]]:

        if output.ndim != 2:
            raise ValueError(
                f"Unexpected YOLO output shape: {output.shape}"
            )

        if output.shape[0] < output.shape[1]:
            output = output.T

        results = []

        for row in output:
            if len(row) < 5:
                continue

            scores = row[4:]
            class_id = int(np.argmax(scores))
            score = float(scores[class_id])

            if score >= self.confidence:
                results.append(
                    (
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                        class_id,
                        score,
                    )
                )

        return results

    def _rescale_and_nms(self, detections, original_width, original_height):
        if len(detections) == 0:
            return [], [], []

        boxes = []
        scores = []
        class_ids = []

        for det in detections:
            cx, cy, box_width, box_height, class_id, score = det

            # TFLite output is normalized to 320x320
            x1 = (cx - box_width / 2) * self.input_width
            y1 = (cy - box_height / 2) * self.input_height
            x2 = (cx + box_width / 2) * self.input_width
            y2 = (cy + box_height / 2) * self.input_height

            # Scale from 320x320 to original frame
            x1 *= original_width / self.input_width
            x2 *= original_width / self.input_width
            y1 *= original_height / self.input_height
            y2 *= original_height / self.input_height

            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(class_id)

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        class_ids = np.array(class_ids, dtype=np.int32)

        keep = []

        for class_id in np.unique(class_ids):
            indices = np.where(class_ids == class_id)[0]

            class_boxes = boxes[indices]
            class_scores = scores[indices]

            x1 = class_boxes[:, 0]
            y1 = class_boxes[:, 1]
            x2 = class_boxes[:, 2]
            y2 = class_boxes[:, 3]

            areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            order = class_scores.argsort()[::-1]

            while len(order) > 0:
                i = order[0]
                keep.append(indices[i])

                if len(order) == 1:
                    break

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)

                intersection = w * h
                union = areas[i] + areas[order[1:]] - intersection
                iou = intersection / np.maximum(union, 1e-6)

                order = order[1:][iou < self.iou]

        return (
            boxes[keep],
            scores[keep],
            class_ids[keep],
        )

    @staticmethod
    def _class_color(class_id: int):
        hue = (class_id * 47) % 180
        hsv = np.uint8([[[hue, 220, 255]]])
        bgr = cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2BGR,
        )[0, 0]

        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    @staticmethod
    def _iou(a, b):
        left = max(a[0], b[0])
        top = max(a[1], b[1])
        right = min(a[2], b[2])
        bottom = min(a[3], b[3])

        intersection = (
            max(0.0, right - left)
            * max(0.0, bottom - top)
        )

        union = (
            (a[2] - a[0]) * (a[3] - a[1])
            + (b[2] - b[0]) * (b[3] - b[1])
            - intersection
        )

        return intersection / union if union > 0 else 0.0