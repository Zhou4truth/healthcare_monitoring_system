import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from ultralytics import YOLO
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False


class PersonDetector:
    """
    YOLOv8-based person detection optimized for Jetson devices with TensorRT acceleration.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.4, use_tensorrt: bool = True):
        """
        Initialize person detector.
        
        Args:
            model_path: Path to YOLOv8 model
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            use_tensorrt: Use TensorRT optimization for Jetson
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_tensorrt = use_tensorrt
        
        self.model = None
        self.input_size = (640, 640)
        self.class_names = []
        
        self.logger = logging.getLogger(__name__)
        
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load and optimize YOLOv8 model."""
        try:
            self.model = YOLO(model_path)
            
            # Get class names - we only care about person (class 0)
            self.class_names = self.model.names
            
            # Export to TensorRT for Jetson optimization
            if self.use_tensorrt:
                try:
                    trt_path = model_path.replace('.pt', '.engine')
                    self.model.export(format='engine', device=0, half=True)
                    self.model = YOLO(trt_path)
                    self.logger.info("TensorRT optimization enabled")
                except Exception as e:
                    self.logger.warning(f"TensorRT optimization failed: {e}")
            
            self.logger.info(f"Person detector loaded: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect persons in frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of (x1, y1, x2, y2, confidence) tuples for person detections
        """
        if self.model is None:
            return []
        
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, 
                               iou=self.iou_threshold, classes=[0])  # class 0 = person
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Ensure coordinates are within frame bounds
                        h, w = frame.shape[:2]
                        x1 = max(0, min(int(x1), w-1))
                        y1 = max(0, min(int(y1), h-1))
                        x2 = max(x1+1, min(int(x2), w))
                        y2 = max(y1+1, min(int(y2), h))
                        
                        detections.append((x1, y1, x2, y2, float(confidence)))
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error during person detection: {e}")
            return []
    
    def get_largest_person(self, detections: List[Tuple[int, int, int, int, float]]) -> Optional[Tuple[int, int, int, int, float]]:
        """
        Get the largest person detection (by area).
        
        Args:
            detections: List of person detections
            
        Returns:
            Largest detection or None if no detections
        """
        if not detections:
            return None
        
        largest_detection = None
        largest_area = 0
        
        for detection in detections:
            x1, y1, x2, y2, conf = detection
            area = (x2 - x1) * (y2 - y1)
            
            if area > largest_area:
                largest_area = area
                largest_detection = detection
        
        return largest_detection
    
    def filter_by_size(self, detections: List[Tuple[int, int, int, int, float]], 
                       min_area: int = 1000, max_area: int = 500000) -> List[Tuple[int, int, int, int, float]]:
        """
        Filter detections by bounding box area.
        
        Args:
            detections: List of person detections
            min_area: Minimum bounding box area
            max_area: Maximum bounding box area
            
        Returns:
            Filtered detections
        """
        filtered = []
        
        for detection in detections:
            x1, y1, x2, y2, conf = detection
            area = (x2 - x1) * (y2 - y1)
            
            if min_area <= area <= max_area:
                filtered.append(detection)
        
        return filtered
    
    def draw_detections(self, frame: np.ndarray, 
                       detections: List[Tuple[int, int, int, int, float]]) -> np.ndarray:
        """
        Draw person detections on frame.
        
        Args:
            frame: Input frame
            detections: List of person detections
            
        Returns:
            Frame with drawn detections
        """
        result_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2, conf = detection
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"Person: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(result_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_frame
    
    def get_person_crop(self, frame: np.ndarray, detection: Tuple[int, int, int, int, float],
                       padding: float = 0.1) -> Optional[np.ndarray]:
        """
        Extract person crop from frame.
        
        Args:
            frame: Input frame
            detection: Person detection (x1, y1, x2, y2, conf)
            padding: Padding ratio around bounding box
            
        Returns:
            Cropped person image or None
        """
        if detection is None:
            return None
        
        x1, y1, x2, y2, _ = detection
        h, w = frame.shape[:2]
        
        # Add padding
        box_w = x2 - x1
        box_h = y2 - y1
        pad_w = int(box_w * padding)
        pad_h = int(box_h * padding)
        
        # Expand bounding box with padding
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        return frame[y1:y2, x1:x2]