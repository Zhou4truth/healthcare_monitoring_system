import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PoseKeypoint:
    """Represents a single pose keypoint."""
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class PoseLandmarks:
    """Represents all pose landmarks for a person."""
    landmarks: Dict[str, PoseKeypoint]
    timestamp: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2


class PoseEstimator:
    """
    MediaPipe-based pose estimation optimized for fall detection.
    """
    
    # MediaPipe pose landmark indices
    POSE_LANDMARKS = {
        'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
        'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
        'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
        'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13,
        'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
        'left_pinky': 17, 'right_pinky': 18, 'left_index': 19, 'right_index': 20,
        'left_thumb': 21, 'right_thumb': 22, 'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26, 'left_ankle': 27, 'right_ankle': 28,
        'left_heel': 29, 'right_heel': 30, 'left_foot_index': 31, 'right_foot_index': 32
    }
    
    # Key points for fall detection
    FALL_DETECTION_KEYPOINTS = [
        'nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, smooth_landmarks: bool = True):
        """
        Initialize pose estimator.
        
        Args:
            model_complexity: Model complexity (0, 1, or 2)
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
            smooth_landmarks: Enable landmark smoothing
        """
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.smooth_landmarks = smooth_landmarks
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = None
        self.logger = logging.getLogger(__name__)
        
        self._initialize_pose()
    
    def _initialize_pose(self):
        """Initialize MediaPipe pose estimation."""
        try:
            self.pose = self.mp_pose.Pose(
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                smooth_landmarks=self.smooth_landmarks
            )
            self.logger.info("MediaPipe pose estimator initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize pose estimator: {e}")
            raise
    
    def estimate_pose(self, frame: np.ndarray, person_bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[PoseLandmarks]:
        """
        Estimate pose landmarks for a person in the frame.
        
        Args:
            frame: Input frame (BGR format)
            person_bbox: Optional bounding box (x1, y1, x2, y2) to crop person
            
        Returns:
            PoseLandmarks object or None if no pose detected
        """
        if self.pose is None:
            return None
        
        try:
            # Crop frame to person bbox if provided
            if person_bbox is not None:
                x1, y1, x2, y2 = person_bbox
                person_frame = frame[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1
            else:
                person_frame = frame
                offset_x, offset_y = 0, 0
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks is None:
                return None
            
            # Extract landmarks
            landmarks_dict = {}
            h, w = person_frame.shape[:2]
            
            for name, idx in self.POSE_LANDMARKS.items():
                landmark = results.pose_landmarks.landmark[idx]
                
                # Convert normalized coordinates to pixel coordinates
                x = landmark.x * w + offset_x
                y = landmark.y * h + offset_y
                z = landmark.z * w  # Depth relative to hip
                visibility = landmark.visibility
                
                landmarks_dict[name] = PoseKeypoint(x, y, z, visibility)
            
            # Calculate bounding box from landmarks
            if person_bbox is not None:
                bbox = person_bbox
            else:
                bbox = self._calculate_pose_bbox(landmarks_dict, frame.shape[:2])
            
            return PoseLandmarks(
                landmarks=landmarks_dict,
                timestamp=cv2.getTickCount() / cv2.getTickFrequency(),
                bbox=bbox
            )
            
        except Exception as e:
            self.logger.error(f"Error estimating pose: {e}")
            return None
    
    def _calculate_pose_bbox(self, landmarks: Dict[str, PoseKeypoint], frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Calculate bounding box from pose landmarks."""
        h, w = frame_shape
        
        valid_landmarks = [lm for lm in landmarks.values() if lm.visibility > 0.5]
        if not valid_landmarks:
            return (0, 0, w, h)
        
        x_coords = [lm.x for lm in valid_landmarks]
        y_coords = [lm.y for lm in valid_landmarks]
        
        x1 = max(0, int(min(x_coords)) - 20)
        y1 = max(0, int(min(y_coords)) - 20)
        x2 = min(w, int(max(x_coords)) + 20)
        y2 = min(h, int(max(y_coords)) + 20)
        
        return (x1, y1, x2, y2)
    
    def get_fall_detection_features(self, pose_landmarks: PoseLandmarks) -> Dict[str, float]:
        """
        Extract features relevant for fall detection.
        
        Args:
            pose_landmarks: Pose landmarks
            
        Returns:
            Dictionary of fall detection features
        """
        features = {}
        landmarks = pose_landmarks.landmarks
        
        # Body orientation features
        if all(kp in landmarks for kp in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            # Calculate torso angle
            shoulder_center = (
                (landmarks['left_shoulder'].x + landmarks['right_shoulder'].x) / 2,
                (landmarks['left_shoulder'].y + landmarks['right_shoulder'].y) / 2
            )
            hip_center = (
                (landmarks['left_hip'].x + landmarks['right_hip'].x) / 2,
                (landmarks['left_hip'].y + landmarks['right_hip'].y) / 2
            )
            
            torso_angle = np.arctan2(
                hip_center[1] - shoulder_center[1],
                hip_center[0] - shoulder_center[0]
            )
            features['torso_angle'] = float(torso_angle)
            features['torso_vertical_deviation'] = abs(torso_angle - np.pi/2)
        
        # Height ratios
        if 'nose' in landmarks and 'left_ankle' in landmarks:
            body_height = abs(landmarks['nose'].y - landmarks['left_ankle'].y)
            features['body_height'] = float(body_height)
            
            # Aspect ratio of bounding box
            x1, y1, x2, y2 = pose_landmarks.bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            features['bbox_aspect_ratio'] = float(bbox_width / max(bbox_height, 1))
        
        # Center of mass position
        valid_keypoints = [kp for name, kp in landmarks.items() 
                          if name in self.FALL_DETECTION_KEYPOINTS and kp.visibility > 0.5]
        
        if valid_keypoints:
            com_x = sum(kp.x for kp in valid_keypoints) / len(valid_keypoints)
            com_y = sum(kp.y for kp in valid_keypoints) / len(valid_keypoints)
            features['center_of_mass_x'] = float(com_x)
            features['center_of_mass_y'] = float(com_y)
        
        # Limb positions relative to torso
        if all(kp in landmarks for kp in ['left_hip', 'right_hip', 'left_knee', 'right_knee']):
            hip_y = (landmarks['left_hip'].y + landmarks['right_hip'].y) / 2
            knee_y = (landmarks['left_knee'].y + landmarks['right_knee'].y) / 2
            features['knee_hip_ratio'] = float((knee_y - hip_y) / max(abs(hip_y), 1))
        
        # Head position relative to body
        if all(kp in landmarks for kp in ['nose', 'left_hip', 'right_hip']):
            hip_y = (landmarks['left_hip'].y + landmarks['right_hip'].y) / 2
            head_body_ratio = (landmarks['nose'].y - hip_y) / max(abs(hip_y), 1)
            features['head_body_ratio'] = float(head_body_ratio)
        
        return features
    
    def draw_pose(self, frame: np.ndarray, pose_landmarks: PoseLandmarks) -> np.ndarray:
        """
        Draw pose landmarks on frame.
        
        Args:
            frame: Input frame
            pose_landmarks: Pose landmarks to draw
            
        Returns:
            Frame with drawn pose
        """
        result_frame = frame.copy()
        
        # Convert landmarks back to MediaPipe format for drawing
        mp_landmarks = self.mp_pose.PoseLandmark
        landmarks_list = []
        
        for i in range(33):  # MediaPipe has 33 pose landmarks
            for name, idx in self.POSE_LANDMARKS.items():
                if idx == i and name in pose_landmarks.landmarks:
                    kp = pose_landmarks.landmarks[name]
                    # Normalize coordinates
                    h, w = frame.shape[:2]
                    normalized_landmark = type('Landmark', (), {
                        'x': kp.x / w,
                        'y': kp.y / h,
                        'z': kp.z / w,
                        'visibility': kp.visibility
                    })()
                    landmarks_list.append(normalized_landmark)
                    break
            else:
                # Add dummy landmark if not found
                landmarks_list.append(type('Landmark', (), {
                    'x': 0, 'y': 0, 'z': 0, 'visibility': 0
                })())
        
        # Create MediaPipe landmarks object
        mp_landmarks_obj = type('Landmarks', (), {'landmark': landmarks_list})()
        
        # Draw the pose
        self.mp_drawing.draw_landmarks(
            result_frame,
            mp_landmarks_obj,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        return result_frame
    
    def is_valid_pose(self, pose_landmarks: PoseLandmarks, min_visibility: float = 0.5) -> bool:
        """
        Check if pose has sufficient landmark visibility for fall detection.
        
        Args:
            pose_landmarks: Pose landmarks
            min_visibility: Minimum visibility threshold
            
        Returns:
            True if pose is valid for fall detection
        """
        required_keypoints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        
        for keypoint_name in required_keypoints:
            if keypoint_name not in pose_landmarks.landmarks:
                return False
            
            if pose_landmarks.landmarks[keypoint_name].visibility < min_visibility:
                return False
        
        return True