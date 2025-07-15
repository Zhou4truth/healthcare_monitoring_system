import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import List, Tuple, Optional, Dict
from .pose_estimator import PoseLandmarks


class BodySegmenter:
    """
    Body segmentation and privacy protection using MediaPipe Selfie Segmentation.
    Provides real-time body part detection and privacy zone blurring.
    """
    
    def __init__(self, model_selection: int = 1, blur_strength: int = 51):
        """
        Initialize body segmenter.
        
        Args:
            model_selection: 0 for general model, 1 for landscape model (better for full body)
            blur_strength: Gaussian blur kernel size for privacy zones
        """
        self.model_selection = model_selection
        self.blur_strength = blur_strength
        
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmentation = None
        
        # Privacy zones (body parts to blur)
        self.privacy_zones = {
            'face': True,
            'torso': False,  # Keep torso visible for fall detection
            'private_areas': True,
            'full_body': False  # Emergency mode: blur entire body
        }
        
        self.logger = logging.getLogger(__name__)
        self._initialize_segmentation()
    
    def _initialize_segmentation(self):
        """Initialize MediaPipe selfie segmentation."""
        try:
            self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
                model_selection=self.model_selection
            )
            self.logger.info("MediaPipe body segmentation initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize body segmentation: {e}")
            raise
    
    def segment_body(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Segment body from background.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Segmentation mask (0-255) or None if failed
        """
        if self.segmentation is None:
            return None
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process segmentation
            results = self.segmentation.process(rgb_frame)
            
            if results.segmentation_mask is not None:
                # Convert to 0-255 range
                mask = (results.segmentation_mask * 255).astype(np.uint8)
                return mask
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in body segmentation: {e}")
            return None
    
    def create_privacy_mask(self, frame: np.ndarray, pose_landmarks: Optional[PoseLandmarks] = None) -> np.ndarray:
        """
        Create privacy mask based on body parts and pose landmarks.
        
        Args:
            frame: Input frame
            pose_landmarks: Optional pose landmarks for precise masking
            
        Returns:
            Privacy mask (0=transparent, 255=blur)
        """
        h, w = frame.shape[:2]
        privacy_mask = np.zeros((h, w), dtype=np.uint8)
        
        if pose_landmarks is None:
            # Use full body segmentation if no pose landmarks
            if self.privacy_zones['full_body']:
                body_mask = self.segment_body(frame)
                if body_mask is not None:
                    privacy_mask = body_mask
            return privacy_mask
        
        landmarks = pose_landmarks.landmarks
        
        # Face privacy zone
        if self.privacy_zones['face'] and self._has_face_landmarks(landmarks):
            face_mask = self._create_face_mask(frame.shape[:2], landmarks)
            privacy_mask = cv2.bitwise_or(privacy_mask, face_mask)
        
        # Private areas privacy zone
        if self.privacy_zones['private_areas'] and self._has_torso_landmarks(landmarks):
            private_mask = self._create_private_areas_mask(frame.shape[:2], landmarks)
            privacy_mask = cv2.bitwise_or(privacy_mask, private_mask)
        
        # Full body privacy mode
        if self.privacy_zones['full_body']:
            body_mask = self.segment_body(frame)
            if body_mask is not None:
                privacy_mask = body_mask
        
        return privacy_mask
    
    def _has_face_landmarks(self, landmarks: Dict) -> bool:
        """Check if face landmarks are available and visible."""
        face_landmarks = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']
        return any(
            name in landmarks and landmarks[name].visibility > 0.5 
            for name in face_landmarks
        )
    
    def _has_torso_landmarks(self, landmarks: Dict) -> bool:
        """Check if torso landmarks are available and visible."""
        torso_landmarks = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        return all(
            name in landmarks and landmarks[name].visibility > 0.5 
            for name in torso_landmarks
        )
    
    def _create_face_mask(self, frame_shape: Tuple[int, int], landmarks: Dict) -> np.ndarray:
        """Create mask for face area."""
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        face_landmarks = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                         'mouth_left', 'mouth_right']
        
        face_points = []
        for name in face_landmarks:
            if name in landmarks and landmarks[name].visibility > 0.5:
                kp = landmarks[name]
                face_points.append((int(kp.x), int(kp.y)))
        
        if len(face_points) >= 3:
            # Create convex hull around face points
            face_points = np.array(face_points)
            hull = cv2.convexHull(face_points)
            
            # Expand the hull slightly
            center = np.mean(hull, axis=0)
            expanded_hull = center + 1.3 * (hull - center)
            expanded_hull = expanded_hull.astype(np.int32)
            
            cv2.fillPoly(mask, [expanded_hull], 255)
        
        return mask
    
    def _create_private_areas_mask(self, frame_shape: Tuple[int, int], landmarks: Dict) -> np.ndarray:
        """Create mask for private body areas."""
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Define private area around torso (excluding arms for fall detection)
        if self._has_torso_landmarks(landmarks):
            # Get torso landmarks
            left_shoulder = landmarks['left_shoulder']
            right_shoulder = landmarks['right_shoulder']
            left_hip = landmarks['left_hip']
            right_hip = landmarks['right_hip']
            
            # Create torso rectangle with some margin
            torso_points = [
                (int(left_shoulder.x), int(left_shoulder.y)),
                (int(right_shoulder.x), int(right_shoulder.y)),
                (int(right_hip.x), int(right_hip.y)),
                (int(left_hip.x), int(left_hip.y))
            ]
            
            # Add margin to torso area
            center_x = sum(p[0] for p in torso_points) / 4
            center_y = sum(p[1] for p in torso_points) / 4
            
            expanded_points = []
            for x, y in torso_points:
                # Expand by 20% from center
                new_x = center_x + 1.2 * (x - center_x)
                new_y = center_y + 1.2 * (y - center_y)
                expanded_points.append((int(new_x), int(new_y)))
            
            cv2.fillPoly(mask, [np.array(expanded_points)], 255)
        
        return mask
    
    def apply_privacy_filter(self, frame: np.ndarray, pose_landmarks: Optional[PoseLandmarks] = None) -> np.ndarray:
        """
        Apply privacy filter to frame.
        
        Args:
            frame: Input frame
            pose_landmarks: Optional pose landmarks for precise filtering
            
        Returns:
            Frame with privacy filter applied
        """
        # Create privacy mask
        privacy_mask = self.create_privacy_mask(frame, pose_landmarks)
        
        if np.sum(privacy_mask) == 0:
            # No privacy zones detected, return original frame
            return frame.copy()
        
        # Apply Gaussian blur to privacy areas
        blurred_frame = cv2.GaussianBlur(frame, (self.blur_strength, self.blur_strength), 0)
        
        # Combine original and blurred frames using mask
        result_frame = frame.copy()
        privacy_mask_3ch = cv2.cvtColor(privacy_mask, cv2.COLOR_GRAY2BGR)
        privacy_mask_norm = privacy_mask_3ch.astype(np.float32) / 255.0
        
        result_frame = (
            frame.astype(np.float32) * (1 - privacy_mask_norm) +
            blurred_frame.astype(np.float32) * privacy_mask_norm
        ).astype(np.uint8)
        
        return result_frame
    
    def set_privacy_mode(self, mode: str):
        """
        Set privacy protection mode.
        
        Args:
            mode: 'minimal', 'standard', 'maximum', 'emergency'
        """
        if mode == 'minimal':
            self.privacy_zones = {
                'face': False,
                'torso': False,
                'private_areas': False,
                'full_body': False
            }
        elif mode == 'standard':
            self.privacy_zones = {
                'face': True,
                'torso': False,
                'private_areas': True,
                'full_body': False
            }
        elif mode == 'maximum':
            self.privacy_zones = {
                'face': True,
                'torso': True,
                'private_areas': True,
                'full_body': False
            }
        elif mode == 'emergency':
            self.privacy_zones = {
                'face': True,
                'torso': True,
                'private_areas': True,
                'full_body': True
            }
        
        self.logger.info(f"Privacy mode set to: {mode}")
    
    def create_anonymized_frame(self, frame: np.ndarray, pose_landmarks: Optional[PoseLandmarks] = None) -> np.ndarray:
        """
        Create fully anonymized frame for data collection/analysis.
        
        Args:
            frame: Input frame
            pose_landmarks: Optional pose landmarks
            
        Returns:
            Anonymized frame (stick figure representation)
        """
        h, w = frame.shape[:2]
        
        # Create black background
        anonymized = np.zeros((h, w, 3), dtype=np.uint8)
        
        if pose_landmarks is not None:
            # Draw stick figure representation
            landmarks = pose_landmarks.landmarks
            
            # Draw body connections as white lines
            connections = [
                ('left_shoulder', 'right_shoulder'),
                ('left_shoulder', 'left_hip'),
                ('right_shoulder', 'right_hip'),
                ('left_hip', 'right_hip'),
                ('left_shoulder', 'left_elbow'),
                ('left_elbow', 'left_wrist'),
                ('right_shoulder', 'right_elbow'),
                ('right_elbow', 'right_wrist'),
                ('left_hip', 'left_knee'),
                ('left_knee', 'left_ankle'),
                ('right_hip', 'right_knee'),
                ('right_knee', 'right_ankle')
            ]
            
            for start_name, end_name in connections:
                if (start_name in landmarks and end_name in landmarks and
                    landmarks[start_name].visibility > 0.5 and landmarks[end_name].visibility > 0.5):
                    
                    start_point = (int(landmarks[start_name].x), int(landmarks[start_name].y))
                    end_point = (int(landmarks[end_name].x), int(landmarks[end_name].y))
                    
                    cv2.line(anonymized, start_point, end_point, (255, 255, 255), 3)
            
            # Draw key points as circles
            key_points = ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
                         'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
            
            for point_name in key_points:
                if point_name in landmarks and landmarks[point_name].visibility > 0.5:
                    point = (int(landmarks[point_name].x), int(landmarks[point_name].y))
                    cv2.circle(anonymized, point, 5, (255, 255, 255), -1)
        
        return anonymized