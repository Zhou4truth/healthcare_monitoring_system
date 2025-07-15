import cv2
import numpy as np
import logging
import json
from typing import List, Dict, Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass, asdict
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import os

from .pose_estimator import PoseLandmarks


@dataclass
class FallEvent:
    """Represents a detected fall event."""
    timestamp: float
    confidence: float
    trigger_type: str  # 'rule_based', 'lstm', 'combined'
    pose_features: Dict[str, float]
    bbox: Tuple[int, int, int, int]
    severity: str  # 'low', 'medium', 'high'


class FallDetector:
    """
    LSTM + rule-based fall detection system with temporal analysis.
    Combines deep learning with heuristic rules for robust fall detection.
    """
    
    def __init__(self, sequence_length: int = 30, model_path: Optional[str] = None,
                 confidence_threshold: float = 0.7, rule_weight: float = 0.4, lstm_weight: float = 0.6):
        """
        Initialize fall detector.
        
        Args:
            sequence_length: Number of frames to analyze for temporal patterns
            model_path: Path to pre-trained LSTM model
            confidence_threshold: Minimum confidence for fall detection
            rule_weight: Weight for rule-based detection
            lstm_weight: Weight for LSTM detection
        """
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.rule_weight = rule_weight
        self.lstm_weight = lstm_weight
        
        # Temporal data storage
        self.pose_history: Deque[Dict[str, float]] = deque(maxlen=sequence_length)
        self.timestamp_history: Deque[float] = deque(maxlen=sequence_length)
        
        # LSTM model and scaler
        self.lstm_model = None
        self.feature_scaler = None
        self.feature_names = []
        
        # Rule-based thresholds
        self.rule_thresholds = {
            'torso_angle_threshold': np.pi / 3,  # 60 degrees from vertical
            'aspect_ratio_threshold': 1.5,       # Width/height ratio
            'velocity_threshold': 50,            # Pixels per frame
            'acceleration_threshold': 10,        # Change in velocity
            'height_drop_threshold': 0.3,        # Relative height drop
            'pose_confidence_threshold': 0.5     # Minimum pose visibility
        }
        
        # Fall state tracking
        self.fall_state = 'normal'  # 'normal', 'suspicious', 'falling', 'fallen'
        self.fall_start_time = None
        self.consecutive_fall_frames = 0
        self.min_fall_duration = 0.5  # Minimum fall duration in seconds
        
        self.logger = logging.getLogger(__name__)
        
        if model_path and os.path.exists(model_path):
            self._load_lstm_model(model_path)
        else:
            self._create_default_lstm_model()
    
    def _create_default_lstm_model(self):
        """Create a default LSTM model architecture."""
        try:
            # Define feature names (must match pose_estimator features)
            self.feature_names = [
                'torso_angle', 'torso_vertical_deviation', 'body_height',
                'bbox_aspect_ratio', 'center_of_mass_x', 'center_of_mass_y',
                'knee_hip_ratio', 'head_body_ratio'
            ]
            
            # Create LSTM model
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(self.sequence_length, len(self.feature_names))),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(32, return_sequences=False),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.lstm_model = model
            self.feature_scaler = StandardScaler()
            
            self.logger.info("Default LSTM model created")
            
        except Exception as e:
            self.logger.error(f"Failed to create LSTM model: {e}")
            self.lstm_model = None
    
    def _load_lstm_model(self, model_path: str):
        """Load pre-trained LSTM model."""
        try:
            # Load model
            self.lstm_model = tf.keras.models.load_model(model_path)
            
            # Load scaler and feature names
            scaler_path = model_path.replace('.h5', '_scaler.pkl').replace('.keras', '_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
            
            config_path = model_path.replace('.h5', '_config.json').replace('.keras', '_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.feature_names = config.get('feature_names', self.feature_names)
            
            self.logger.info(f"LSTM model loaded: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load LSTM model: {e}")
            self._create_default_lstm_model()
    
    def save_model(self, model_path: str):
        """Save trained LSTM model."""
        try:
            if self.lstm_model is not None:
                self.lstm_model.save(model_path)
                
                # Save scaler
                if self.feature_scaler is not None:
                    scaler_path = model_path.replace('.h5', '_scaler.pkl').replace('.keras', '_scaler.pkl')
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.feature_scaler, f)
                
                # Save configuration
                config_path = model_path.replace('.h5', '_config.json').replace('.keras', '_config.json')
                config = {
                    'feature_names': self.feature_names,
                    'sequence_length': self.sequence_length,
                    'confidence_threshold': self.confidence_threshold
                }
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                self.logger.info(f"Model saved: {model_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    def analyze_frame(self, pose_landmarks: PoseLandmarks) -> Optional[FallEvent]:
        """
        Analyze current frame for fall detection.
        
        Args:
            pose_landmarks: Current pose landmarks
            
        Returns:
            FallEvent if fall detected, None otherwise
        """
        # Extract features from pose landmarks
        features = self._extract_features(pose_landmarks)
        if not features:
            return None
        
        # Add to temporal history
        self.pose_history.append(features)
        self.timestamp_history.append(pose_landmarks.timestamp)
        
        # Need minimum history for analysis
        if len(self.pose_history) < min(10, self.sequence_length):
            return None
        
        # Rule-based detection
        rule_confidence = self._rule_based_detection(features)
        
        # LSTM-based detection
        lstm_confidence = self._lstm_detection()
        
        # Combine detections
        combined_confidence = (
            self.rule_weight * rule_confidence + 
            self.lstm_weight * lstm_confidence
        )
        
        # Temporal filtering
        combined_confidence = self._temporal_filtering(combined_confidence, pose_landmarks.timestamp)
        
        # Check if fall detected
        if combined_confidence >= self.confidence_threshold:
            fall_event = FallEvent(
                timestamp=pose_landmarks.timestamp,
                confidence=combined_confidence,
                trigger_type=self._get_trigger_type(rule_confidence, lstm_confidence),
                pose_features=features,
                bbox=pose_landmarks.bbox,
                severity=self._assess_severity(combined_confidence, features)
            )
            
            self.logger.warning(f"Fall detected! Confidence: {combined_confidence:.3f}")
            return fall_event
        
        return None
    
    def _extract_features(self, pose_landmarks: PoseLandmarks) -> Dict[str, float]:
        """Extract features for fall detection."""
        # Get features from pose estimator
        base_features = {}
        
        # Import here to avoid circular imports
        from .pose_estimator import PoseEstimator
        pose_estimator = PoseEstimator()
        base_features = pose_estimator.get_fall_detection_features(pose_landmarks)
        
        # Add temporal features if we have history
        if len(self.pose_history) > 1:
            # Velocity features
            prev_features = self.pose_history[-1]
            if 'center_of_mass_x' in base_features and 'center_of_mass_x' in prev_features:
                dt = self.timestamp_history[-1] - self.timestamp_history[-2] if len(self.timestamp_history) > 1 else 1.0
                dt = max(dt, 0.001)  # Avoid division by zero
                
                vel_x = (base_features['center_of_mass_x'] - prev_features['center_of_mass_x']) / dt
                vel_y = (base_features['center_of_mass_y'] - prev_features['center_of_mass_y']) / dt
                
                base_features['velocity_x'] = vel_x
                base_features['velocity_y'] = vel_y
                base_features['velocity_magnitude'] = np.sqrt(vel_x**2 + vel_y**2)
        
        # Add acceleration features if we have more history
        if len(self.pose_history) > 2:
            # Simple acceleration from last two velocity measurements
            if 'velocity_magnitude' in base_features:
                prev_vel = self.pose_history[-1].get('velocity_magnitude', 0)
                dt = self.timestamp_history[-1] - self.timestamp_history[-2] if len(self.timestamp_history) > 1 else 1.0
                dt = max(dt, 0.001)
                
                acceleration = (base_features['velocity_magnitude'] - prev_vel) / dt
                base_features['acceleration'] = acceleration
        
        return base_features
    
    def _rule_based_detection(self, features: Dict[str, float]) -> float:
        """Rule-based fall detection."""
        confidence = 0.0
        
        # Rule 1: Torso angle deviation
        if 'torso_vertical_deviation' in features:
            angle_confidence = min(1.0, features['torso_vertical_deviation'] / self.rule_thresholds['torso_angle_threshold'])
            confidence += 0.3 * angle_confidence
        
        # Rule 2: Aspect ratio (person becomes horizontal)
        if 'bbox_aspect_ratio' in features:
            ratio_confidence = min(1.0, max(0, features['bbox_aspect_ratio'] - 1.0) / 0.5)
            confidence += 0.25 * ratio_confidence
        
        # Rule 3: Rapid vertical movement
        if 'velocity_y' in features:
            velocity_confidence = min(1.0, abs(features['velocity_y']) / self.rule_thresholds['velocity_threshold'])
            confidence += 0.2 * velocity_confidence
        
        # Rule 4: High acceleration (sudden movement)
        if 'acceleration' in features:
            accel_confidence = min(1.0, abs(features['acceleration']) / self.rule_thresholds['acceleration_threshold'])
            confidence += 0.15 * accel_confidence
        
        # Rule 5: Center of mass position
        if 'center_of_mass_y' in features and 'body_height' in features:
            # Check if center of mass is low relative to body height
            if features['body_height'] > 0:
                relative_height = features['center_of_mass_y'] / features['body_height']
                height_confidence = min(1.0, max(0, 0.7 - relative_height) / 0.3)
                confidence += 0.1 * height_confidence
        
        return min(1.0, confidence)
    
    def _lstm_detection(self) -> float:
        """LSTM-based fall detection."""
        if self.lstm_model is None or len(self.pose_history) < self.sequence_length:
            return 0.0
        
        try:
            # Prepare sequence data
            sequence_data = []
            for features in list(self.pose_history)[-self.sequence_length:]:
                feature_vector = []
                for feature_name in self.feature_names:
                    feature_vector.append(features.get(feature_name, 0.0))
                sequence_data.append(feature_vector)
            
            # Convert to numpy array
            sequence_array = np.array(sequence_data).reshape(1, self.sequence_length, len(self.feature_names))
            
            # Scale features if scaler is available
            if self.feature_scaler is not None:
                # Reshape for scaling
                original_shape = sequence_array.shape
                sequence_array = sequence_array.reshape(-1, sequence_array.shape[-1])
                sequence_array = self.feature_scaler.transform(sequence_array)
                sequence_array = sequence_array.reshape(original_shape)
            
            # Get prediction
            prediction = self.lstm_model.predict(sequence_array, verbose=0)
            confidence = float(prediction[0][0])
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"LSTM prediction error: {e}")
            return 0.0
    
    def _temporal_filtering(self, confidence: float, timestamp: float) -> float:
        """Apply temporal filtering to reduce false positives."""
        # State machine for fall detection
        if confidence >= self.confidence_threshold:
            if self.fall_state == 'normal':
                self.fall_state = 'suspicious'
                self.fall_start_time = timestamp
                self.consecutive_fall_frames = 1
                return confidence * 0.5  # Reduce confidence for first detection
            
            elif self.fall_state == 'suspicious':
                self.consecutive_fall_frames += 1
                if timestamp - self.fall_start_time >= self.min_fall_duration:
                    self.fall_state = 'falling'
                    return confidence
                else:
                    return confidence * 0.7  # Still building confidence
            
            elif self.fall_state == 'falling':
                self.consecutive_fall_frames += 1
                return confidence  # Full confidence
        
        else:
            # Low confidence, reset state
            if self.fall_state in ['suspicious', 'falling']:
                # Check if we should reset or continue monitoring
                if confidence < 0.3:
                    self.fall_state = 'normal'
                    self.consecutive_fall_frames = 0
                    self.fall_start_time = None
            
            return confidence * 0.1  # Very low confidence for non-fall frames
    
    def _get_trigger_type(self, rule_confidence: float, lstm_confidence: float) -> str:
        """Determine what triggered the fall detection."""
        if rule_confidence > 0.7 and lstm_confidence > 0.7:
            return 'combined'
        elif rule_confidence > lstm_confidence:
            return 'rule_based'
        else:
            return 'lstm'
    
    def _assess_severity(self, confidence: float, features: Dict[str, float]) -> str:
        """Assess fall severity based on confidence and features."""
        if confidence >= 0.9:
            return 'high'
        elif confidence >= 0.75:
            # Check for additional severity indicators
            severity_indicators = 0
            
            if features.get('velocity_magnitude', 0) > 100:
                severity_indicators += 1
            if features.get('acceleration', 0) > 20:
                severity_indicators += 1
            if features.get('torso_vertical_deviation', 0) > np.pi/2:
                severity_indicators += 1
            
            if severity_indicators >= 2:
                return 'high'
            else:
                return 'medium'
        else:
            return 'low'
    
    def train_lstm(self, training_data: List[Tuple[List[Dict[str, float]], bool]], 
                   epochs: int = 50, validation_split: float = 0.2):
        """
        Train the LSTM model with labeled data.
        
        Args:
            training_data: List of (sequence, is_fall) tuples
            epochs: Number of training epochs
            validation_split: Fraction of data for validation
        """
        if self.lstm_model is None:
            self.logger.error("LSTM model not initialized")
            return
        
        try:
            # Prepare training data
            X_train = []
            y_train = []
            
            for sequence, is_fall in training_data:
                if len(sequence) >= self.sequence_length:
                    # Extract feature vectors
                    feature_sequence = []
                    for features in sequence[-self.sequence_length:]:
                        feature_vector = []
                        for feature_name in self.feature_names:
                            feature_vector.append(features.get(feature_name, 0.0))
                        feature_sequence.append(feature_vector)
                    
                    X_train.append(feature_sequence)
                    y_train.append(1 if is_fall else 0)
            
            if len(X_train) == 0:
                self.logger.error("No valid training sequences found")
                return
            
            # Convert to numpy arrays
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Scale features
            original_shape = X_train.shape
            X_train_flat = X_train.reshape(-1, X_train.shape[-1])
            
            if self.feature_scaler is None:
                self.feature_scaler = StandardScaler()
            
            X_train_scaled = self.feature_scaler.fit_transform(X_train_flat)
            X_train_scaled = X_train_scaled.reshape(original_shape)
            
            # Train model
            history = self.lstm_model.fit(
                X_train_scaled, y_train,
                epochs=epochs,
                validation_split=validation_split,
                batch_size=32,
                verbose=1
            )
            
            self.logger.info(f"LSTM training completed. Final accuracy: {history.history['accuracy'][-1]:.3f}")
            
        except Exception as e:
            self.logger.error(f"LSTM training failed: {e}")
    
    def reset_state(self):
        """Reset fall detection state."""
        self.fall_state = 'normal'
        self.fall_start_time = None
        self.consecutive_fall_frames = 0
        self.pose_history.clear()
        self.timestamp_history.clear()
        self.logger.info("Fall detector state reset")
    
    def get_status(self) -> Dict[str, any]:
        """Get current detector status."""
        return {
            'fall_state': self.fall_state,
            'consecutive_fall_frames': self.consecutive_fall_frames,
            'history_length': len(self.pose_history),
            'lstm_available': self.lstm_model is not None,
            'scaler_available': self.feature_scaler is not None
        }