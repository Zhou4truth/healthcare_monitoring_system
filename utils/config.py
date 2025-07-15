import json
import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from pathlib import Path

from .alert_manager import AlertConfig


@dataclass
class CameraConfig:
    """Camera configuration."""
    source: str = 0  # Camera index or RTSP URL
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 5
    auto_exposure: bool = True
    exposure_value: int = -6  # For manual exposure


@dataclass
class PersonDetectionConfig:
    """Person detection configuration."""
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.4
    use_tensorrt: bool = True
    min_person_area: int = 1000
    max_person_area: int = 500000


@dataclass
class PoseEstimationConfig:
    """Pose estimation configuration."""
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    smooth_landmarks: bool = True
    min_visibility_threshold: float = 0.5


@dataclass
class FallDetectionConfig:
    """Fall detection configuration."""
    sequence_length: int = 30
    confidence_threshold: float = 0.7
    rule_weight: float = 0.4
    lstm_weight: float = 0.6
    model_path: Optional[str] = None
    min_fall_duration: float = 0.5
    
    # Rule-based thresholds
    torso_angle_threshold: float = 1.047  # 60 degrees in radians
    aspect_ratio_threshold: float = 1.5
    velocity_threshold: float = 50.0
    acceleration_threshold: float = 10.0
    height_drop_threshold: float = 0.3


@dataclass
class PrivacyConfig:
    """Privacy protection configuration."""
    enabled: bool = True
    mode: str = "standard"  # minimal, standard, maximum, emergency
    blur_strength: int = 51
    model_selection: int = 1
    anonymize_output: bool = False


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    target_fps: int = 30
    max_cpu_usage: float = 80.0
    max_memory_mb: int = 2048
    enable_gpu_acceleration: bool = True
    enable_tensorrt: bool = True
    optimization_level: str = "balanced"  # fast, balanced, accurate


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file_path: str = "fall_detection.log"
    max_file_size_mb: int = 50
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True


@dataclass
class SystemConfig:
    """Complete system configuration."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    person_detection: PersonDetectionConfig = field(default_factory=PersonDetectionConfig)
    pose_estimation: PoseEstimationConfig = field(default_factory=PoseEstimationConfig)
    fall_detection: FallDetectionConfig = field(default_factory=FallDetectionConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # System settings
    enable_display: bool = True
    display_fps: bool = True
    display_detections: bool = True
    display_pose: bool = True
    save_events: bool = True
    events_directory: str = "events"
    max_event_files: int = 1000


class ConfigManager:
    """
    Configuration management system for the fall detection application.
    """
    
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file)
        self.config = SystemConfig()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration if file exists
        if self.config_file.exists():
            self.load_config()
        else:
            self.save_config()  # Create default config file
    
    def load_config(self) -> bool:
        """
        Load configuration from file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_file, 'r') as f:
                config_dict = json.load(f)
            
            # Update config object with loaded values
            self._update_config_from_dict(config_dict)
            
            self.logger.info(f"Configuration loaded from {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return False
    
    def save_config(self) -> bool:
        """
        Save current configuration to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert config to dictionary
            config_dict = asdict(self.config)
            
            # Save to file with pretty formatting
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            self.logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration object from dictionary."""
        # Update camera config
        if 'camera' in config_dict:
            camera_dict = config_dict['camera']
            for key, value in camera_dict.items():
                if hasattr(self.config.camera, key):
                    setattr(self.config.camera, key, value)
        
        # Update person detection config
        if 'person_detection' in config_dict:
            person_dict = config_dict['person_detection']
            for key, value in person_dict.items():
                if hasattr(self.config.person_detection, key):
                    setattr(self.config.person_detection, key, value)
        
        # Update pose estimation config
        if 'pose_estimation' in config_dict:
            pose_dict = config_dict['pose_estimation']
            for key, value in pose_dict.items():
                if hasattr(self.config.pose_estimation, key):
                    setattr(self.config.pose_estimation, key, value)
        
        # Update fall detection config
        if 'fall_detection' in config_dict:
            fall_dict = config_dict['fall_detection']
            for key, value in fall_dict.items():
                if hasattr(self.config.fall_detection, key):
                    setattr(self.config.fall_detection, key, value)
        
        # Update privacy config
        if 'privacy' in config_dict:
            privacy_dict = config_dict['privacy']
            for key, value in privacy_dict.items():
                if hasattr(self.config.privacy, key):
                    setattr(self.config.privacy, key, value)
        
        # Update alerts config
        if 'alerts' in config_dict:
            alerts_dict = config_dict['alerts']
            # Reconstruct AlertConfig object
            alert_config = AlertConfig()
            for key, value in alerts_dict.items():
                if hasattr(alert_config, key):
                    setattr(alert_config, key, value)
            self.config.alerts = alert_config
        
        # Update performance config
        if 'performance' in config_dict:
            performance_dict = config_dict['performance']
            for key, value in performance_dict.items():
                if hasattr(self.config.performance, key):
                    setattr(self.config.performance, key, value)
        
        # Update logging config
        if 'logging' in config_dict:
            logging_dict = config_dict['logging']
            for key, value in logging_dict.items():
                if hasattr(self.config.logging, key):
                    setattr(self.config.logging, key, value)
        
        # Update system settings
        system_keys = ['enable_display', 'display_fps', 'display_detections', 
                      'display_pose', 'save_events', 'events_directory', 'max_event_files']
        for key in system_keys:
            if key in config_dict:
                setattr(self.config, key, config_dict[key])
    
    def get_config(self) -> SystemConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> bool:
        """
        Update configuration with keyword arguments.
        
        Args:
            **kwargs: Configuration values to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    # Try to update nested configs
                    parts = key.split('.')
                    if len(parts) == 2:
                        section, param = parts
                        if hasattr(self.config, section):
                            section_obj = getattr(self.config, section)
                            if hasattr(section_obj, param):
                                setattr(section_obj, param, value)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False
    
    def reset_to_defaults(self):
        """Reset configuration to default values."""
        self.config = SystemConfig()
        self.save_config()
        self.logger.info("Configuration reset to defaults")
    
    def validate_config(self) -> List[str]:
        """
        Validate current configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate camera config
        if not isinstance(self.config.camera.source, (int, str)):
            errors.append("Camera source must be integer or string")
        
        if self.config.camera.width <= 0 or self.config.camera.height <= 0:
            errors.append("Camera resolution must be positive")
        
        if self.config.camera.fps <= 0:
            errors.append("Camera FPS must be positive")
        
        # Validate detection thresholds
        if not 0 <= self.config.person_detection.confidence_threshold <= 1:
            errors.append("Person detection confidence threshold must be between 0 and 1")
        
        if not 0 <= self.config.pose_estimation.min_detection_confidence <= 1:
            errors.append("Pose detection confidence must be between 0 and 1")
        
        if not 0 <= self.config.fall_detection.confidence_threshold <= 1:
            errors.append("Fall detection confidence threshold must be between 0 and 1")
        
        # Validate weights
        weights_sum = self.config.fall_detection.rule_weight + self.config.fall_detection.lstm_weight
        if abs(weights_sum - 1.0) > 0.01:
            errors.append("Rule weight + LSTM weight should equal 1.0")
        
        # Validate file paths
        if self.config.fall_detection.model_path:
            if not Path(self.config.fall_detection.model_path).exists():
                errors.append(f"Fall detection model file not found: {self.config.fall_detection.model_path}")
        
        # Validate privacy mode
        valid_privacy_modes = ['minimal', 'standard', 'maximum', 'emergency']
        if self.config.privacy.mode not in valid_privacy_modes:
            errors.append(f"Privacy mode must be one of: {valid_privacy_modes}")
        
        # Validate performance settings
        if self.config.performance.target_fps <= 0:
            errors.append("Target FPS must be positive")
        
        if self.config.performance.max_cpu_usage <= 0 or self.config.performance.max_cpu_usage > 100:
            errors.append("Max CPU usage must be between 0 and 100")
        
        # Validate logging settings
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.config.logging.level not in valid_log_levels:
            errors.append(f"Log level must be one of: {valid_log_levels}")
        
        return errors
    
    def create_sample_config(self, filename: str = "sample_config.json"):
        """Create a sample configuration file with comments."""
        sample_config = {
            "_comment": "Edge AI Fall Detection System Configuration",
            "_version": "1.0",
            "_description": {
                "camera": "Camera input settings",
                "person_detection": "YOLO person detection parameters",
                "pose_estimation": "MediaPipe pose estimation settings",
                "fall_detection": "LSTM + rule-based fall detection config",
                "privacy": "Privacy protection and anonymization",
                "alerts": "Notification and alert settings",
                "performance": "System performance optimization",
                "logging": "Logging configuration"
            },
            
            "camera": {
                "source": "0 for webcam, or RTSP URL like 'rtsp://camera_ip:554/stream'",
                "width": 640,
                "height": 480,
                "fps": 30,
                "buffer_size": 5,
                "auto_exposure": True,
                "exposure_value": -6
            },
            
            "person_detection": {
                "model_path": "yolov8n.pt",
                "confidence_threshold": 0.5,
                "iou_threshold": 0.4,
                "use_tensorrt": True,
                "min_person_area": 1000,
                "max_person_area": 500000
            },
            
            "pose_estimation": {
                "model_complexity": 1,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
                "smooth_landmarks": True,
                "min_visibility_threshold": 0.5
            },
            
            "fall_detection": {
                "sequence_length": 30,
                "confidence_threshold": 0.7,
                "rule_weight": 0.4,
                "lstm_weight": 0.6,
                "model_path": None,
                "min_fall_duration": 0.5,
                "torso_angle_threshold": 1.047,
                "aspect_ratio_threshold": 1.5,
                "velocity_threshold": 50.0,
                "acceleration_threshold": 10.0,
                "height_drop_threshold": 0.3
            },
            
            "privacy": {
                "enabled": True,
                "mode": "standard",
                "blur_strength": 51,
                "model_selection": 1,
                "anonymize_output": False
            },
            
            "alerts": {
                "email_enabled": False,
                "email_smtp_server": "smtp.gmail.com",
                "email_smtp_port": 587,
                "email_username": "your-email@gmail.com",
                "email_password": "your-app-password",
                "email_recipients": ["recipient@example.com"],
                "webhook_enabled": False,
                "webhook_url": "https://your-webhook-url.com/endpoint",
                "webhook_headers": {"Content-Type": "application/json"},
                "sms_enabled": False,
                "sms_api_key": "your-sms-api-key",
                "sms_api_url": "https://api.sms-provider.com/send",
                "sms_recipients": ["+1234567890"],
                "sound_enabled": True,
                "sound_file": "/usr/share/sounds/alsa/Front_Left.wav",
                "local_logging": True,
                "log_file": "fall_alerts.log",
                "cooldown_period": 60,
                "max_alerts_per_hour": 10
            },
            
            "performance": {
                "target_fps": 30,
                "max_cpu_usage": 80.0,
                "max_memory_mb": 2048,
                "enable_gpu_acceleration": True,
                "enable_tensorrt": True,
                "optimization_level": "balanced"
            },
            
            "logging": {
                "level": "INFO",
                "file_path": "fall_detection.log",
                "max_file_size_mb": 50,
                "backup_count": 5,
                "enable_console": True,
                "enable_file": True
            },
            
            "enable_display": True,
            "display_fps": True,
            "display_detections": True,
            "display_pose": True,
            "save_events": True,
            "events_directory": "events",
            "max_event_files": 1000
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(sample_config, f, indent=2)
            self.logger.info(f"Sample configuration created: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to create sample config: {e}")
    
    def get_env_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables."""
        overrides = {}
        
        # Camera settings
        if os.getenv('CAMERA_SOURCE'):
            overrides['camera.source'] = os.getenv('CAMERA_SOURCE')
        if os.getenv('CAMERA_WIDTH'):
            overrides['camera.width'] = int(os.getenv('CAMERA_WIDTH'))
        if os.getenv('CAMERA_HEIGHT'):
            overrides['camera.height'] = int(os.getenv('CAMERA_HEIGHT'))
        
        # Detection thresholds
        if os.getenv('FALL_CONFIDENCE_THRESHOLD'):
            overrides['fall_detection.confidence_threshold'] = float(os.getenv('FALL_CONFIDENCE_THRESHOLD'))
        
        # Privacy settings
        if os.getenv('PRIVACY_MODE'):
            overrides['privacy.mode'] = os.getenv('PRIVACY_MODE')
        
        # Alert settings
        if os.getenv('EMAIL_ENABLED'):
            overrides['alerts.email_enabled'] = os.getenv('EMAIL_ENABLED').lower() == 'true'
        if os.getenv('EMAIL_USERNAME'):
            overrides['alerts.email_username'] = os.getenv('EMAIL_USERNAME')
        if os.getenv('EMAIL_PASSWORD'):
            overrides['alerts.email_password'] = os.getenv('EMAIL_PASSWORD')
        
        return overrides