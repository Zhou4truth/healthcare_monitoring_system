#!/usr/bin/env python3
"""
Edge AI Fall Detection System
Real-time fall detection using YOLOv8, MediaPipe, and LSTM with privacy protection.
Optimized for NVIDIA Jetson Nano/Xavier NX.
"""

import cv2
import time
import logging
import argparse
import signal
import sys
from pathlib import Path
from typing import Optional

# Import system components
from utils.config import ConfigManager
from utils.video_capture import VideoCapture
from utils.alert_manager import AlertManager
from detectors.person_detector import PersonDetector
from detectors.pose_estimator import PoseEstimator
from detectors.privacy_filter import BodySegmenter
from detectors.fall_detector import FallDetector


class FallDetectionSystem:
    """
    Main fall detection system integrating all components.
    """
    
    def __init__(self, config_file: str = "config.json"):
        """Initialize the fall detection system."""
        # Load configuration
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.get_config()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.video_capture = None
        self.person_detector = None
        self.pose_estimator = None
        self.body_segmenter = None
        self.fall_detector = None
        self.alert_manager = None
        
        # Runtime state
        self.running = False
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Performance monitoring
        self.processing_times = {
            'total': 0.0,
            'person_detection': 0.0,
            'pose_estimation': 0.0,
            'fall_detection': 0.0,
            'privacy_filter': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Fall Detection System initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.logging
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, log_config.level))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        if log_config.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_config.level))
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_config.enable_file:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_config.file_path,
                maxBytes=log_config.max_file_size_mb * 1024 * 1024,
                backupCount=log_config.backup_count
            )
            file_handler.setLevel(getattr(logging, log_config.level))
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    def initialize_components(self) -> bool:
        """Initialize all system components."""
        try:
            # Initialize video capture
            self.logger.info("Initializing video capture...")
            self.video_capture = VideoCapture(
                source=self.config.camera.source,
                buffer_size=self.config.camera.buffer_size,
                width=self.config.camera.width,
                height=self.config.camera.height,
                fps=self.config.camera.fps
            )
            
            # Initialize person detector
            self.logger.info("Initializing person detector...")
            self.person_detector = PersonDetector(
                model_path=self.config.person_detection.model_path,
                confidence_threshold=self.config.person_detection.confidence_threshold,
                iou_threshold=self.config.person_detection.iou_threshold,
                use_tensorrt=self.config.person_detection.use_tensorrt
            )
            
            # Initialize pose estimator
            self.logger.info("Initializing pose estimator...")
            self.pose_estimator = PoseEstimator(
                model_complexity=self.config.pose_estimation.model_complexity,
                min_detection_confidence=self.config.pose_estimation.min_detection_confidence,
                min_tracking_confidence=self.config.pose_estimation.min_tracking_confidence,
                smooth_landmarks=self.config.pose_estimation.smooth_landmarks
            )
            
            # Initialize body segmenter for privacy
            if self.config.privacy.enabled:
                self.logger.info("Initializing privacy protection...")
                self.body_segmenter = BodySegmenter(
                    model_selection=self.config.privacy.model_selection,
                    blur_strength=self.config.privacy.blur_strength
                )
                self.body_segmenter.set_privacy_mode(self.config.privacy.mode)
            
            # Initialize fall detector
            self.logger.info("Initializing fall detector...")
            self.fall_detector = FallDetector(
                sequence_length=self.config.fall_detection.sequence_length,
                model_path=self.config.fall_detection.model_path,
                confidence_threshold=self.config.fall_detection.confidence_threshold,
                rule_weight=self.config.fall_detection.rule_weight,
                lstm_weight=self.config.fall_detection.lstm_weight
            )
            
            # Initialize alert manager
            self.logger.info("Initializing alert manager...")
            self.alert_manager = AlertManager(self.config.alerts)
            self.alert_manager.start()
            
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            return False
    
    def start(self) -> bool:
        """Start the fall detection system."""
        if not self.initialize_components():
            return False
        
        if not self.video_capture.start():
            self.logger.error("Failed to start video capture")
            return False
        
        self.running = True
        self.logger.info("Fall Detection System started")
        
        # Create events directory if saving events
        if self.config.save_events:
            Path(self.config.events_directory).mkdir(exist_ok=True)
        
        return True
    
    def stop(self):
        """Stop the fall detection system."""
        self.running = False
        
        if self.video_capture:
            self.video_capture.stop()
        
        if self.alert_manager:
            self.alert_manager.stop()
        
        self.logger.info("Fall Detection System stopped")
    
    def process_frame(self, frame):
        """Process a single frame through the detection pipeline."""
        start_time = time.time()
        
        # Person detection
        person_start = time.time()
        person_detections = self.person_detector.detect_persons(frame)
        
        # Filter detections by size
        person_detections = self.person_detector.filter_by_size(
            person_detections,
            min_area=self.config.person_detection.min_person_area,
            max_area=self.config.person_detection.max_person_area
        )
        
        # Get the largest person for tracking
        main_person = self.person_detector.get_largest_person(person_detections)
        self.processing_times['person_detection'] = time.time() - person_start
        
        pose_landmarks = None
        fall_event = None
        
        if main_person is not None:
            # Pose estimation
            pose_start = time.time()
            pose_landmarks = self.pose_estimator.estimate_pose(frame, main_person)
            self.processing_times['pose_estimation'] = time.time() - pose_start
            
            if pose_landmarks and self.pose_estimator.is_valid_pose(pose_landmarks):
                # Fall detection
                fall_start = time.time()
                fall_event = self.fall_detector.analyze_frame(pose_landmarks)
                self.processing_times['fall_detection'] = time.time() - fall_start
                
                # Send alert if fall detected
                if fall_event:
                    self.alert_manager.send_alert(fall_event, frame)
                    
                    # Save event if configured
                    if self.config.save_events:
                        self._save_fall_event(fall_event, frame)
        
        # Apply privacy filter
        privacy_start = time.time()
        if self.config.privacy.enabled and self.body_segmenter:
            if self.config.privacy.anonymize_output:
                frame = self.body_segmenter.create_anonymized_frame(frame, pose_landmarks)
            else:
                frame = self.body_segmenter.apply_privacy_filter(frame, pose_landmarks)
        self.processing_times['privacy_filter'] = time.time() - privacy_start
        
        # Draw visualizations
        if self.config.enable_display:
            frame = self._draw_visualizations(frame, person_detections, pose_landmarks, fall_event)
        
        self.processing_times['total'] = time.time() - start_time
        return frame
    
    def _draw_visualizations(self, frame, person_detections, pose_landmarks, fall_event):
        """Draw detection visualizations on frame."""
        # Draw person detections
        if self.config.display_detections and person_detections:
            frame = self.person_detector.draw_detections(frame, person_detections)
        
        # Draw pose landmarks
        if self.config.display_pose and pose_landmarks:
            frame = self.pose_estimator.draw_pose(frame, pose_landmarks)
        
        # Draw fall alert
        if fall_event:
            cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 255), -1)
            cv2.putText(frame, "FALL DETECTED!", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {fall_event.confidence:.1%}", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw FPS and performance info
        if self.config.display_fps:
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                       (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw processing times
            y_offset = 60
            for name, proc_time in self.processing_times.items():
                if proc_time > 0:
                    cv2.putText(frame, f"{name}: {proc_time*1000:.1f}ms", 
                               (frame.shape[1] - 200, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    y_offset += 20
        
        return frame
    
    def _save_fall_event(self, fall_event, frame):
        """Save fall event to disk."""
        try:
            timestamp = int(fall_event.timestamp)
            event_dir = Path(self.config.events_directory) / f"fall_{timestamp}"
            event_dir.mkdir(exist_ok=True)
            
            # Save frame
            frame_path = event_dir / "frame.jpg"
            cv2.imwrite(str(frame_path), frame)
            
            # Save event data
            import json
            event_data = {
                'timestamp': fall_event.timestamp,
                'confidence': fall_event.confidence,
                'trigger_type': fall_event.trigger_type,
                'severity': fall_event.severity,
                'bbox': fall_event.bbox,
                'pose_features': fall_event.pose_features
            }
            
            event_file = event_dir / "event.json"
            with open(event_file, 'w') as f:
                json.dump(event_data, f, indent=2)
            
            self.logger.info(f"Fall event saved to {event_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save fall event: {e}")
    
    def _update_fps(self):
        """Update FPS calculation."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run(self):
        """Main processing loop."""
        if not self.start():
            return False
        
        self.logger.info("Starting main processing loop...")
        
        try:
            while self.running:
                # Read frame
                ret, frame = self.video_capture.read()
                if not ret or frame is None:
                    continue
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame if enabled
                if self.config.enable_display:
                    cv2.imshow('Fall Detection System', processed_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):  # Reset fall detector
                        self.fall_detector.reset_state()
                        self.logger.info("Fall detector state reset")
                    elif key == ord('t'):  # Test alerts
                        self.logger.info("Testing alert system...")
                        results = self.alert_manager.test_alerts()
                        self.logger.info(f"Alert test results: {results}")
                
                # Update counters
                self.frame_count += 1
                self._update_fps()
                
                # Performance monitoring
                if self.frame_count % 100 == 0:
                    self.logger.info(f"Processed {self.frame_count} frames, "
                                   f"Current FPS: {self.current_fps:.1f}")
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.stop()
            if self.config.enable_display:
                cv2.destroyAllWindows()
        
        return True


def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    logging.getLogger(__name__).info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Edge AI Fall Detection System")
    parser.add_argument('--config', '-c', default='config.json',
                       help='Configuration file path')
    parser.add_argument('--create-config', action='store_true',
                       help='Create sample configuration file')
    parser.add_argument('--test-alerts', action='store_true',
                       help='Test alert system and exit')
    parser.add_argument('--validate-config', action='store_true',
                       help='Validate configuration and exit')
    parser.add_argument('--camera', help='Override camera source')
    parser.add_argument('--privacy-mode', choices=['minimal', 'standard', 'maximum', 'emergency'],
                       help='Override privacy mode')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display output')
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create sample configuration
    if args.create_config:
        config_manager = ConfigManager()
        config_manager.create_sample_config("sample_config.json")
        print("Sample configuration created: sample_config.json")
        return
    
    # Initialize system
    try:
        system = FallDetectionSystem(args.config)
        
        # Apply command line overrides
        if args.camera is not None:
            system.config.camera.source = args.camera
        if args.privacy_mode is not None:
            system.config.privacy.mode = args.privacy_mode
        if args.no_display:
            system.config.enable_display = False
        
        # Validate configuration
        if args.validate_config:
            errors = system.config_manager.validate_config()
            if errors:
                print("Configuration validation errors:")
                for error in errors:
                    print(f"  - {error}")
                return 1
            else:
                print("Configuration is valid")
                return 0
        
        # Test alerts
        if args.test_alerts:
            if not system.initialize_components():
                print("Failed to initialize components")
                return 1
            
            results = system.alert_manager.test_alerts()
            print("Alert test results:")
            for channel, success in results.items():
                status = "✓" if success else "✗"
                print(f"  {status} {channel}")
            return 0
        
        # Run main system
        print("Starting Edge AI Fall Detection System...")
        print("Press 'q' to quit, 'r' to reset fall detector, 't' to test alerts")
        
        success = system.run()
        return 0 if success else 1
        
    except Exception as e:
        logging.getLogger(__name__).error(f"System error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())