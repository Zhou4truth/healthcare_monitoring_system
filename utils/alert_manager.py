import smtplib
import json
import logging
import requests
import threading
import time
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, asdict
import cv2
import numpy as np

from detectors.fall_detector import FallEvent


@dataclass
class AlertConfig:
    """Configuration for alert notifications."""
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = None
    
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = None
    
    sms_enabled: bool = False
    sms_api_key: str = ""
    sms_api_url: str = ""
    sms_recipients: List[str] = None
    
    sound_enabled: bool = True
    sound_file: str = "/usr/share/sounds/alsa/Front_Left.wav"
    
    local_logging: bool = True
    log_file: str = "fall_alerts.log"
    
    cooldown_period: int = 60  # Seconds between alerts for same event
    max_alerts_per_hour: int = 10
    
    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = []
        if self.webhook_headers is None:
            self.webhook_headers = {"Content-Type": "application/json"}
        if self.sms_recipients is None:
            self.sms_recipients = []


class AlertManager:
    """
    Comprehensive alert management system for fall detection events.
    Supports email, webhook, SMS, sound, and local logging notifications.
    """
    
    def __init__(self, config: AlertConfig):
        """
        Initialize alert manager.
        
        Args:
            config: Alert configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Alert tracking
        self.last_alert_time = 0
        self.alerts_this_hour = 0
        self.hour_start = time.time()
        
        # Alert queue for async processing
        self.alert_queue = []
        self.processing_thread = None
        self.running = False
        
        # Custom alert handlers
        self.custom_handlers: List[Callable[[FallEvent], None]] = []
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup local logging for alerts."""
        if self.config.local_logging:
            try:
                # Create alert-specific logger
                alert_logger = logging.getLogger('fall_alerts')
                alert_logger.setLevel(logging.INFO)
                
                # File handler
                file_handler = logging.FileHandler(self.config.log_file)
                file_handler.setLevel(logging.INFO)
                
                # Formatter
                formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(formatter)
                
                alert_logger.addHandler(file_handler)
                
                self.alert_logger = alert_logger
                
            except Exception as e:
                self.logger.error(f"Failed to setup alert logging: {e}")
                self.alert_logger = self.logger
        else:
            self.alert_logger = self.logger
    
    def start(self):
        """Start alert processing thread."""
        if not self.running:
            self.running = True
            self.processing_thread = threading.Thread(target=self._process_alerts)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            self.logger.info("Alert manager started")
    
    def stop(self):
        """Stop alert processing."""
        self.running = False
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=5.0)
        self.logger.info("Alert manager stopped")
    
    def send_alert(self, fall_event: FallEvent, frame: Optional[np.ndarray] = None):
        """
        Send alert for fall event.
        
        Args:
            fall_event: Detected fall event
            frame: Optional frame image to include
        """
        # Check rate limiting
        current_time = time.time()
        
        # Reset hourly counter
        if current_time - self.hour_start >= 3600:
            self.alerts_this_hour = 0
            self.hour_start = current_time
        
        # Check cooldown period
        if current_time - self.last_alert_time < self.config.cooldown_period:
            self.logger.info("Alert suppressed due to cooldown period")
            return
        
        # Check hourly limit
        if self.alerts_this_hour >= self.config.max_alerts_per_hour:
            self.logger.warning("Maximum alerts per hour reached")
            return
        
        # Queue alert for processing
        alert_data = {
            'fall_event': fall_event,
            'frame': frame,
            'timestamp': current_time
        }
        
        self.alert_queue.append(alert_data)
        self.last_alert_time = current_time
        self.alerts_this_hour += 1
        
        self.alert_logger.critical(f"FALL DETECTED - Confidence: {fall_event.confidence:.3f}, "
                                  f"Type: {fall_event.trigger_type}, Severity: {fall_event.severity}")
    
    def _process_alerts(self):
        """Process alerts in background thread."""
        while self.running:
            try:
                if self.alert_queue:
                    alert_data = self.alert_queue.pop(0)
                    self._handle_alert(alert_data)
                else:
                    time.sleep(0.1)  # Short sleep when no alerts
                    
            except Exception as e:
                self.logger.error(f"Error processing alert: {e}")
    
    def _handle_alert(self, alert_data: Dict):
        """Handle individual alert."""
        fall_event = alert_data['fall_event']
        frame = alert_data['frame']
        timestamp = alert_data['timestamp']
        
        # Create alert message
        alert_message = self._create_alert_message(fall_event, timestamp)
        
        # Send via different channels
        if self.config.email_enabled:
            self._send_email_alert(alert_message, fall_event, frame)
        
        if self.config.webhook_enabled:
            self._send_webhook_alert(alert_message, fall_event)
        
        if self.config.sms_enabled:
            self._send_sms_alert(alert_message, fall_event)
        
        if self.config.sound_enabled:
            self._play_alert_sound()
        
        # Call custom handlers
        for handler in self.custom_handlers:
            try:
                handler(fall_event)
            except Exception as e:
                self.logger.error(f"Custom handler error: {e}")
    
    def _create_alert_message(self, fall_event: FallEvent, timestamp: float) -> str:
        """Create formatted alert message."""
        dt = datetime.fromtimestamp(timestamp)
        
        message = f"""
FALL DETECTION ALERT

Time: {dt.strftime('%Y-%m-%d %H:%M:%S')}
Confidence: {fall_event.confidence:.1%}
Detection Type: {fall_event.trigger_type}
Severity: {fall_event.severity.upper()}

Location: Camera coordinates ({fall_event.bbox[0]}, {fall_event.bbox[1]})
Bounding Box: {fall_event.bbox}

Please check on the monitored person immediately.

This is an automated alert from the Edge AI Fall Detection System.
"""
        return message.strip()
    
    def _send_email_alert(self, message: str, fall_event: FallEvent, frame: Optional[np.ndarray]):
        """Send email alert."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.email_username
            msg['To'] = ', '.join(self.config.email_recipients)
            msg['Subject'] = f"URGENT: Fall Detected - Severity {fall_event.severity.upper()}"
            
            # Add text
            msg.attach(MIMEText(message, 'plain'))
            
            # Add image if available
            if frame is not None:
                try:
                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    img_data = buffer.tobytes()
                    
                    # Create image attachment
                    img = MIMEImage(img_data)
                    img.add_header('Content-Disposition', 'attachment; filename="fall_detection.jpg"')
                    msg.attach(img)
                    
                except Exception as e:
                    self.logger.error(f"Failed to attach image: {e}")
            
            # Send email
            with smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port) as server:
                server.starttls()
                server.login(self.config.email_username, self.config.email_password)
                server.send_message(msg)
            
            self.logger.info("Email alert sent successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, message: str, fall_event: FallEvent):
        """Send webhook alert."""
        try:
            # Prepare payload
            payload = {
                'event_type': 'fall_detected',
                'timestamp': fall_event.timestamp,
                'confidence': fall_event.confidence,
                'severity': fall_event.severity,
                'trigger_type': fall_event.trigger_type,
                'bbox': fall_event.bbox,
                'message': message,
                'features': fall_event.pose_features
            }
            
            # Send webhook
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                headers=self.config.webhook_headers,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info("Webhook alert sent successfully")
            else:
                self.logger.error(f"Webhook failed with status {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
    
    def _send_sms_alert(self, message: str, fall_event: FallEvent):
        """Send SMS alert."""
        try:
            # This is a generic SMS implementation
            # You'll need to adapt this to your SMS provider's API
            
            short_message = f"FALL DETECTED - Severity: {fall_event.severity.upper()}, " \
                          f"Confidence: {fall_event.confidence:.1%}, " \
                          f"Time: {datetime.fromtimestamp(fall_event.timestamp).strftime('%H:%M:%S')}"
            
            for recipient in self.config.sms_recipients:
                payload = {
                    'to': recipient,
                    'message': short_message,
                    'api_key': self.config.sms_api_key
                }
                
                response = requests.post(
                    self.config.sms_api_url,
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    self.logger.info(f"SMS sent to {recipient}")
                else:
                    self.logger.error(f"SMS failed for {recipient}: {response.status_code}")
                    
        except Exception as e:
            self.logger.error(f"Failed to send SMS alert: {e}")
    
    def _play_alert_sound(self):
        """Play alert sound."""
        try:
            import subprocess
            import os
            
            if os.path.exists(self.config.sound_file):
                # Try different sound players
                players = ['aplay', 'paplay', 'ffplay']
                
                for player in players:
                    try:
                        subprocess.run([player, self.config.sound_file], 
                                     check=True, capture_output=True, timeout=5)
                        self.logger.info("Alert sound played")
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                        continue
                else:
                    self.logger.warning("No suitable audio player found")
            else:
                self.logger.warning(f"Sound file not found: {self.config.sound_file}")
                
        except Exception as e:
            self.logger.error(f"Failed to play alert sound: {e}")
    
    def add_custom_handler(self, handler: Callable[[FallEvent], None]):
        """
        Add custom alert handler.
        
        Args:
            handler: Function that takes FallEvent as parameter
        """
        self.custom_handlers.append(handler)
        self.logger.info("Custom alert handler added")
    
    def test_alerts(self) -> Dict[str, bool]:
        """
        Test all alert channels.
        
        Returns:
            Dictionary of test results
        """
        results = {}
        
        # Create test fall event
        test_event = FallEvent(
            timestamp=time.time(),
            confidence=0.95,
            trigger_type='test',
            pose_features={},
            bbox=(100, 100, 200, 300),
            severity='high'
        )
        
        # Test email
        if self.config.email_enabled:
            try:
                test_message = "This is a test alert from the Fall Detection System."
                self._send_email_alert(test_message, test_event, None)
                results['email'] = True
            except Exception as e:
                self.logger.error(f"Email test failed: {e}")
                results['email'] = False
        
        # Test webhook
        if self.config.webhook_enabled:
            try:
                test_message = "This is a test alert from the Fall Detection System."
                self._send_webhook_alert(test_message, test_event)
                results['webhook'] = True
            except Exception as e:
                self.logger.error(f"Webhook test failed: {e}")
                results['webhook'] = False
        
        # Test SMS
        if self.config.sms_enabled:
            try:
                test_message = "TEST: Fall Detection System"
                self._send_sms_alert(test_message, test_event)
                results['sms'] = True
            except Exception as e:
                self.logger.error(f"SMS test failed: {e}")
                results['sms'] = False
        
        # Test sound
        if self.config.sound_enabled:
            try:
                self._play_alert_sound()
                results['sound'] = True
            except Exception as e:
                self.logger.error(f"Sound test failed: {e}")
                results['sound'] = False
        
        return results
    
    def get_statistics(self) -> Dict[str, any]:
        """Get alert statistics."""
        return {
            'alerts_this_hour': self.alerts_this_hour,
            'last_alert_time': self.last_alert_time,
            'hour_start': self.hour_start,
            'queue_size': len(self.alert_queue),
            'running': self.running,
            'config': asdict(self.config)
        }