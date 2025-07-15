import cv2
import threading
import queue
import time
import logging
from typing import Optional, Tuple, Union


class VideoCapture:
    """
    Thread-safe video capture class for RTSP/HTTP streams and local cameras.
    Optimized for Jetson Nano/Xavier NX with hardware acceleration.
    """
    
    def __init__(self, source: Union[str, int], buffer_size: int = 5, 
                 width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize video capture.
        
        Args:
            source: Camera index (int) or stream URL (str)
            buffer_size: Frame buffer size for smooth playback
            width: Target frame width
            height: Target frame height
            fps: Target frames per second
        """
        self.source = source
        self.buffer_size = buffer_size
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.capture_thread = None
        self.running = False
        self.last_frame = None
        self.frame_count = 0
        self.start_time = time.time()
        
        self.logger = logging.getLogger(__name__)
        
    def start(self) -> bool:
        """Start video capture."""
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            # Configure for Jetson hardware acceleration
            if isinstance(self.source, str):  # RTSP/HTTP stream
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set resolution and FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video source: {self.source}")
                return False
                
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            self.logger.info(f"Video capture started: {self.source}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting video capture: {e}")
            return False
    
    def _capture_frames(self):
        """Capture frames in separate thread."""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame")
                    continue
                
                # Clear old frames if buffer is full
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                # Add new frame
                try:
                    self.frame_queue.put_nowait(frame)
                    self.last_frame = frame
                    self.frame_count += 1
                except queue.Full:
                    pass
                    
            except Exception as e:
                self.logger.error(f"Error capturing frame: {e}")
                
    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Read the latest frame.
        
        Returns:
            Tuple of (success, frame)
        """
        try:
            # Try to get the latest frame
            frame = self.frame_queue.get_nowait()
            return True, frame
        except queue.Empty:
            # Return last frame if no new frame available
            if self.last_frame is not None:
                return True, self.last_frame
            return False, None
    
    def get_fps(self) -> float:
        """Get actual capture FPS."""
        if self.frame_count == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get current resolution."""
        if self.cap is not None:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return self.width, self.height
    
    def is_opened(self) -> bool:
        """Check if capture is running."""
        return self.running and self.cap is not None and self.cap.isOpened()
    
    def stop(self):
        """Stop video capture."""
        self.running = False
        
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)
            
        if self.cap is not None:
            self.cap.release()
            
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        self.logger.info("Video capture stopped")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()