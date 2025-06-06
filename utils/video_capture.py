"""
Video Capture module for the AI Detection System.

This module provides the VideoCapture class, which enhances OpenCV's
video capture functionality with:
- Automatic reconnection for network cameras
- Frame rate control and monitoring
- Resolution control
- Robust error handling
- Thread-safe frame access

Usage:
    from utils.video_capture import VideoCapture
    
    # Create a video capture from webcam (index 0)
    cap = VideoCapture(source=0, width=640, height=480, fps=30)
    
    # Start capturing
    cap.start()
    
    # Get the latest frame
    frame = cap.read()
    
    # Stop capturing when done
    cap.stop()
"""

import cv2
import time
import threading
import logging
import numpy as np
from typing import Union, Optional, Tuple, Any

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('video')

class VideoCapture:
    """
    Enhanced video capture class with automatic reconnection and performance monitoring.
    
    This class wraps OpenCV's VideoCapture with additional features:
    - Runs in a separate thread to avoid blocking the main application
    - Provides automatic reconnection for network cameras
    - Controls and monitors frame rate
    - Handles resolution settings
    - Provides robust error handling
    """
    
    def __init__(
        self, 
        source: Union[int, str], 
        width: int = 640, 
        height: int = 480, 
        fps: int = 30,
        buffer_size: int = 1
    ):
        """
        Initialize the video capture.
        
        Args:
            source: The video source (camera index or URL)
            width: The desired frame width
            height: The desired frame height
            fps: The desired frame rate
            buffer_size: Size of the frame buffer (use 1 for lowest latency)
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = max(1, buffer_size)
        
        self.cap = None
        self.frame_buffer = []
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = 0
        self.current_fps = 0
        self.reconnect_count = 0
        self.last_frame_time = 0
        self.empty_reads = 0
        
        logger.info(f"Video Capture initialized for source {source} at {width}x{height} {fps}fps")
    
    def open_camera(self) -> bool:
        """
        Open the camera with the specified settings.
        
        Returns:
            bool: True if the camera was opened successfully, False otherwise
        """
        try:
            # Create a new capture object
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"Camera opened: requested {self.width}x{self.height} at {self.fps}fps, "
                       f"got {actual_width}x{actual_height} at {actual_fps}fps")
            
            return True
        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False
    
    def start(self) -> bool:
        """
        Start the video capture thread.
        
        Returns:
            bool: True if the capture was started successfully, False otherwise
        """
        if self.running:
            logger.warning("Video capture is already running")
            return True
        
        # Try to open the camera
        if not self.open_camera():
            logger.error("Failed to start video capture: could not open camera")
            return False
        
        # Initialize the frame buffer
        with self.lock:
            black_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.frame_buffer = [black_frame] * self.buffer_size
        
        # Start the capture thread
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        self.reconnect_count = 0
        self.empty_reads = 0
        
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Video capture started")
        return True
    
    def _capture_loop(self):
        """
        Main capture loop that runs in a separate thread.
        """
        frame_interval = 1.0 / self.fps
        next_frame_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Maintain desired frame rate by sleeping if necessary
            sleep_time = next_frame_time - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Calculate the next frame time
            next_frame_time = max(current_time, next_frame_time) + frame_interval
            
            # Try to read a frame
            if self.cap is None or not self.cap.isOpened():
                self._handle_camera_error()
                continue
            
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None or frame.size == 0:
                    self.empty_reads += 1
                    
                    # Log warning for frequent empty reads
                    if self.empty_reads % 10 == 1:
                        logger.warning(f"Empty frame received ({self.empty_reads} consecutive empty reads)")
                    
                    # Try to reconnect after multiple empty reads
                    if self.empty_reads >= 30:
                        self._handle_camera_error()
                    
                    continue
                
                # Reset empty reads counter on successful read
                self.empty_reads = 0
                
                # Update the frame buffer (thread-safe)
                with self.lock:
                    # For buffer_size > 1, we keep a history of frames
                    if self.buffer_size > 1:
                        self.frame_buffer.pop(0)
                        self.frame_buffer.append(frame.copy())
                    else:
                        self.frame_buffer[0] = frame.copy()
                
                # Update performance metrics
                self.frame_count += 1
                self.last_frame_time = current_time
                
                # Update FPS calculation every second
                elapsed = current_time - self.start_time
                if elapsed >= 1.0:
                    self.current_fps = self.frame_count / elapsed
                    if self.frame_count % 30 == 0:
                        logger.debug(f"Video capture running at {self.current_fps:.2f} FPS")
                    
                    # Reset counters for next interval
                    self.start_time = current_time
                    self.frame_count = 0
            
            except Exception as e:
                logger.error(f"Error capturing frame: {e}")
                self._handle_camera_error()
    
    def _handle_camera_error(self):
        """
        Handle camera errors by attempting to reconnect.
        """
        # Release the current camera if it exists
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            finally:
                self.cap = None
        
        self.reconnect_count += 1
        logger.warning(f"Attempting to reconnect to camera (attempt {self.reconnect_count})")
        
        # Wait before reconnecting
        time.sleep(1.0)
        
        # Try to reopen the camera
        self.open_camera()
    
    def read(self) -> np.ndarray:
        """
        Read the latest frame from the video capture.
        
        Returns:
            numpy.ndarray: The latest frame, or a black frame if no frames are available
        """
        with self.lock:
            # Return the most recent frame in the buffer
            if self.frame_buffer:
                return self.frame_buffer[-1].copy()
            else:
                # Return a black frame if the buffer is empty
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def read_buffer(self) -> list:
        """
        Read all frames in the buffer.
        
        Returns:
            list: A list of frames in the buffer
        """
        with self.lock:
            return [frame.copy() for frame in self.frame_buffer]
    
    def get_stats(self) -> dict:
        """
        Get capture statistics.
        
        Returns:
            dict: Dictionary with capture statistics
        """
        current_time = time.time()
        latency = current_time - self.last_frame_time if self.last_frame_time > 0 else 0
        
        stats = {
            "running": self.running,
            "fps": self.current_fps,
            "target_fps": self.fps,
            "frame_latency": f"{latency * 1000:.1f}ms",
            "reconnect_count": self.reconnect_count,
            "empty_reads": self.empty_reads,
            "resolution": f"{self.width}x{self.height}",
            "source": self.source,
            "buffer_size": self.buffer_size
        }
        
        return stats
    
    def stop(self):
        """
        Stop the video capture and clean up resources.
        """
        if not self.running:
            return
        
        self.running = False
        logger.info("Stopping video capture...")
        
        # Wait for the capture thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        # Release the camera
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
            finally:
                self.cap = None
        
        # Log final statistics
        stats = self.get_stats()
        logger.info(f"Video capture stopped. Final statistics: {stats}")