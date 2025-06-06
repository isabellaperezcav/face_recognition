"""
RTSP Streamer for the AI Detection System.

This module provides the RTSPStreamer class, which implements reliable
RTSP streaming using ffmpeg. It includes features such as:
- Asynchronous frame processing
- Automatic recovery from streaming errors
- Performance optimization with frame queues
- Comprehensive logging

Usage:
    from utils.rtsp_streamer import RTSPStreamer
    
    # Create streamer
    streamer = RTSPStreamer(rtsp_url="rtsp://user:pass@ip:port/path", 
                           width=640, height=480, fps=30)
    
    # Start streaming
    streamer.start()
    
    # Send frames
    while True:
        ret, frame = camera.read()
        streamer.send_frame(frame)
    
    # Stop streaming when done
    streamer.stop()
"""

import os
import time
import threading
import subprocess
import logging
from queue import Queue, Full
from typing import Optional, Tuple, List
import numpy as np

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rtsp')

class RTSPStreamer:
    """
    Handles RTSP streaming using ffmpeg with robust error handling and recovery.
    
    This class implements efficient RTSP streaming by:
    - Using a separate thread for ffmpeg communication
    - Buffering frames in a queue to prevent blocking
    - Providing automatic recovery from streaming errors
    - Optimizing for performance and reliability
    """
    
    def __init__(self, 
                rtsp_url: str, 
                width: int, 
                height: int, 
                fps: int = 30,
                queue_size: int = 10,
                ffmpeg_path: Optional[str] = None):
        """
        Initialize the RTSP streamer.
        
        Args:
            rtsp_url: The RTSP URL to stream to
            width: The width of the video stream
            height: The height of the video stream
            fps: The frames per second of the video stream
            queue_size: Size of the frame buffer queue
            ffmpeg_path: Path to ffmpeg executable (optional)
        """
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.queue_size = queue_size
        self.ffmpeg_path = ffmpeg_path if ffmpeg_path else 'ffmpeg'
        
        self.process = None
        self.frame_queue = Queue(maxsize=queue_size)
        self.running = False
        self.stream_thread = None
        self.monitor_thread = None
        self.last_frame_time = 0
        self.frames_sent = 0
        self.dropped_frames = 0
        
        logger.info(f"RTSP Streamer initialized for {rtsp_url} at {width}x{height} {fps}fps")
    
    def check_ffmpeg(self) -> bool:
        """
        Check if ffmpeg is available.
        
        Returns:
            bool: True if ffmpeg is available, False otherwise
        """
        try:
            cmd = [self.ffmpeg_path, '-version']
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                check=False
            )
            if result.returncode == 0:
                ffmpeg_version = result.stdout.decode('utf-8').split('\n')[0]
                logger.info(f"Found {ffmpeg_version}")
                return True
            else:
                logger.error(f"ffmpeg check failed with return code {result.returncode}")
                return False
        except FileNotFoundError:
            logger.error(f"ffmpeg not found at path: {self.ffmpeg_path}")
            return False
        except Exception as e:
            logger.error(f"Error checking ffmpeg: {e}")
            return False
    
    def build_ffmpeg_command(self) -> List[str]:
        """
        Build the ffmpeg command for RTSP streaming.
        
        Returns:
            List[str]: The ffmpeg command as a list of strings
        """
        return [
            self.ffmpeg_path,
            '-y',  # Overwrite output file without asking
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f"{self.width}x{self.height}",
            '-r', f"{self.fps}",
            '-i', '-',  # Read from stdin
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-b:v', '1000k',  # Bitrate
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',  # More reliable than UDP
            self.rtsp_url
        ]
    
    def start(self) -> bool:
        """
        Start the RTSP stream.
        
        Returns:
            bool: True if streaming started successfully, False otherwise
        """
        if self.running:
            logger.warning("RTSP Streamer is already running")
            return True
        
        if not self.check_ffmpeg():
            logger.error("Cannot start RTSP stream: ffmpeg not available")
            return False
        
        command = self.build_ffmpeg_command()
        
        try:
            logger.info(f"Starting RTSP stream: {' '.join(command)}")
            self.process = subprocess.Popen(
                command, 
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            
            self.running = True
            self.frames_sent = 0
            self.dropped_frames = 0
            
            # Start the streaming thread
            self.stream_thread = threading.Thread(
                target=self._stream_worker,
                name="RTSPStreamWorker"
            )
            self.stream_thread.daemon = True
            self.stream_thread.start()
            
            # Start the monitoring thread
            self.monitor_thread = threading.Thread(
                target=self._monitor_worker,
                name="RTSPMonitorWorker"
            )
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            logger.info(f"RTSP stream started: {self.rtsp_url}")
            return True
        except Exception as e:
            logger.error(f"Error starting RTSP stream: {e}")
            self.running = False
            return False
    
    def _stream_worker(self):
        """
        Worker thread that sends frames from the queue to ffmpeg.
        """
        logger.info("RTSP streaming thread started")
        
        while self.running and self.process and self.process.poll() is None:
            try:
                # Get a frame from the queue with timeout
                frame = self.frame_queue.get(timeout=1.0)
                
                # Track frame timestamp for FPS calculation
                self.last_frame_time = time.time()
                
                try:
                    # Send frame to ffmpeg
                    self.process.stdin.write(frame.tobytes())
                    self.frames_sent += 1
                    self.frame_queue.task_done()
                except (BrokenPipeError, IOError) as e:
                    logger.error(f"RTSP streaming error: {e}")
                    self.running = False
                    break
            except Exception:
                # Queue timeout or other error, just continue
                pass
        
        logger.info("RTSP streaming thread stopped")
    
    def _monitor_worker(self):
        """
        Worker thread that monitors the ffmpeg process and restarts it if necessary.
        """
        logger.info("RTSP monitoring thread started")
        
        restart_attempts = 0
        max_restart_attempts = 5
        last_restart_time = 0
        restart_cooldown = 30  # seconds
        
        while self.running:
            time.sleep(1.0)
            
            # Check if process is still running
            if self.process and self.process.poll() is not None:
                logger.warning(f"ffmpeg process exited with code {self.process.poll()}")
                
                # Check if we should attempt restart
                current_time = time.time()
                if (current_time - last_restart_time) > restart_cooldown:
                    restart_attempts = 0
                
                if restart_attempts < max_restart_attempts:
                    logger.info(f"Attempting to restart RTSP stream (attempt {restart_attempts + 1}/{max_restart_attempts})")
                    
                    # Clean up old process
                    if self.process:
                        try:
                            self.process.stdin.close()
                        except:
                            pass
                        self.process = None
                    
                    # Start new process
                    command = self.build_ffmpeg_command()
                    try:
                        self.process = subprocess.Popen(
                            command, 
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            bufsize=10**8
                        )
                        logger.info("RTSP stream restarted successfully")
                    except Exception as e:
                        logger.error(f"Failed to restart RTSP stream: {e}")
                    
                    restart_attempts += 1
                    last_restart_time = current_time
                else:
                    logger.error(f"Maximum restart attempts ({max_restart_attempts}) reached. Stopping RTSP stream.")
                    self.running = False
                    break
        
        logger.info("RTSP monitoring thread stopped")
    
    def send_frame(self, frame: np.ndarray) -> bool:
        """
        Send a frame to the RTSP stream.
        
        Args:
            frame: The frame to send (numpy array)
            
        Returns:
            bool: True if the frame was sent successfully, False otherwise
        """
        if not self.running:
            return False
        
        if frame is None or frame.size == 0:
            logger.warning("Cannot send empty frame to RTSP stream")
            return False
        
        # Check frame dimensions
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            logger.warning(f"Frame dimensions ({frame.shape[1]}x{frame.shape[0]}) "
                          f"don't match configured dimensions ({self.width}x{self.height})")
            # You could resize the frame here, but it's usually better to handle this upstream
        
        try:
            # Non-blocking put with timeout
            self.frame_queue.put(frame, block=True, timeout=0.1)
            return True
        except Full:
            # Queue is full, drop the frame
            self.dropped_frames += 1
            
            # Log periodic warnings about dropped frames
            if self.dropped_frames % 100 == 1:
                logger.warning(f"RTSP frame queue full, dropped {self.dropped_frames} frames")
            return False
        except Exception as e:
            logger.error(f"Error sending frame to RTSP queue: {e}")
            return False
    
    def get_stats(self) -> dict:
        """
        Get streaming statistics.
        
        Returns:
            dict: Dictionary with streaming statistics
        """
        current_time = time.time()
        stream_duration = current_time - self.last_frame_time if self.last_frame_time > 0 else 0
        
        queue_size = self.frame_queue.qsize()
        queue_fullness = queue_size / self.queue_size if self.queue_size > 0 else 0
        
        stats = {
            "running": self.running,
            "frames_sent": self.frames_sent,
            "dropped_frames": self.dropped_frames,
            "queue_size": queue_size,
            "queue_fullness": f"{queue_fullness:.2%}",
            "stream_fps": self.fps,
            "stream_resolution": f"{self.width}x{self.height}",
            "stream_url": self.rtsp_url
        }
        
        return stats
    
    def stop(self):
        """
        Stop the RTSP stream and clean up resources.
        """
        if not self.running:
            logger.info("RTSP Streamer is not running")
            return
        
        self.running = False
        logger.info("Stopping RTSP stream...")
        
        # Wait for streaming thread to finish
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2.0)
        
        # Wait for monitoring thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        # Terminate the ffmpeg process
        if self.process:
            try:
                logger.info("Terminating ffmpeg process...")
                self.process.stdin.close()
                self.process.terminate()
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    logger.warning("ffmpeg process did not terminate gracefully, forcing kill")
                    self.process.kill()
            except Exception as e:
                logger.error(f"Error stopping ffmpeg process: {e}")
        
        # Clear the frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.task_done()
            except:
                pass
        
        logger.info("RTSP stream stopped")
        
        # Log final statistics
        stats = self.get_stats()
        logger.info(f"RTSP streaming statistics: {stats}")