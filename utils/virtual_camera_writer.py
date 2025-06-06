import os
import cv2
import logging

logger = logging.getLogger(__name__)

class VirtualCamera:
    def __init__(self, device='/dev/video2', width=640, height=480, fps=30):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.writer = None
        self.enabled = os.path.exists(self.device)

        if self.enabled:
            logger.info(f"Virtual camera found at {self.device}, initializing...")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.device, fourcc, self.fps, (self.width, self.height))
            if not self.writer.isOpened():
                logger.error(f"Failed to open virtual camera at {self.device}")
                self.enabled = False
                self.writer = None
        else:
            logger.warning(f"Virtual camera device {self.device} not found.")

    def write(self, frame):
        """Write frame to virtual camera (alias of send_frame)."""
        self.send_frame(frame)

    def send_frame(self, frame):
        """Send a frame to the virtual camera."""
        if self.enabled and self.writer:
            try:
                self.writer.write(frame)
            except Exception as e:
                logger.error(f"Failed to write frame to virtual camera: {e}")
        else:
            logger.debug("Virtual camera not ready, skipping frame.")

    def is_opened(self):
        """Check if the virtual camera is ready to receive frames."""
        return self.enabled and self.writer is not None and self.writer.isOpened()

    def stop(self):
        """Release the video writer."""
        if self.writer:
            self.writer.release()
            self.writer = None
            logger.info("Virtual camera writer released.")

    def release(self):
        """Alias for stop()."""
        self.stop()
