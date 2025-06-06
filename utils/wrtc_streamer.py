import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import cv2
import numpy as np
import os
import logging

logger = logging.getLogger('kinesis_streamer')

class KinesisStreamer:
    """Clase para transmitir video a AWS Kinesis Video Streams usando GStreamer."""
    
    def __init__(self, stream_name, region, width, height, fps):
        self.stream_name = stream_name
        self.region = region
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.appsrc = None
        self.running = False

        # Inicializar GStreamer
        Gst.init(None)
        self._build_pipeline()

    def _build_pipeline(self):
        """Construye el pipeline de GStreamer para Kinesis."""
        pipeline_str = (
            f"appsrc name=appsrc ! videoconvert ! "
            f"nvv4l2h264enc bitrate=500000 ! "  # 500 kbps
            f"h264parse ! matroskamux ! "
            f"kvssink stream-name={self.stream_name} "
            f"aws-region={self.region} "
            f"access-key={os.getenv('AWS_ACCESS_KEY_ID')} "
            f"secret-key={os.getenv('AWS_SECRET_ACCESS_KEY')}"
        )
        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
            self.appsrc = self.pipeline.get_by_name('appsrc')
            self.appsrc.set_property('is-live', True)
            self.appsrc.set_property('format', Gst.Format.TIME)
        except Exception as e:
            logger.error(f"Error al construir el pipeline de GStreamer: {e}")
            raise

    def start(self):
        """Inicia el streaming a Kinesis."""
        if self.running:
            logger.warning("Kinesis streamer ya está corriendo")
            return True
        try:
            self.pipeline.set_state(Gst.State.PLAYING)
            self.running = True
            logger.info("Kinesis streamer iniciado")
            return True
        except Exception as e:
            logger.error(f"Error al iniciar el Kinesis streamer: {e}")
            return False

    def send_frame(self, frame):
        """Envía un frame al stream de Kinesis."""
        if not self.running:
            return
        
        try:
            # Convertir el frame de BGR (OpenCV) a RGB (GStreamer)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf = Gst.Buffer.new_wrapped(frame_rgb.tobytes())
            self.appsrc.emit('push-buffer', buf)
        except Exception as e:
            logger.error(f"Error al enviar frame a Kinesis: {e}")

    def stop(self):
        """Detiene el streaming a Kinesis."""
        if not self.running:
            return
        try:
            self.pipeline.set_state(Gst.State.NULL)
            self.running = False
            logger.info("Kinesis streamer detenido")
        except Exception as e:
            logger.error(f"Error al detener el Kinesis streamer: {e}")

    def get_stats(self):
        """Devuelve estadísticas del streamer (opcional)."""
        return {
            "running": self.running,
            "stream_name": self.stream_name,
            "region": self.region,
            "resolution": f"{self.width}x{self.height}",
            "fps": self.fps
        }