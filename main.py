"""
AI Detection System with MQTT, Virtual Camera, and JavaScript Kinesis Integration

Main Application Entry Point

This module provides the main application entry point for the AI Detection System.
It integrates:
- Face recognition using FaceNet
- Object detection using YOLO
- MQTT communication for detection events
- Output to a virtual camera for an external JavaScript application to stream to Kinesis
- Performance monitoring and statistics

Usage:
    python main.py

Configuration is managed through environment variables or a .env file.
See the README.md file for details on configuration options.
"""

import os
import cv2
import time
import signal
import threading # Si lo usas para otras tareas
import argparse
import json
import logging # Asegúrate de que esté importado si setup_logger no lo hace globalmente
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

# NUEVO: Import para subprocess
import subprocess

# Configuration
from config.settings import config

# Utilities
from utils.logging_utils import setup_logging, setup_logger
from utils.mqtt_handler import MQTTHandler
from utils.video_capture import VideoCapture
from utils.virtual_camera_writer import VirtualCamera


# Data handlers
from data_handlers.database import FaceDatabase
from data_handlers.json_logger import JSONLogger

# AI Models
from models.face_recognition import FaceRecognition
from models.object_detection import ObjectDetection

# Setup logger
logger = setup_logger('main')

# NUEVO: Directorio de la aplicación JavaScript
# Ajusta "js_kinesis_streamer" si tu carpeta se llama diferente
# y asegúrate de que la ruta sea correcta respecto a la ubicación de main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JS_APP_DIRECTORY = os.path.join(BASE_DIR, "js_kinesis_streamer")


class AIDetectionSystem:
    """
    Main application class for the AI Detection System.
    
    This class orchestrates all components:
    - Video capture
    - Face recognition
    - Object detection
    - MQTT communication
    - Virtual Camera output
    - JavaScript Kinesis streamer management
    - Performance monitoring
    """
    
    def __init__(self, args):
        """
        Initialize the AI Detection System.
        
        Args:
            args: Command-line arguments
        """
        self.args = args
        
        # Initialize flags
        self.running = False
        self.paused = False
        
        # FPS control
        self.target_fps = config.camera.fps
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = 0
        
        # NUEVO: Para el proceso JavaScript
        self.js_process = None
        
        # Initialize components
        self._init_video_capture()
        self._init_models()
        self._init_data_handlers()
        self._init_mqtt()
        # self._init_kinesis() # COMENTADO: Asumimos que JS maneja Kinesis
        self._init_virtual_camera() # Esto ya lo tenías y es importante

        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("AI Detection System initialized")
    
    def _init_video_capture(self):
        """
        Initialize the video capture component.
        """
        logger.info("Initializing video capture...")
        
        self.video = VideoCapture(
            source=config.camera.source,
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.fps
        )
    
    def _init_models(self):
        """
        Initialize the AI models.
        """
        logger.info("Initializing AI models...")
        
        # Initialize face recognition
        self.face_recognizer = FaceRecognition(
            model_path=config.models.facenet_model_path,
            shape_predictor_path=config.models.shape_predictor_path
        )
        
        # Initialize object detection
        self.object_detector = ObjectDetection(
            model_path=config.models.yolo_model_path,
            confidence_threshold=0.5 # O usa config.models.yolo_confidence_threshold
        )
        
        self.object_detector.set_classes_of_interest(config.models.classes_of_interest)
    
    def _init_data_handlers(self):
        """
        Initialize the data handlers.
        """
        logger.info("Initializing data handlers...")
        
        self.face_db = FaceDatabase(
            db_path=config.models.face_database_path,
            threshold=config.models.face_recognition_threshold if hasattr(config.models, 'face_recognition_threshold') else 0.65
        )
        
        self.json_logger = JSONLogger(
            base_dir='logs', # O usa config.logging.json_log_dir
            face_log_file='personas.json', # O usa config.logging.face_log_filename
            object_log_file='objetos.json' # O usa config.logging.object_log_filename
        )
    
    def _init_virtual_camera(self):
        """
        Initialize the virtual camera output if enabled.
        """
        self.virtual_cam = None # Correcto
        # Tu lógica para leer de config o env var para habilitar la cámara virtual es buena.
        # Usaré config.virtual_camera.enabled y config.virtual_camera.device para consistencia.
        if config.virtual_camera.enabled: # Asumiendo que tienes esto en tu config
            self.virtual_cam = VirtualCamera(
                device_path=config.virtual_camera.device, # Asumiendo config
                width=config.camera.width,
                height=config.camera.height,
                fps=int(config.camera.fps) # Asegurar que fps sea entero
            )
            if self.virtual_cam.is_opened():
                 logger.info(f"Virtual camera enabled and writing to {config.virtual_camera.device}")
            else:
                 logger.error(f"Failed to open virtual camera at {config.virtual_camera.device}. It will be disabled.")
                 self.virtual_cam = None # Deshabilitar si no se pudo abrir
        else:
            logger.info("Virtual camera output is disabled in configuration.")


    
    def _init_mqtt(self):
        """
        Initialize the MQTT handler.
        """
        logger.info("Initializing MQTT handler...")
        # Asumo que config.mqtt.enabled existe
        if config.mqtt.enabled:
            self.mqtt = MQTTHandler(
                broker=config.mqtt.broker,
                port=config.mqtt.port,
                topic=config.mqtt.topic_prefix if hasattr(config.mqtt, 'topic_prefix') else config.mqtt.topic, # topic_prefix es más común
                username=config.mqtt.username,
                password=config.mqtt.password
            )
        else:
            self.mqtt = None
            logger.info("MQTT is disabled in configuration.")
    
    # def _init_kinesis(self):  # COMENTADO
    #     """Initialize Kinesis streamer (Python version)."""
    #     logger.info("Initializing Kinesis streamer (Python)...")
    #     if config.kinesis.enabled: # Asumiendo config.kinesis.enabled
    #         self.kinesis = KinesisStreamer(  
    #             stream_name=config.kinesis.stream_name,  
    #             region=config.kinesis.region,
    #             width=config.camera.width,
    #             height=config.camera.height,
    #             fps=int(config.camera.fps)
    #         )
    #     else:
    #         self.kinesis = None
    #         logger.info("Python Kinesis streamer is disabled in configuration.")
            
    def _start_javascript_streamer(self): # NUEVO MÉTODO
        """
        Starts the JavaScript application for Kinesis streaming.
        """
        if self.js_process and self.js_process.poll() is None:
            logger.info("JavaScript streamer process is already running.")
            return

        if not os.path.isdir(JS_APP_DIRECTORY):
            logger.error(f"JavaScript application directory not found: {JS_APP_DIRECTORY}")
            logger.error("Cannot start JavaScript Kinesis streamer.")
            return

        js_command = ["npm", "run", "develop"]
        
        logger.info(f"Attempting to start JavaScript application from: {JS_APP_DIRECTORY} with command: {' '.join(js_command)}")
        logger.info("IMPORTANT: Ensure 'npm install' has been run in the JS app directory.")
        
        try:
            self.js_process = subprocess.Popen(
                js_command, 
                cwd=JS_APP_DIRECTORY,
                **({'preexec_fn': os.setsid} if os.name != 'nt' else {}) 
            )
            logger.info(f"JavaScript application process started with PID: {self.js_process.pid}")
        except FileNotFoundError:
            logger.error(f"Error: Command 'npm' not found. Ensure Node.js and npm are installed and in the system PATH.")
        except Exception as e:
            logger.error(f"Error starting JavaScript application: {e}", exc_info=True)


    def _signal_handler(self, sig, frame):
        """
        Handle signals (e.g., SIGINT, SIGTERM).
        """
        logger.info(f"Received signal {sig}, shutting down...")
        self.stop() # Llama a tu método stop que ahora también manejará el JS
    
    def _update_fps(self):
        """
        Update FPS calculation.
        """
        self.frame_count += 1
        current_time = time.time()
        # Evitar división por cero si el tiempo no ha avanzado o es el primer cálculo
        elapsed_time = current_time - self.last_fps_time
        if elapsed_time >= 1.0: # Actualizar cada segundo
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def _process_faces(self, frame, detected_faces_from_model):
        """
        Process detected faces. (Tu lógica original, adaptada ligeramente si es necesario)
        """
        annotated_frame = frame.copy()
        recognized_faces_output = []
        
        # Asumo que detected_faces_from_model es la salida directa de self.face_recognizer.process_frame()
        # y que esta salida ya contiene 'bbox' y 'embedding'.
        for face_info in detected_faces_from_model:
            if 'embedding' not in face_info or 'bbox' not in face_info:
                logger.debug(f"Skipping face due to missing embedding or bbox: {face_info}")
                continue
            
            bbox = face_info['bbox']
            embedding = face_info['embedding']
            
            name, similarity = self.face_db.find_match(embedding)
            
            face_data_for_log_mqtt = {
                'name': name,
                'similarity': float(similarity) if similarity is not None else 0.0, # Asegurar que sea serializable
                'bbox': [int(c) for c in bbox] # Asegurar que sean enteros
            }
            recognized_faces_output.append(face_data_for_log_mqtt)
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            text = f"{name} ({similarity:.2f})" if similarity is not None else name
            cv2.putText(annotated_frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if name != "Unknown":
                self.json_logger.log_face_detection(name, similarity)
                if self.mqtt and self.mqtt.is_connected(): # Verificar conexión MQTT
                    self.mqtt.publish({ # Tu estructura de mensaje MQTT original
                        "type": "face", "name": name, "confidence": float(similarity),
                        "timestamp": time.time(), "bbox": [int(c) for c in bbox]
                    })
        
        return annotated_frame, recognized_faces_output
    
    def _process_objects(self, frame, detected_objects_from_model):
        """
        Process detected objects. (Tu lógica original, adaptada ligeramente)
        """
        # Asumo que self.object_detector.draw_detections ya está implementado
        annotated_frame = self.object_detector.draw_detections(frame.copy(), detected_objects_from_model)
        
        objects_output_for_log_mqtt = []
        
        # Asumo que detected_objects_from_model es una lista de diccionarios con 'class', 'confidence', 'bbox'
        for obj_info in detected_objects_from_model:
            obj_class = obj_info.get('class_name', obj_info.get('class', 'N/A')) # Adaptar a la clave real de tu detector
            obj_confidence = obj_info.get('confidence', 0.0)
            obj_bbox = obj_info.get('bbox', [0,0,0,0])

            objects_output_for_log_mqtt.append({
                "class": obj_class,
                "confidence": float(obj_confidence),
                "bbox": [int(c) for c in obj_bbox]
            })

            self.json_logger.log_object_detection([obj_class], [obj_confidence]) # Tu logger original tomaba listas
            if self.mqtt and self.mqtt.is_connected():
                self.mqtt.publish({
                    "type": "object", "class": obj_class, "confidence": float(obj_confidence),
                    "timestamp": time.time(), "bbox": [int(c) for c in obj_bbox]
                })
        
        return annotated_frame, objects_output_for_log_mqtt # Devolver objetos procesados para posible uso futuro
    
    def register_face(self, name: str, embedding: np.ndarray) -> bool:
        """
        Register a new face in the database. (Tu método original)
        """
        return self.face_db.add_face(name, embedding)
    
    def start(self):
        """
        Start the AI Detection System.
        """
        if self.running:
            logger.warning("System is already running")
            return
        
        logger.info("Starting AI Detection System...")
        
        if not self.video.start():
            logger.error("Failed to start video capture. System will not start.")
            return
        
        if self.mqtt and not self.mqtt.is_connected(): # Solo intentar conectar si MQTT está habilitado
            if not self.mqtt.connect(): # connect() debería devolver True/False o manejar excepciones
                logger.warning("Failed to connect to MQTT broker, continuing without MQTT")
        
        # if self.kinesis and not self.kinesis.is_streaming(): # COMENTADO
        #     if not self.kinesis.start():  
        #         logger.warning("Failed to start Kinesis stream (Python), continuing without it")

        # NUEVO: Iniciar el streamer de JavaScript
        self._start_javascript_streamer()
        
        # Activar la cámara virtual (asegurarse de que esté abierta si está habilitada)
        if self.virtual_cam and not self.virtual_cam.is_opened():
            logger.warning("Virtual camera was enabled but failed to open. Output to virtual cam will be skipped.")
            # self.virtual_cam.open() # Podrías intentar reabrirla aquí si tu clase lo permite
            # O simplemente se saltará la escritura si no está abierta

        self.running = True
        self.last_fps_time = time.time() # Inicializar para el cálculo de FPS
        self.frame_count = 0
        
        logger.info("AI Detection System started. Entering main loop...")
        self._main_loop() # Llama a tu bucle principal
    
    def _main_loop(self):
        try:
            while self.running:
                loop_start_time = time.time() # Para control de FPS si es necesario
                
                if self.paused:
                    time.sleep(0.1) # Evitar busy-waiting cuando está pausado
                    continue
                
                frame = self.video.read()
                if frame is None or frame.size == 0:
                    logger.warning("Empty frame received from video source.")
                    # Considerar si es el final de un archivo o un error de cámara
                    if self.video.is_file() and not self.video.is_opened(): # Asumiendo que tu VideoCapture tiene estos métodos
                        logger.info("End of video file reached or video source closed.")
                        self.running = False # Detener si es el final de un archivo
                    time.sleep(0.1) # Esperar un poco antes de reintentar
                    continue
                
                # Procesar el frame (AI y anotaciones)
                processed_frame_for_output = self._process_frame(frame) # Este es el frame que se muestra y envía
                
                self._update_fps() # Actualizar cálculo de FPS
                
                # Enviar frame procesado a la cámara virtual (si está habilitada y abierta)
                if self.virtual_cam and self.virtual_cam.is_opened() and processed_frame_for_output is not None:
                    try:
                        self.virtual_cam.write(processed_frame_for_output) # Tu método se llama 'write' o 'send_frame'? Ajustar si es necesario
                    except Exception as e_vc:
                        logger.error(f"Error writing to virtual camera: {e_vc}")
                
                # Enviar frame a Kinesis (Python version) - COMENTADO
                # if self.kinesis and self.kinesis.is_streaming() and processed_frame_for_output is not None:
                #     self.kinesis.send_frame(processed_frame_for_output)
                
                # Mostrar video si está habilitado
                if self.args.display and processed_frame_for_output is not None:
                    display_frame = processed_frame_for_output.copy()
                    cv2.putText(display_frame, f"FPS: {self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('AI Detection System', display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_key(key) # Manejar entrada de teclado
                
                # Control de FPS para no sobrecargar la CPU (opcional, VideoCapture podría manejarlo)
                # loop_duration = time.time() - loop_start_time
                # sleep_time = (1.0 / self.target_fps) - loop_duration
                # if sleep_time > 0:
                #     time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Critical error in main loop: {e}", exc_info=True)
        finally:
            logger.info("Exiting main loop.")
            # No llamar a self.stop() aquí directamente, dejar que el finally de start() o el signal_handler lo hagan
            # para evitar cierres múltiples si la excepción se propaga.
            # system.start() ya tiene un finally que llama a stop().

    
    def _process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single frame: AI detection, recognition, and annotation.
        """
        try:
            # La copia del frame se hace dentro de _process_faces y _process_objects
            # para que las anotaciones no se acumulen sobre el mismo buffer de frame.
            
            # Detectar caras (asumo que esto devuelve una lista de dicts con 'bbox', 'embedding', etc.)
            # Tu FaceRecognition.process_frame podría hacer detección + extracción de embeddings.
            detected_faces_raw = self.face_recognizer.process_frame(frame) # Ajusta si el método es diferente
            
            # Detectar objetos (asumo que devuelve lista de dicts con 'class', 'confidence', 'bbox')
            detected_objects_raw = self.object_detector.process_frame(frame) # Ajusta si el método es diferente
            
            # Procesar y anotar caras
            # Se pasa el frame original para anotar, y las detecciones crudas
            frame_with_faces, _ = self._process_faces(frame, detected_faces_raw)
            
            # Procesar y anotar objetos sobre el frame ya anotado con caras
            # Se pasa el frame ya anotado con caras, y las detecciones crudas de objetos
            final_annotated_frame, _ = self._process_objects(frame_with_faces, detected_objects_raw)
            
            # Ya no necesitas el FPS aquí si lo pones en _main_loop antes de cv2.imshow
            # cv2.putText(final_annotated_frame, f"FPS: {self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return final_annotated_frame
        
        except Exception as e:
            logger.error(f"Error during frame processing: {e}", exc_info=True)
            return frame # Devolver el frame original en caso de error para no romper el flujo

    def _handle_key(self, key: int):
        """
        Handle keyboard input during display.
        """
        if key == ord('q'):
            logger.info("Quit command ('q') received via display window.")
            self.running = False
        
        elif key == ord('p'):
            self.paused = not self.paused
            logger.info(f"Processing {'paused' if self.paused else 'resumed'} via display window.")
        
        elif key == ord('r'):
            self._handle_face_registration() # Tu lógica de registro
    
    def _handle_face_registration(self): # Tu lógica de registro original
        logger.info("Face registration requested via key 'r'.")
        was_paused = self.paused
        self.paused = True
        # ... (el resto de tu lógica de _handle_face_registration) ...
        # Asegúrate de que las llamadas a cv2.imshow y cv2.waitKey estén dentro de
        # `if self.args.display:` o que se manejen si no hay display.
        # Aquí solo pongo un placeholder para brevedad
        logger.warning("_handle_face_registration logic needs to be fully implemented here based on your original code.")
        # Ejemplo simplificado:
        try:
            frame_for_reg = self.video.read() # Capturar un nuevo frame
            if frame_for_reg is not None:
                # (Tu lógica para detectar la cara más grande, pedir nombre, registrar)
                logger.info("Placeholder: Implement face registration details.")
            else:
                logger.warning("Could not capture frame for registration.")
        except Exception as e:
            logger.error(f"Error during face registration UI: {e}")
        finally:
            self.paused = was_paused # Restaurar estado de pausa
            if self.args.display:
                cv2.destroyWindow('Registration') # Asegurar que se cierre si se abrió
    
    def stop(self):
        """
        Stop the AI Detection System and release resources.
        """
        if not self.running and self.js_process is None: # Si ya está detenido y el JS también, no hacer nada.
            logger.info("System and JS streamer appear to be already stopped.")
            # return # Podrías retornar aquí, o dejar que intente limpiar por si acaso.

        logger.info("Stopping AI Detection System...")
        self.running = False # Importante para detener el _main_loop
        
        # NUEVO: Detener el proceso JavaScript
        if self.js_process:
            logger.info(f"Attempting to stop JavaScript streamer process (PID: {self.js_process.pid})...")
            try:
                if os.name == 'nt':
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.js_process.pid)])
                else:
                    os.killpg(os.getpgid(self.js_process.pid), signal.SIGTERM)
                self.js_process.wait(timeout=10)
                logger.info("JavaScript streamer process terminated gracefully.")
            except ProcessLookupError:
                logger.warning("JavaScript streamer process already terminated.")
            except subprocess.TimeoutExpired:
                logger.warning("JS streamer did not terminate with SIGTERM in time. Forcing kill...")
                try:
                    if os.name == 'nt': pass # taskkill /F ya es forzado
                    else: os.killpg(os.getpgid(self.js_process.pid), signal.SIGKILL)
                    self.js_process.wait(timeout=5)
                    logger.info("JavaScript streamer process killed.")
                except Exception as e_kill:
                    logger.error(f"Error forcing kill on JS streamer: {e_kill}")
            except Exception as e:
                logger.error(f"Error stopping JS streamer: {e}")
            finally:
                self.js_process = None

        # if self.kinesis: # COMENTADO
        #     logger.info("Stopping Python Kinesis streamer...")
        #     self.kinesis.stop()
        
        if self.mqtt and self.mqtt.is_connected():
            logger.info("Disconnecting MQTT handler...")
            self.mqtt.disconnect()
        
        if self.video:
            logger.info("Stopping video capture...")
            self.video.stop() # Asumo que tu VideoCapture tiene un método stop

        if self.virtual_cam and self.virtual_cam.is_opened(): # Solo intentar cerrar si está abierta
            logger.info("Releasing virtual camera...")
            self.virtual_cam.release() # o self.virtual_cam.stop() dependiendo de tu clase

        if self.args.display:
            cv2.destroyAllWindows()
        
        logger.info("AI Detection System stopped.")
    
    def get_stats(self): # Tu método de estadísticas, añadiendo estado de JS
        stats = {
            "system": {"running": self.running, "paused": self.paused, "fps": self.fps, "target_fps": self.target_fps},
            "video": self.video.get_stats() if self.video else None,
            # "kinesis_python": self.kinesis.get_stats() if self.kinesis else None, # COMENTADO
            "js_streamer_running": self.js_process.poll() is None if self.js_process else False, # NUEVO
            "face_recognition": self.face_recognizer.get_performance_stats() if self.face_recognizer else None,
            "object_detection": self.object_detector.get_performance_stats() if self.object_detector else None,
            "face_db": self.face_db.get_stats() if self.face_db else None
        }
        return stats

def parse_args():
    """
    Parse command-line arguments. (Tu función original)
    """
    parser = argparse.ArgumentParser(description='AI Detection System')
    parser.add_argument('--display', action='store_true', help='Display video frames')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default=config.logging.level if hasattr(config.logging, 'level') else 'INFO', # Usa config para default
                       help='Logging level')
    return parser.parse_args()

def main():
    """
    Main entry point.
    """
    args = parse_args()
    
    # Setup logging (ajusta app_name si es necesario)
    setup_logging(log_level=args.log_level, app_name='SKAION_AI_System') # Usa un nombre de app consistente
    
    system = None # Definir system fuera del try para que esté en el scope del finally
    try:
        logger.info("Initializing main AI Detection System object...")
        system = AIDetectionSystem(args)
        system.start() # start() ahora contiene el _main_loop y maneja su propio finally
    except KeyboardInterrupt: # Esto es redundante si _signal_handler llama a stop()
        logger.info("KeyboardInterrupt detected in main. System should be stopping.")
        # El _signal_handler ya debería haber llamado a system.stop()
        # O el finally de system.start() -> _main_loop() -> system.stop()
    except Exception as e:
        logger.critical(f"Unhandled exception at main level: {e}", exc_info=True)
    finally:
        if system: # Asegurar que system se haya inicializado
            logger.info("Main function's finally block: ensuring system shutdown.")
            system.stop() # Llama a stop una vez más para asegurar limpieza si algo falló antes
        logger.info("Application shutdown complete.")

if __name__ == "__main__":
    main()