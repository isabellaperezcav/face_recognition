"""
Configuration settings for the AI Detection System.

This module loads and validates environment variables, providing a centralized
configuration management for the application. It handles configuration for:
- MQTT connectivity
- AWS Kinesis streaming
- Camera settings
- AI model paths
- Logging configuration

Usage:
    from config.settings import config
    mqtt_broker = config.mqtt.broker
    kinesis_stream_name = config.kinesis.stream_name
    log_level = config.logging.level
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import logging
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('config')

load_dotenv()

### Configuración de MQTT
@dataclass
class MQTTConfig:
    """MQTT configuration parameters."""
    broker: str
    port: int
    topic: str
    username: str
    password: str
    enabled: bool
    topic_prefix: Optional[str] = None

### Configuración de AWS Kinesis 
@dataclass
class KinesisConfig:
    """AWS Kinesis configuration parameters."""
    stream_name: str
    region: str

### Configuración de la cámara
@dataclass
class CameraConfig:
    """Camera configuration parameters."""
    source: str  # Can be an integer for webcam or RTSP URL for IP camera
    width: int
    height: int
    fps: int

### Configuración de los modelos de IA
@dataclass
class ModelConfig:
    """AI model configuration parameters."""
    facenet_model_path: str
    shape_predictor_path: str
    yolo_model_path: str
    face_database_path: str
    classes_of_interest: list = field(default_factory=list)  # Empty list means detect all classes
    face_recognition_threshold: float = 0.60

@dataclass
class VirtualCameraConfig:
    """Configuration for virtual camera output."""
    enabled: bool
    device: str 

### Configuración de logging
@dataclass
class LoggingConfig:
    """Logging configuration parameters."""
    level: str  # Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### Configuración principal de la aplicación
@dataclass
class AppConfig:
    """Main application configuration."""
    mqtt: MQTTConfig
    kinesis: KinesisConfig
    camera: CameraConfig
    models: ModelConfig
    debug: bool
    logging: LoggingConfig  # Nuevo atributo para configuración de logging
    virtual_camera: VirtualCameraConfig




### Función para cargar la configuración
def load_config() -> AppConfig:
    """
    Load and validate configuration from environment variables.
    
    Returns:
        AppConfig: Application configuration object
        
    Raises:
        ValueError: If required configuration is missing
    """
    # Variables de entorno requeridas
    required_vars = {
        'MQTT': ['MQTT_BROKER', 'MQTT_PORT', 'MQTT_TOPIC', 'MQTT_USER', 'MQTT_PASSWORD'],
        'KINESIS': ['KINESIS_STREAM_NAME', 'KINESIS_REGION'],  
        'CAMERA': ['CAMERA_SOURCE', 'CAMERA_WIDTH', 'CAMERA_HEIGHT', 'CAMERA_FPS'],
        'MODELS': ['FACENET_MODEL_PATH', 'SHAPE_PREDICTOR_PATH', 'YOLO_MODEL_PATH', 'FACE_DB_PATH']
    }
    
    # Verificar variables de entorno faltantes
    missing_vars: Dict[str, List[str]] = {}
    for category, vars_list in required_vars.items():
        missing = [var for var in vars_list if not os.getenv(var)]
        if missing:
            missing_vars[category] = missing
    
    if missing_vars:
        for category, vars_list in missing_vars.items():
            logger.error(f"Missing {category} environment variables: {', '.join(vars_list)}")
        raise ValueError("Missing required environment variables. Please check your .env file.")
    
    # Cargar configuración de MQTT
    mqtt_config = MQTTConfig(
        broker=os.getenv('MQTT_BROKER'),
        port=int(os.getenv('MQTT_PORT')),
        topic=os.getenv('MQTT_TOPIC'),
        username=os.getenv('MQTT_USER'),
        password=os.getenv('MQTT_PASSWORD'),
        enabled=os.getenv('MQTT_ENABLED'),
        topic_prefix=os.getenv('MQTT_TOPIC_PREFIX', None)

    )
    
    # Cargar configuración de Kinesis
    kinesis_config = KinesisConfig(
        stream_name=os.getenv('KINESIS_STREAM_NAME'),
        region=os.getenv('KINESIS_REGION')
    )
    
    # Cargar configuración de la cámara
    try:
        camera_source = os.getenv('CAMERA_SOURCE')
        # Convertir a entero si es un índice numérico (webcam)
        if camera_source.isdigit():
            camera_source = int(camera_source)
        
        camera_config = CameraConfig(
            source=camera_source,
            width=int(os.getenv('CAMERA_WIDTH')),
            height=int(os.getenv('CAMERA_HEIGHT')),
            fps=int(os.getenv('CAMERA_FPS'))
        )
    except (ValueError, AttributeError) as e:
        logger.error(f"Invalid camera configuration: {e}")
        raise ValueError("Invalid camera configuration. Please check your .env file.")
    
    # Cargar rutas de los modelos
    model_config = ModelConfig(
        facenet_model_path=os.getenv('FACENET_MODEL_PATH'),
        shape_predictor_path=os.getenv('SHAPE_PREDICTOR_PATH'),
        yolo_model_path=os.getenv('YOLO_MODEL_PATH'),
        face_database_path=os.getenv('FACE_DB_PATH'),
        face_recognition_threshold=float(os.getenv('FACE_RECOGNITION_THRESHOLD', 0.60))
    )
    
    # Cargar configuración de logging
    logging_config = LoggingConfig(
        level=os.getenv('LOGGING_LEVEL', 'INFO')
    )

        # Configuración de la cámara virtual
    virtual_camera_config = VirtualCameraConfig(
        enabled=os.getenv('VIRTUAL_CAMERA_ENABLED', 'False').lower() in ('true', '1', 'yes'),
        device=os.getenv('VIRTUAL_CAMERA_DEVICE', '/dev/video10')  
    )

    
    # Cargar la bandera de depuración
    debug = os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
    
    # Crear la configuración de la aplicación
    return AppConfig(
        mqtt=mqtt_config,
        kinesis=kinesis_config,
        camera=camera_config,
        models=model_config,
        debug=debug,
        logging=logging_config,  # Incluir configuración de logging
        virtual_camera=virtual_camera_config
    )



### Carga inicial de la configuración
try:
    # Cargar la configuración al importar el módulo
    config = load_config()
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.critical(f"Failed to load configuration: {e}")
    sys.exit(1)