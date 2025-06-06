# AI Detection System with MQTT and RTSP Integration

*[English](#english-documentation) | [Español](#documentación-en-español)*

---

## English Documentation

### Overview

The AI Detection System is a computer vision application that combines face recognition and object detection capabilities with MQTT messaging and RTSP streaming. The system is designed to provide reliable surveillance and monitoring functionality with the following key features:

- **Face Recognition**: Detect and identify faces using FaceNet embeddings
- **Object Detection**: Detect and classify objects using YOLO
- **MQTT Integration**: Send detection events to an MQTT broker
- **RTSP Streaming**: Stream processed video via RTSP
- **Robust Error Handling**: Handle network disruptions and camera failures gracefully
- **Performance Monitoring**: Track and report system performance metrics

### System Architecture

The system is built with a modular architecture, making it easy to maintain and extend:

```
ai_detection_system/
├── config/                # Configuration management
├── models/                # AI models (face recognition, object detection)
├── utils/                 # Utility modules (MQTT, RTSP, video capture)
├── data_handlers/         # Data storage and retrieval
├── main.py                # Main application entry point
└── requirements.txt       # Dependencies
```

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster inference)
- Camera (webcam or IP camera)
- MQTT broker (e.g., Mosquitto)
- RTSP server (e.g., ffmpeg)

Required Python packages are listed in `requirements.txt`.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai_detection_system.git
   cd ai_detection_system
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install additional dependencies:
   - ffmpeg (for RTSP streaming)
   - CUDA and cuDNN (optional, for GPU acceleration)

5. Configure the application:
   ```bash
   cp .env.template .env
   ```
   Edit the `.env` file with your configuration values.

6. Prepare the models:
   - Download the shape predictor file from dlib
   - Download or train the FaceNet model
   - Download a YOLO model (e.g., YOLOv10m)
   
   Place these models in the appropriate directories as configured in your `.env` file.

### Usage

Run the application:

```bash
python main.py
```

Command-line options:
- `--display`: Display video frames in a window
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

While the system is running:
- Press 'q' to quit
- Press 'p' to pause/resume processing
- Press 'r' to register a new face

### MQTT Message Format

The system publishes detection events to the configured MQTT topic in JSON format:

**Face Detection**:
```json
{
  "type": "face",
  "name": "John",
  "confidence": 0.92,
  "timestamp": 1679401235.765
}
```

**Object Detection**:
```json
{
  "type": "objects",
  "classes": ["person", "car"],
  "confidences": [0.98, 0.87],
  "timestamp": 1679401236.123
}
```

### RTSP Streaming

The processed video stream is available at the configured RTSP URL:
```
rtsp://username:password@ip:port/path
```

You can view this stream using any RTSP-compatible player (e.g., VLC).

### Face Registration

To register new faces in the system:
1. Press 'r' when a face is visible in the camera
2. Enter the person's name when prompted
3. The face will be added to the database for future recognition

### Troubleshooting

**System doesn't detect the camera**:
- Check if the camera is properly connected
- Verify that the camera index or URL is correct in the `.env` file

**MQTT connection fails**:
- Ensure the MQTT broker is running
- Check the MQTT configuration in the `.env` file
- Verify network connectivity to the broker

**RTSP streaming doesn't work**:
- Ensure ffmpeg is installed and in your PATH
- Check the RTSP configuration in the `.env` file
- Verify that the RTSP server is accessible

**Models not loading**:
- Verify that model files exist at the paths specified in the `.env` file
- Check if the model formats are compatible with the system

### License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Documentación en Español

### Descripción General

El Sistema de Detección con IA es una aplicación de visión por computadora que combina capacidades de reconocimiento facial y detección de objetos con mensajería MQTT y transmisión RTSP. El sistema está diseñado para proporcionar funciones fiables de vigilancia y monitoreo con las siguientes características clave:

- **Reconocimiento Facial**: Detecta e identifica rostros utilizando embeddings de FaceNet
- **Detección de Objetos**: Detecta y clasifica objetos utilizando YOLO
- **Integración MQTT**: Envía eventos de detección a un broker MQTT
- **Transmisión RTSP**: Transmite video procesado vía RTSP
- **Manejo Robusto de Errores**: Gestiona interrupciones de red y fallos de cámara con elegancia
- **Monitoreo de Rendimiento**: Rastrea e informa métricas de rendimiento del sistema

### Arquitectura del Sistema

El sistema está construido con una arquitectura modular, lo que facilita su mantenimiento y extensión:

```
ai_detection_system/
├── config/                # Gestión de configuración
├── models/                # Modelos de IA (reconocimiento facial, detección de objetos)
├── utils/                 # Módulos de utilidad (MQTT, RTSP, captura de video)
├── data_handlers/         # Almacenamiento y recuperación de datos
├── main.py                # Punto de entrada principal de la aplicación
└── requirements.txt       # Dependencias
```

### Requisitos

- Python 3.8 o superior
- GPU compatible con CUDA (opcional, para inferencia más rápida)
- Cámara (webcam o cámara IP)
- Broker MQTT (por ejemplo, Mosquitto)
- Servidor RTSP (por ejemplo, ffmpeg)

Los paquetes de Python requeridos están listados en `requirements.txt`.

### Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/yourusername/ai_detection_system.git
   cd ai_detection_system
   ```

2. Crear un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Instalar dependencias adicionales:
   - ffmpeg (para transmisión RTSP)
   - CUDA y cuDNN (opcional, para aceleración GPU)

5. Configurar la aplicación:
   ```bash
   cp .env.template .env
   ```
   Editar el archivo `.env` con tus valores de configuración.

6. Preparar los modelos:
   - Descargar el archivo shape predictor de dlib
   - Descargar o entrenar el modelo FaceNet
   - Descargar un modelo YOLO (por ejemplo, YOLOv10m)
   
   Colocar estos modelos en los directorios apropiados según lo configurado en tu archivo `.env`.

### Uso

Ejecutar la aplicación:

```bash
python main.py
```

Opciones de línea de comandos:
- `--display`: Mostrar frames de video en una ventana
- `--log-level`: Establecer nivel de registro (DEBUG, INFO, WARNING, ERROR, CRITICAL)

Mientras el sistema está en ejecución:
- Presiona 'q' para salir
- Presiona 'p' para pausar/reanudar el procesamiento
- Presiona 'r' para registrar un nuevo rostro

### Formato de Mensajes MQTT

El sistema publica eventos de detección en el tema MQTT configurado en formato JSON:

**Detección de Rostros**:
```json
{
  "type": "face",
  "name": "Juan",
  "confidence": 0.92,
  "timestamp": 1679401235.765
}
```

**Detección de Objetos**:
```json
{
  "type": "objects",
  "classes": ["person", "car"],
  "confidences": [0.98, 0.87],
  "timestamp": 1679401236.123
}
```

### Transmisión RTSP

La transmisión de video procesado está disponible en la URL RTSP configurada:
```
rtsp://usuario:contraseña@ip:puerto/ruta
```

Puedes ver esta transmisión usando cualquier reproductor compatible con RTSP (por ejemplo, VLC).

### Registro de Rostros

Para registrar nuevos rostros en el sistema:
1. Presiona 'r' cuando un rostro sea visible en la cámara
2. Ingresa el nombre de la persona cuando se te solicite
3. El rostro se añadirá a la base de datos para reconocimiento futuro

### Solución de Problemas

**El sistema no detecta la cámara**:
- Verifica que la cámara esté correctamente conectada
- Comprueba que el índice o URL de la cámara sea correcto en el archivo `.env`

**La conexión MQTT falla**:
- Asegúrate de que el broker MQTT esté en ejecución
- Revisa la configuración MQTT en el archivo `.env`
- Verifica la conectividad de red al broker

**La transmisión RTSP no funciona**:
- Asegúrate de que ffmpeg esté instalado y en tu PATH
- Revisa la configuración RTSP en el archivo `.env`
- Verifica que el servidor RTSP sea accesible

**Los modelos no se cargan**:
- Verifica que los archivos del modelo existan en las rutas especificadas en el archivo `.env`
- Comprueba si los formatos de los modelos son compatibles con el sistema

### Licencia

Este proyecto está licenciado bajo la Licencia MIT - consulta el archivo LICENSE para más detalles.