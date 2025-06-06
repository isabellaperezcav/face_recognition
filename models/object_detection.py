"""
Object Detection Model for the AI Detection System.

This module provides the ObjectDetection class, which encapsulates the object 
detection functionality using YOLO. It includes:
- Loading and initializing the YOLO model
- Processing frames to detect objects
- Filtering and post-processing detections
- Performance monitoring and statistics

Usage:
    from models.object_detection import ObjectDetection
    
    # Create an instance
    detector = ObjectDetection(model_path='yolov10m.pt')
    
    # Process a frame
    detections = detector.process_frame(frame)
    
    # Each detection contains class, confidence, and bounding box
"""

import time
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import cv2

# Configure logger
from utils.logging_utils import setup_logger
logger = setup_logger('object_detection')

class ObjectDetection:
    """
    Handles object detection using YOLO.
    
    This class encapsulates the object detection pipeline using:
    - YOLO for object detection
    - Filtering by confidence and class
    - Performance monitoring
    """
    
    def __init__(
        self, 
        model_path: str,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
        device: str = 'cpu'
    ):
        """
        Initialize the object detection model.
        
        Args:
            model_path: Path to the YOLO model file
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            input_size: Input size for the model (width, height)
            device: Device to run inference on ('cpu' or 'cuda:0')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.device = device
        
        # Model will be loaded on first use
        self.model = None
        
        # Track classes of interest (empty means all classes)
        self.classes_of_interest = []
        
        # Performance tracking
        self.inference_times = []
        self.performance_tracking_window = 100  # Keep last N measurements
        
        logger.info(f"Object Detection initialized with model: {model_path}")
    
    def _load_model(self):
        """
        Load the YOLO model.
        """
        try:
            # Import here to avoid loading ultralytics at module import time
            from ultralytics import YOLO
            
            logger.info(f"Loading YOLO model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            
            # Get class names
            self.class_names = self.model.names
            logger.info(f"Model loaded with {len(self.class_names)} classes")
            
            # Log model info
            logger.info(f"Model info: {self.model}")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise
    
    def set_classes_of_interest(self, classes: List[str]):
        """
        Set classes of interest to filter detections.
        
        Args:
            classes: List of class names to detect
                    (empty list means detect all classes)
        """
        self.classes_of_interest = classes
        logger.info(f"Set classes of interest: {classes if classes else 'all'}")
    
    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process a frame to detect objects.
        
        Args:
            frame: Input image frame
            
        Returns:
            List[Dict[str, Any]]: List of detected objects with class, confidence,
                                 and bounding box
        """
        # Load model on first use
        if self.model is None:
            self._load_model()
        
        start_time = time.time()
        
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                device=self.device
            )
            
            # Process results
            detections = []
            
            for result in results:
                # Extract boxes, confidences, and class IDs
                for box in result.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, class_id = box
                    
                    # Convert class ID to integer
                    class_id = int(class_id)
                    
                    # Get class name
                    class_name = self.class_names[class_id]
                    
                    # Filter by classes of interest
                    if self.classes_of_interest and class_name not in self.classes_of_interest:
                        continue
                    
                    # Store detection
                    detections.append({
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': (int(x1), int(y1), int(x2), int(y2))
                    })
            
            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > self.performance_tracking_window:
                self.inference_times.pop(0)
            
            logger.debug(f"Detected {len(detections)} objects in {inference_time:.4f} seconds")
            return detections
        
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detection results on a frame.
        
        Args:
            frame: Input image frame
            detections: List of detections
            
        Returns:
            np.ndarray: Frame with detection results drawn
        """
        # Make a copy of the frame
        output = frame.copy()
        
        # Colors for different classes (generated based on class ID)
        colors = {
            'person': (0, 255, 0),  # Green
            'car': (0, 0, 255),     # Red
            'truck': (255, 0, 0),   # Blue
            'bicycle': (255, 255, 0), # Cyan
            'motorcycle': (255, 0, 255), # Magenta
            'default': (0, 255, 255) # Yellow
        }
        
        for det in detections:
            # Get detection info
            bbox = det['bbox']
            conf = det['confidence']
            class_name = det['class']
            
            # Get color for class
            color = colors.get(class_name, colors['default'])
            
            # Draw bounding box
            cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Prepare label text
            label = f"{class_name} ({conf:.2f})"
            
            # Draw label
            cv2.putText(output, label, (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output
    
    def get_class_counts(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Get counts of detected classes.
        
        Args:
            detections: List of detections
            
        Returns:
            Dict[str, int]: Counts of detected classes
        """
        counts = {}
        for det in detections:
            class_name = det['class']
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts
    
    def filter_detections(self, 
                        detections: List[Dict[str, Any]],
                        min_confidence: Optional[float] = None,
                        classes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Filter detections by confidence and class.
        
        Args:
            detections: List of detections
            min_confidence: Minimum confidence threshold (optional)
            classes: List of classes to include (optional)
            
        Returns:
            List[Dict[str, Any]]: Filtered detections
        """
        filtered = []
        
        for det in detections:
            # Filter by confidence
            if min_confidence is not None and det['confidence'] < min_confidence:
                continue
            
            # Filter by class
            if classes is not None and det['class'] not in classes:
                continue
            
            filtered.append(det)
        
        return filtered
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the object detection model.
        
        Returns:
            Dict[str, Any]: Performance statistics
        """
        stats = {
            "avg_inference_time": np.mean(self.inference_times) if self.inference_times else 0,
            "fps": 1.0 / np.mean(self.inference_times) if self.inference_times else 0,
            "model_path": self.model_path,
            "input_size": self.input_size,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "device": self.device
        }
        return stats