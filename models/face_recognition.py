"""
Face Recognition Model for the AI Detection System.

This module provides the FaceRecognition class, which encapsulates the face 
recognition functionality using FaceNet and dlib. It includes:
- Face detection using dlib
- Face alignment and preprocessing
- Face embedding generation using FaceNet
- Face matching against a database

Usage:
    from models.face_recognition import FaceRecognition
    
    # Create an instance
    face_recognizer = FaceRecognition(
        model_path='facenet_model.keras',
        shape_predictor_path='shape_predictor_68_face_landmarks.dat'
    )
    
    # Process a frame
    faces = face_recognizer.process_frame(frame)
    
    # Each face contains detection, embedding, and recognition results
"""

import cv2
import dlib
import numpy as np
import tensorflow as tf
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
import time

# Configure logger
from utils.logging_utils import setup_logger
logger = setup_logger('face_recognition')

class FaceRecognition:
    """
    Handles face detection, alignment, embedding generation and recognition.
    
    This class encapsulates the face recognition pipeline using:
    - dlib's face detector and shape predictor for face detection and alignment
    - FaceNet for face embedding generation
    - L2 distance for face matching
    """
    
    def __init__(
        self, 
        model_path: str,
        shape_predictor_path: str,
        detection_threshold: float = 0.5,
        recognition_threshold: float = 0.65,
        batch_size: int = 8,
        target_size: Tuple[int, int] = (160, 160)
    ):
        """
        Initialize the face recognition model.
        
        Args:
            model_path: Path to the FaceNet model file
            shape_predictor_path: Path to the dlib shape predictor file
            detection_threshold: Confidence threshold for face detection
            recognition_threshold: Distance threshold for face recognition
            batch_size: Batch size for inference
            target_size: Input size expected by the FaceNet model
        """
        self.model_path = model_path
        self.shape_predictor_path = shape_predictor_path
        self.detection_threshold = detection_threshold
        self.recognition_threshold = recognition_threshold
        self.batch_size = batch_size
        self.target_size = target_size
        
        # Load models
        self._load_models()
        
        # Performance tracking
        self.inference_times = []
        self.detection_times = []
        self.alignment_times = []
        self.embedding_times = []
        self.performance_tracking_window = 100  # Keep last N measurements
        
        logger.info("Face Recognition model initialized")
    
    def _load_models(self):
        """
        Load the face detection and recognition models.
        """
        try:
            # Load dlib models
            logger.info("Loading dlib models...")
            self.detector = dlib.get_frontal_face_detector()
            self.shape_predictor = dlib.shape_predictor(self.shape_predictor_path)
            
            # Load FaceNet model
            logger.info(f"Loading FaceNet model from {self.model_path}...")
            
            # Define the triplet loss function
            def triplet_loss(margin=0.2):
                def _triplet_loss(y_true, y_pred):
                    batch_size = tf.shape(y_pred)[0] // 3
                    anchor = y_pred[0:batch_size]
                    positive = y_pred[batch_size:2*batch_size]
                    negative = y_pred[2*batch_size:3*batch_size]
                    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
                    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
                    basic_loss = pos_dist - neg_dist + margin
                    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
                    return loss
                return _triplet_loss
            
            # Load the model with custom loss
            self.model = tf.keras.models.load_model(
                self.model_path, 
                compile=False
            )
            
            # Compile the model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss=triplet_loss(margin=0.2)
            )
            
            # Warm up the model
            logger.info("Warming up the FaceNet model...")
            dummy_input = np.zeros((1, *self.target_size, 3), dtype=np.float32)
            _ = self.model.predict(dummy_input, verbose=0)
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a frame using dlib.
        
        Args:
            frame: Input image frame
            
        Returns:
            List[Dict[str, Any]]: List of detected faces with bounding box and confidence
        """
        start_time = time.time()
        
        # Convert to grayscale for dlib
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Detect faces
        dlib_faces = self.detector(gray, 0)
        
        # Convert dlib results to our format
        faces = []
        for face in dlib_faces:
            # Extract coordinates
            x, y = face.left(), face.top()
            w, h = face.width(), face.height()
            
            # Ensure coordinates are within frame boundaries
            x = max(0, x)
            y = max(0, y)
            x_end = min(frame.shape[1], x + w)
            y_end = min(frame.shape[0], y + h)
            
            # Extract face region
            face_img = frame[y:y_end, x:x_end]
            
            # Skip if the face region is empty
            if face_img.size == 0:
                continue
            
            # Get shape landmarks
            shape = self.shape_predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            
            # Store results
            faces.append({
                'bbox': (x, y, x_end, y_end),
                'confidence': 1.0,  # dlib doesn't provide confidence scores
                'landmarks': landmarks,
                'face_img': face_img
            })
        
        # Track performance
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        if len(self.detection_times) > self.performance_tracking_window:
            self.detection_times.pop(0)
        
        logger.debug(f"Detected {len(faces)} faces in {detection_time:.4f} seconds")
        return faces
    
    def _align_face(self, face_img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Align a face using facial landmarks.
        
        Args:
            face_img: The face image
            landmarks: Facial landmarks
            
        Returns:
            np.ndarray: Aligned face image
        """
        start_time = time.time()
        
        try:
            # Get left and right eye centers
            left_eye = landmarks[36:42]  # Left eye landmarks
            right_eye = landmarks[42:48]  # Right eye landmarks
            
            # Compute the center of each eye
            left_eye_center = np.mean(left_eye, axis=0).astype("int")
            right_eye_center = np.mean(right_eye, axis=0).astype("int")
            
            # Compute angle between eyes
            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Compute desired distance between eyes (30% of image width)
            desired_dist = face_img.shape[1] * 0.3
            
            # Compute scaling factor
            dist = np.sqrt((dx ** 2) + (dy ** 2))
            scale = desired_dist / dist if dist > 0 else 1.0
            
            # Compute center point between eyes
            eyes_center = (
                int((left_eye_center[0] + right_eye_center[0]) / 2),
                int((left_eye_center[1] + right_eye_center[1]) / 2)
            )
            
            # Get rotation matrix
            M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
            
            # Apply affine transformation
            aligned_face = cv2.warpAffine(
                face_img, 
                M, 
                (face_img.shape[1], face_img.shape[0]),
                flags=cv2.INTER_CUBIC
            )
            
            # Track performance
            alignment_time = time.time() - start_time
            self.alignment_times.append(alignment_time)
            if len(self.alignment_times) > self.performance_tracking_window:
                self.alignment_times.pop(0)
            
            return aligned_face
        
        except Exception as e:
            logger.warning(f"Face alignment failed: {e}")
            # Return original image if alignment fails
            return face_img
    
    def _preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocess a face image for the FaceNet model.
        
        Args:
            face_img: The face image to preprocess
            
        Returns:
            np.ndarray: Preprocessed face image
        """
        # Resize to target size
        resized = cv2.resize(face_img, self.target_size)
        
        # Convert to float and scale to [0, 1]
        processed = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        return np.expand_dims(processed, axis=0)
    
    def _generate_embedding(self, processed_face: np.ndarray) -> np.ndarray:
        """
        Generate a face embedding using the FaceNet model.
        
        Args:
            processed_face: Preprocessed face image
            
        Returns:
            np.ndarray: Face embedding vector
        """
        start_time = time.time()
        
        # Generate embedding
        embedding = self.model.predict(processed_face, verbose=0)[0]
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Track performance
        embedding_time = time.time() - start_time
        self.embedding_times.append(embedding_time)
        if len(self.embedding_times) > self.performance_tracking_window:
            self.embedding_times.pop(0)
        
        return embedding
    
    def _generate_batch_embeddings(self, processed_faces: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of faces.
        
        Args:
            processed_faces: List of preprocessed face images
            
        Returns:
            List[np.ndarray]: List of face embedding vectors
        """
        if not processed_faces:
            return []
        
        start_time = time.time()
        
        # Combine into a batch
        batch = np.vstack(processed_faces)
        
        # Generate embeddings for the batch
        embeddings = self.model.predict(batch, verbose=0)
        
        # Normalize embeddings
        normalized_embeddings = []
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            normalized_embeddings.append(embedding)
        
        # Track performance
        embedding_time = time.time() - start_time
        self.embedding_times.append(embedding_time / len(processed_faces))
        if len(self.embedding_times) > self.performance_tracking_window:
            self.embedding_times.pop(0)
        
        return normalized_embeddings
    
    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process a frame to detect, align and generate embeddings for faces.
        
        Args:
            frame: Input image frame
            
        Returns:
            List[Dict[str, Any]]: List of processed faces with bounding boxes,
                                 landmarks, aligned images, and embeddings
        """
        start_time = time.time()
        
        # Detect faces
        faces = self._detect_faces(frame)
        
        # Process each face
        processed_faces = []
        aligned_faces = []
        
        for face in faces:
            try:
                # Align face
                aligned_face = self._align_face(face['face_img'], face['landmarks'])
                
                # Preprocess face
                processed_face = self._preprocess_face(aligned_face)
                
                # Store for batch processing
                processed_faces.append(processed_face)
                aligned_faces.append(aligned_face)
            except Exception as e:
                logger.warning(f"Error processing face: {e}")
        
        # Generate embeddings in batches
        if processed_faces:
            embeddings = self._generate_batch_embeddings(processed_faces)
            
            # Add embeddings and aligned faces to results
            for i, (embedding, aligned_face) in enumerate(zip(embeddings, aligned_faces)):
                faces[i]['embedding'] = embedding
                faces[i]['aligned_face'] = aligned_face
        
        # Track performance
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        if len(self.inference_times) > self.performance_tracking_window:
            self.inference_times.pop(0)
        
        logger.debug(f"Processed {len(faces)} faces in {inference_time:.4f} seconds")
        return faces
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the face recognition model.
        
        Returns:
            Dict[str, Any]: Performance statistics
        """
        stats = {
            "avg_inference_time": np.mean(self.inference_times) if self.inference_times else 0,
            "avg_detection_time": np.mean(self.detection_times) if self.detection_times else 0,
            "avg_alignment_time": np.mean(self.alignment_times) if self.alignment_times else 0,
            "avg_embedding_time": np.mean(self.embedding_times) if self.embedding_times else 0,
            "fps": 1.0 / np.mean(self.inference_times) if self.inference_times else 0,
            "model_path": self.model_path,
            "target_size": self.target_size,
            "batch_size": self.batch_size
        }
        return stats