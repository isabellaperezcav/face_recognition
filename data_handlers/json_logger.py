"""
JSON Logger for the AI Detection System.

This module provides the JSONLogger class, which handles JSON-formatted
data logging for detections (faces, objects). It includes functions for:
- Logging face and object detections to JSON files
- Maintaining uniqueness rules to avoid duplicate entries
- Managing file rotation and cleanup
- Providing query capabilities for logged data

Usage:
    from data_handlers.json_logger import JSONLogger
    
    # Create logger instance
    logger = JSONLogger(base_dir='logs')
    
    # Log a face detection
    logger.log_face_detection('John', 0.95)
    
    # Log an object detection
    logger.log_object_detection(['person', 'car'], [0.98, 0.85])
"""

import os
import json
import time
import logging
import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from threading import Lock

# Configure logger
from utils.logging_utils import setup_logger
logger = setup_logger('json_logger')

class JSONLogger:
    """
    Manages logging of detections to JSON files.
    
    This class provides methods to:
    - Log face and object detections
    - Apply uniqueness rules to avoid duplicate entries
    - Rotate log files based on size or time
    - Query logged data
    """
    
    def __init__(
        self, 
        base_dir: str = 'logs',
        face_log_file: str = 'personas.json',
        object_log_file: str = 'objetos.json',
        rotate_size_mb: float = 10.0,
        max_files: int = 10
    ):
        """
        Initialize the JSON logger.
        
        Args:
            base_dir: Base directory for log files
            face_log_file: Filename for face detection logs
            object_log_file: Filename for object detection logs
            rotate_size_mb: Maximum file size before rotation (in MB)
            max_files: Maximum number of log files to keep
        """
        self.base_dir = base_dir
        self.face_log_file = os.path.join(base_dir, face_log_file)
        self.object_log_file = os.path.join(base_dir, object_log_file)
        self.rotate_size_mb = rotate_size_mb
        self.max_files = max_files
        
        # Create locks for thread safety
        self.face_lock = Lock()
        self.object_lock = Lock()
        
        # Track the last entries to implement uniqueness
        self.last_face = None
        self.last_objects = set()
        
        # Create the log directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        # Initialize log files if they don't exist
        self._initialize_log_files()
        
        logger.info(f"JSON Logger initialized: face_log={face_log_file}, object_log={object_log_file}")
    
    def _initialize_log_files(self):
        """
        Initialize log files if they don't exist.
        """
        for filepath in [self.face_log_file, self.object_log_file]:
            if not os.path.exists(filepath):
                with open(filepath, 'w') as f:
                    json.dump([], f)
                logger.info(f"Created new log file: {filepath}")
    
    def _rotate_if_needed(self, filepath: str):
        """
        Rotate the log file if it exceeds the maximum size.
        
        Args:
            filepath: Path to the log file to check
        """
        if not os.path.exists(filepath):
            return
        
        # Check file size
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if size_mb < self.rotate_size_mb:
            return
        
        # Rotate the file
        logger.info(f"Rotating log file: {filepath} ({size_mb:.2f} MB)")
        
        # Get the directory and filename
        directory, filename = os.path.split(filepath)
        name, ext = os.path.splitext(filename)
        
        # Create timestamp for the rotated file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_path = os.path.join(directory, f"{name}_{timestamp}{ext}")
        
        try:
            # Rename the current file
            os.rename(filepath, rotated_path)
            
            # Create a new empty file
            with open(filepath, 'w') as f:
                json.dump([], f)
            
            logger.info(f"Rotated log file: {filepath} -> {rotated_path}")
            
            # Clean up old files if needed
            self._cleanup_old_files(directory, name, ext)
        except Exception as e:
            logger.error(f"Error rotating log file {filepath}: {e}")
    
    def _cleanup_old_files(self, directory: str, name: str, ext: str):
        """
        Clean up old rotated log files.
        
        Args:
            directory: The directory containing the log files
            name: The base name of the log files
            ext: The file extension
        """
        # Find all rotated files
        rotated_files = []
        for filename in os.listdir(directory):
            if filename.startswith(name + '_') and filename.endswith(ext):
                filepath = os.path.join(directory, filename)
                rotated_files.append((filepath, os.path.getmtime(filepath)))
        
        # Sort by modification time (oldest first)
        rotated_files.sort(key=lambda x: x[1])
        
        # Delete oldest files if we have too many
        if len(rotated_files) > self.max_files:
            files_to_delete = rotated_files[:(len(rotated_files) - self.max_files)]
            for filepath, _ in files_to_delete:
                try:
                    os.remove(filepath)
                    logger.info(f"Deleted old log file: {filepath}")
                except Exception as e:
                    logger.error(f"Error deleting old log file {filepath}: {e}")
    
    def _read_log_file(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Read a log file and return its contents.
        
        Args:
            filepath: Path to the log file
            
        Returns:
            List[Dict[str, Any]]: The log entries
        """
        try:
            if not os.path.exists(filepath):
                return []
            
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading log file {filepath}: {e}")
            return []
    
    def _write_log_file(self, filepath: str, entries: List[Dict[str, Any]]) -> bool:
        """
        Write entries to a log file.
        
        Args:
            filepath: Path to the log file
            entries: The log entries to write
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(entries, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Error writing to log file {filepath}: {e}")
            return False
    
    def log_face_detection(self, name: str, distance: float, 
                          extra_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log a face detection.
        
        Args:
            name: The detected person's name
            distance: The face recognition distance/confidence
            extra_data: Additional data to include in the log (optional)
            
        Returns:
            bool: True if logged successfully, False otherwise
        """
        # Skip if this is the same as the last detection
        if name == self.last_face:
            return True
        
        # Prepare log entry
        timestamp = time.time()
        fecha = time.strftime("%Y-%m-%d", time.localtime(timestamp))
        hora = time.strftime("%H:%M:%S", time.localtime(timestamp))
        
        entry = {
            "fecha": fecha,
            "hora": hora,
            "timestamp": timestamp,
            "Nombre_Persona": name,
            "distancia": float(distance)
        }
        
        # Add extra data if provided
        if extra_data:
            entry.update(extra_data)
        
        # Update last face
        self.last_face = name
        
        # Thread-safe write to file
        with self.face_lock:
            # Check if rotation is needed
            self._rotate_if_needed(self.face_log_file)
            
            # Read current entries
            entries = self._read_log_file(self.face_log_file)
            
            # Add new entry
            entries.append(entry)
            
            # Write updated entries
            success = self._write_log_file(self.face_log_file, entries)
            
            if success:
                logger.debug(f"Logged face detection: {name}")
            
            return success
    
    def log_object_detection(self, objects: List[str], confidences: List[float],
                            extra_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log object detections.
        
        Args:
            objects: List of detected object classes
            confidences: List of confidence scores for each object
            extra_data: Additional data to include in the log (optional)
            
        Returns:
            bool: True if logged successfully, False otherwise
        """
        # Convert to set for comparison and filtering
        objects_set = set(objects)
        
        # Skip if these are the same objects as the last detection
        if objects_set == self.last_objects:
            return True
        
        # Find new objects
        new_objects = objects_set - self.last_objects
        
        # Skip if no new objects
        if not new_objects:
            return True
        
        # Prepare log entry
        timestamp = time.time()
        fecha = time.strftime("%Y-%m-%d", time.localtime(timestamp))
        hora = time.strftime("%H:%M:%S", time.localtime(timestamp))
        
        # Filter confidences for new objects
        new_object_indices = [i for i, obj in enumerate(objects) if obj in new_objects]
        new_object_list = [objects[i] for i in new_object_indices]
        new_confidence_list = [confidences[i] for i in new_object_indices]
        
        entry = {
            "fecha": fecha,
            "hora": hora,
            "timestamp": timestamp,
            "objeto_detectado": list(new_object_list),
            "confianza": [float(conf) for conf in new_confidence_list]
        }
        
        # Add extra data if provided
        if extra_data:
            entry.update(extra_data)
        
        # Update last objects
        self.last_objects = objects_set
        
        # Thread-safe write to file
        with self.object_lock:
            # Check if rotation is needed
            self._rotate_if_needed(self.object_log_file)
            
            # Read current entries
            entries = self._read_log_file(self.object_log_file)
            
            # Add new entry
            entries.append(entry)
            
            # Write updated entries
            success = self._write_log_file(self.object_log_file, entries)
            
            if success:
                logger.debug(f"Logged object detection: {new_object_list}")
            
            return success
    
    def get_face_detections(self, 
                           start_time: Optional[float] = None,
                           end_time: Optional[float] = None,
                           person_name: Optional[str] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query face detection logs.
        
        Args:
            start_time: Start timestamp for query (optional)
            end_time: End timestamp for query (optional)
            person_name: Filter by person name (optional)
            limit: Maximum number of entries to return
            
        Returns:
            List[Dict[str, Any]]: Filtered log entries
        """
        # Thread-safe read
        with self.face_lock:
            entries = self._read_log_file(self.face_log_file)
        
        # Apply filters
        filtered_entries = []
        for entry in entries:
            # Filter by timestamp
            timestamp = entry.get("timestamp", 0)
            if start_time is not None and timestamp < start_time:
                continue
            if end_time is not None and timestamp > end_time:
                continue
            
            # Filter by person name
            if person_name is not None and entry.get("Nombre_Persona") != person_name:
                continue
            
            filtered_entries.append(entry)
        
        # Sort by timestamp (newest first)
        filtered_entries.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        # Apply limit
        return filtered_entries[:limit]
    
    def get_object_detections(self,
                             start_time: Optional[float] = None,
                             end_time: Optional[float] = None,
                             object_class: Optional[str] = None,
                             limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query object detection logs.
        
        Args:
            start_time: Start timestamp for query (optional)
            end_time: End timestamp for query (optional)
            object_class: Filter by object class (optional)
            limit: Maximum number of entries to return
            
        Returns:
            List[Dict[str, Any]]: Filtered log entries
        """
        # Thread-safe read
        with self.object_lock:
            entries = self._read_log_file(self.object_log_file)
        
        # Apply filters
        filtered_entries = []
        for entry in entries:
            # Filter by timestamp
            timestamp = entry.get("timestamp", 0)
            if start_time is not None and timestamp < start_time:
                continue
            if end_time is not None and timestamp > end_time:
                continue
            
            # Filter by object class
            if object_class is not None:
                objects = entry.get("objeto_detectado", [])
                if object_class not in objects:
                    continue
            
            filtered_entries.append(entry)
        
        # Sort by timestamp (newest first)
        filtered_entries.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        # Apply limit
        return filtered_entries[:limit]
    
    def clear_logs(self, log_type: Optional[str] = None) -> bool:
        """
        Clear log files.
        
        Args:
            log_type: Type of log to clear ('face', 'object', or None for both)
            
        Returns:
            bool: True if successful, False otherwise
        """
        success = True
        
        if log_type in [None, 'face']:
            with self.face_lock:
                success &= self._write_log_file(self.face_log_file, [])
                self.last_face = None
        
        if log_type in [None, 'object']:
            with self.object_lock:
                success &= self._write_log_file(self.object_log_file, [])
                self.last_objects = set()
        
        if success:
            logger.info(f"Cleared logs: {log_type if log_type else 'all'}")
        
        return success
    
    def reset_uniqueness_tracking(self):
        """
        Reset the uniqueness tracking for face and object detections.
        """
        self.last_face = None
        self.last_objects = set()
        logger.info("Reset uniqueness tracking")