"""
Face Database Handler for the AI Detection System.

This module provides the FaceDatabase class, which manages the face embedding
database for face recognition. It includes functions for:
- Loading and saving the face database
- Adding and removing face embeddings
- Finding the best match for a face embedding
- Managing face database metadata

Usage:
    from data_handlers.database import FaceDatabase
    
    # Create a database instance
    db = FaceDatabase('face_database.pkl')
    
    # Register a new face
    db.add_face('John', embedding)
    
    # Find the best match for a face
    name, similarity = db.find_match(embedding)
    
    # Save the database
    db.save()
"""

import os
import pickle
import time
import logging
import numpy as np
from typing import Optional, Tuple, Dict, List, Any, Union

# Configure logger
from utils.logging_utils import setup_logger
logger = setup_logger('face_db')

class FaceDatabase:
    """
    Manages the face embedding database for face recognition.
    
    This class provides methods to:
    - Load and save face embeddings from/to disk
    - Add and remove face embeddings
    - Find the best match for a face embedding
    - Manage database metadata
    """
    
    def __init__(self, db_path: str, threshold: float = 0.65):
        """
        Initialize the face database.
        
        Args:
            db_path: Path to the face database file
            threshold: Similarity threshold for face matching (lower is stricter)
        """
        self.db_path = db_path
        self.threshold = threshold
        self.faces = {}  # name -> embedding
        self.metadata = {}  # name -> metadata (registration time, etc.)
        self.load()
        
        logger.info(f"Face database initialized with {len(self.faces)} faces")
    
    def load(self) -> bool:
        """
        Load the face database from disk.
        
        Returns:
            bool: True if the database was loaded successfully, False otherwise
        """
        if not os.path.exists(self.db_path):
            logger.info(f"Face database file not found: {self.db_path}")
            return False
        
        try:
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)
                
                # Handle different database formats (backward compatibility)
                if isinstance(data, dict):
                    if 'faces' in data and 'metadata' in data:
                        # New format with metadata
                        self.faces = data['faces']
                        self.metadata = data['metadata']
                    else:
                        # Old format (just faces)
                        self.faces = data
                        self.metadata = {name: {'registered_at': time.time()} for name in self.faces}
                else:
                    logger.error(f"Invalid database format in {self.db_path}")
                    return False
                
                logger.info(f"Loaded {len(self.faces)} faces from database")
                return True
        except Exception as e:
            logger.error(f"Error loading face database: {e}")
            return False
    
    def save(self) -> bool:
        """
        Save the face database to disk.
        
        Returns:
            bool: True if the database was saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            
            # Save database with metadata
            data = {
                'faces': self.faces,
                'metadata': self.metadata
            }
            
            with open(self.db_path, "wb") as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved {len(self.faces)} faces to database")
            return True
        except Exception as e:
            logger.error(f"Error saving face database: {e}")
            return False
    
    def add_face(self, name: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a face to the database.
        
        Args:
            name: The name/identifier for the face
            embedding: The face embedding vector
            metadata: Additional metadata for the face (optional)
            
        Returns:
            bool: True if the face was added successfully, False otherwise
        """
        try:
            # Normalize the embedding if it's not already normalized
            norm = np.linalg.norm(embedding)
            if norm > 0:
                normalized_embedding = embedding / norm
            else:
                normalized_embedding = embedding
            
            # Store the face embedding
            self.faces[name] = normalized_embedding
            
            # Store metadata
            if metadata is None:
                metadata = {}
            
            self.metadata[name] = {
                'registered_at': time.time(),
                'updated_at': time.time(),
                **metadata
            }
            
            # Save the updated database
            self.save()
            
            logger.info(f"Added face '{name}' to database")
            return True
        except Exception as e:
            logger.error(f"Error adding face to database: {e}")
            return False
    
    def remove_face(self, name: str) -> bool:
        """
        Remove a face from the database.
        
        Args:
            name: The name/identifier of the face to remove
            
        Returns:
            bool: True if the face was removed successfully, False otherwise
        """
        if name not in self.faces:
            logger.warning(f"Face '{name}' not found in database")
            return False
        
        try:
            # Remove the face and its metadata
            del self.faces[name]
            if name in self.metadata:
                del self.metadata[name]
            
            # Save the updated database
            self.save()
            
            logger.info(f"Removed face '{name}' from database")
            return True
        except Exception as e:
            logger.error(f"Error removing face from database: {e}")
            return False
    
    def find_match(self, embedding: np.ndarray, top_n: int = 1) -> Union[Tuple[str, float], List[Tuple[str, float]]]:
        """
        Find the best match for a face embedding.
        
        Args:
            embedding: The face embedding to match
            top_n: Number of top matches to return (default: 1)
            
        Returns:
            If top_n == 1:
                Tuple[str, float]: The name and similarity score of the best match
            If top_n > 1:
                List[Tuple[str, float]]: List of (name, similarity) tuples for top matches
        """
        if not self.faces:
            if top_n == 1:
                return "Unknown", 0.0
            else:
                return [("Unknown", 0.0)]
        
        # Normalize the embedding if it's not already normalized
        norm = np.linalg.norm(embedding)
        if norm > 0:
            normalized_embedding = embedding / norm
        else:
            normalized_embedding = embedding
        
        # Calculate similarity scores for all faces
        similarities = {}
        for name, db_embedding in self.faces.items():
            # Use cosine similarity: higher values are more similar
            similarity = 1.0 - np.linalg.norm(normalized_embedding - db_embedding)
            similarities[name] = similarity
        
        # Sort by similarity (higher values first)
        sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Apply threshold to the best match
        best_name, best_similarity = sorted_matches[0]
        threshold_similarity = 1.0 - self.threshold
        
        if best_similarity < threshold_similarity:
            best_name = "Unknown"
        
        # Return single best match or top N matches
        if top_n == 1:
            return best_name, best_similarity
        else:
            # Apply threshold to all matches
            filtered_matches = [(name, similarity) if similarity >= threshold_similarity else ("Unknown", similarity) 
                               for name, similarity in sorted_matches[:top_n]]
            return filtered_matches
    
    def update_metadata(self, name: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a face.
        
        Args:
            name: The name/identifier of the face
            metadata: The metadata to update
            
        Returns:
            bool: True if the metadata was updated successfully, False otherwise
        """
        if name not in self.faces:
            logger.warning(f"Face '{name}' not found in database")
            return False
        
        try:
            # Initialize metadata if it doesn't exist
            if name not in self.metadata:
                self.metadata[name] = {
                    'registered_at': time.time()
                }
            
            # Update metadata
            self.metadata[name].update(metadata)
            self.metadata[name]['updated_at'] = time.time()
            
            # Save the updated database
            self.save()
            
            logger.info(f"Updated metadata for face '{name}'")
            return True
        except Exception as e:
            logger.error(f"Error updating face metadata: {e}")
            return False
    
    def get_all_faces(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all faces with their metadata.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of face names and their metadata
        """
        result = {}
        
        for name in self.faces:
            # Include metadata if available
            meta = self.metadata.get(name, {})
            
            # Add human-readable timestamps
            if 'registered_at' in meta:
                meta['registered_at_str'] = time.strftime('%Y-%m-%d %H:%M:%S', 
                                                         time.localtime(meta['registered_at']))
            
            if 'updated_at' in meta:
                meta['updated_at_str'] = time.strftime('%Y-%m-%d %H:%M:%S', 
                                                      time.localtime(meta['updated_at']))
            
            result[name] = meta
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict[str, Any]: Dictionary with database statistics
        """
        stats = {
            "total_faces": len(self.faces),
            "database_path": self.db_path,
            "threshold": self.threshold,
            "last_modified": os.path.getmtime(self.db_path) if os.path.exists(self.db_path) else None
        }
        
        if stats["last_modified"]:
            stats["last_modified_str"] = time.strftime('%Y-%m-%d %H:%M:%S', 
                                                     time.localtime(stats["last_modified"]))
        
        return stats