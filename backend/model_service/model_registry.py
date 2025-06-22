#!/usr/bin/env python3
# model_registry.py - Manages model versions, configurations, and loading

import os
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Model Registry for managing multiple model versions, configurations, and metadata.
    Enables versioning, tracking, and easy switching between different trained models.
    """
    
    def __init__(self, registry_dir=None, config_path=None):
        """
        Initialize the model registry.
        
        Args:
            registry_dir: Directory where models are stored
            config_path: Path to the model registry configuration file
        """
        self.registry_dir = registry_dir or os.environ.get('MODEL_REGISTRY_DIR', '/app/models')
        self.config_path = config_path or os.environ.get('MODEL_CONFIG_PATH', 
                                                        os.path.join(self.registry_dir, 'model_registry.yaml'))
        self.models = {}
        self.active_model = None
        
        # Ensure registry directory exists
        os.makedirs(self.registry_dir, exist_ok=True)
        
        # Load registry configuration if it exists, or create a default one
        self._load_registry()

    def _load_registry(self):
        """Load the model registry configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.models = config.get('models', {})
                    self.active_model = config.get('active_model')
                logger.info(f"Loaded model registry with {len(self.models)} models")
            else:
                logger.info("Model registry config not found, creating new registry")
                self._save_registry()
        except Exception as e:
            logger.error(f"Error loading model registry: {e}")
            self.models = {}
            self.active_model = None
    
    def _save_registry(self):
        """Save the model registry configuration to file"""
        try:
            config = {
                'models': self.models,
                'active_model': self.active_model
            }
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(config, f)
            logger.info(f"Saved model registry with {len(self.models)} models")
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")
    
    def register_model(self, model_id, model_path, description=None, metadata=None):
        """
        Register a new model in the registry
        
        Args:
            model_id: Unique identifier for the model
            model_path: Path to the model weights file
            description: Human-readable description
            metadata: Additional metadata (training params, performance metrics, etc.)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if model_id in self.models:
                logger.warning(f"Model {model_id} already exists, updating registration")
            
            # Create model entry
            self.models[model_id] = {
                'model_path': model_path,
                'description': description or f"Model {model_id}",
                'metadata': metadata or {},
                'registration_date': datetime.now().isoformat()
            }
            
            self._save_registry()
            return True
        except Exception as e:
            logger.error(f"Error registering model {model_id}: {e}")
            return False
    
    def set_active_model(self, model_id):
        """
        Set the active model to use for inference
        
        Args:
            model_id: ID of model to set as active
        
        Returns:
            bool: True if successful, False otherwise
        """
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        self.active_model = model_id
        self._save_registry()
        logger.info(f"Set active model to {model_id}")
        return True
    
    def get_model_info(self, model_id=None):
        """
        Get information about a specific model or the active model
        
        Args:
            model_id: ID of model to get info for, or None for active model
        
        Returns:
            dict: Model information or None if not found
        """
        if model_id is None:
            model_id = self.active_model
        
        if model_id is None:
            logger.error("No active model set and no model_id provided")
            return None
        
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found in registry")
            return None
        
        return self.models[model_id]
    
    def get_all_models(self):
        """
        Get information about all registered models
        
        Returns:
            dict: All models information
        """
        return self.models
    
    def get_active_model_path(self):
        """
        Get the path to the active model's weights file
        
        Returns:
            str: Path to model weights file or None if not set
        """
        if self.active_model is None or self.active_model not in self.models:
            return None
        
        return self.models[self.active_model]['model_path']

    def delete_model(self, model_id):
        """
        Delete a model from the registry (does not delete the model file)
        
        Args:
            model_id: ID of model to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        # If deleting the active model, unset it
        if self.active_model == model_id:
            self.active_model = None
        
        del self.models[model_id]
        self._save_registry()
        logger.info(f"Deleted model {model_id} from registry")
        return True
