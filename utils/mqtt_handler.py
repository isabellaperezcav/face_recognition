"""
MQTT Handler for the AI Detection System.

This module provides the MQTTHandler class, which implements robust
MQTT connectivity with automatic reconnection, error handling, and
comprehensive logging.

Usage:
    from utils.mqtt_handler import MQTTHandler
    
    # Create handler
    mqtt = MQTTHandler(broker="127.0.0.1", port=1883, topic="ai/detections")
    
    # Connect
    mqtt.connect()
    
    # Publish message
    mqtt.publish(json.dumps({"detected": "person"}))
    
    # Disconnect when done
    mqtt.disconnect()
"""

import time
import json
import logging
from typing import Optional, Callable, Dict, Any, Union
import paho.mqtt.client as mqtt

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mqtt')

class MQTTHandler:
    """
    Handles MQTT communication with robust error handling and reconnection.
    
    This class provides a reliable MQTT client implementation with automatic
    reconnection, error handling, and comprehensive logging.
    """
    
    def __init__(
        self, 
        broker: str, 
        port: int, 
        topic: str, 
        username: Optional[str] = None, 
        password: Optional[str] = None,
        client_id: Optional[str] = None,
        qos: int = 0
    ):
        """
        Initialize the MQTT handler.
        
        Args:
            broker: MQTT broker address
            port: MQTT broker port
            topic: Default topic for publishing messages
            username: MQTT broker username (optional)
            password: MQTT broker password (optional)
            client_id: MQTT client ID (optional)
            qos: Default Quality of Service level (0, 1, or 2)
        """
        self.broker = broker
        self.port = port
        self.topic = topic
        self.username = username
        self.password = password
        self.client_id = client_id
        self.qos = qos
        self.connected = False
        self.client = None
        self.setup_client()
        
        logger.info(f"MQTT Handler initialized for broker {broker}:{port}")
    
    def setup_client(self):
        """
        Setup the MQTT client with appropriate callbacks.
        """
        if self.client_id:
             self.client = mqtt.Client(client_id=self.client_id)

        else:
            self.client = mqtt.Client()

        
        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        self.client.on_message = self._on_message
        
        # Set credentials if provided
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)
        
        # Configure automatic reconnect
        self.client.reconnect_delay_set(min_delay=1, max_delay=120)
    
    def _on_connect(self, client, userdata, flags, rc):
        """
        Callback for when the client receives a CONNACK response from the server.
        
        Args:
            client: The client instance
            userdata: User data provided when creating the client
            flags: Connection flags
            rc: The connection result
        """
        connect_codes = {
            0: "Connection successful",
            1: "Connection refused - incorrect protocol version",
            2: "Connection refused - invalid client identifier",
            3: "Connection refused - server unavailable",
            4: "Connection refused - bad username or password",
            5: "Connection refused - not authorised"
        }
        
        if rc == 0:
            self.connected = True
            logger.info(f"Connected to MQTT broker: {self.broker} ({connect_codes.get(rc, 'Unknown error')})")
        else:
            self.connected = False
            logger.error(f"Failed to connect to MQTT broker: {connect_codes.get(rc, f'Unknown error code {rc}')}")
    
    def _on_disconnect(self, client, userdata, rc):
        """
        Callback for when the client disconnects from the server.
        
        Args:
            client: The client instance
            userdata: User data provided when creating the client
            rc: The disconnection result
        """
        self.connected = False
        if rc == 0:
            logger.info("Disconnected from MQTT broker cleanly")
        else:
            logger.warning(f"Unexpected disconnection from MQTT broker. Code: {rc}")
    
    def _on_publish(self, client, userdata, mid):
        """
        Callback for when a message is published.
        
        Args:
            client: The client instance
            userdata: User data provided when creating the client
            mid: Message ID
        """
        logger.debug(f"Message ID {mid} published successfully")
    
    def _on_message(self, client, userdata, msg):
        """
        Callback for when a message is received.
        
        Args:
            client: The client instance
            userdata: User data provided when creating the client
            msg: The received message
        """
        logger.debug(f"Received message on topic {msg.topic}: {msg.payload}")
    
    def connect(self) -> bool:
        """
        Connect to the MQTT broker.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            logger.info(f"Connecting to MQTT broker {self.broker}:{self.port}...")
            self.client.connect(self.broker, self.port)
            self.client.loop_start()
            
            # Wait a bit for the connection to establish
            for _ in range(10):
                if self.connected:
                    return True
                time.sleep(0.1)
            
            logger.warning("Connection to MQTT broker timed out")
            return False
        except Exception as e:
            logger.error(f"Error connecting to MQTT broker: {e}")
            return False
    
    def disconnect(self):
        """
        Disconnect from the MQTT broker.
        """
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("Disconnected from MQTT broker")
    
    def publish(self, 
                message: Union[str, Dict[str, Any], bytes], 
                topic: Optional[str] = None, 
                qos: Optional[int] = None, 
                retain: bool = False) -> bool:
        """
        Publish a message to the MQTT broker.
        
        Args:
            message: The message to publish (string, dict, or bytes)
            topic: The topic to publish to (defaults to self.topic)
            qos: Quality of Service level (defaults to self.qos)
            retain: Whether to retain the message
            
        Returns:
            bool: True if the message was published successfully, False otherwise
        """
        if not self.connected:
            logger.warning("Cannot publish: MQTT client is not connected")
            return False
        
        # Use default topic if not specified
        if topic is None:
            topic = self.topic
        
        # Use default QoS if not specified
        if qos is None:
            qos = self.qos
        
        # Convert dict to JSON string
        if isinstance(message, dict):
            message = json.dumps(message)
        
        # Add retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self.client.publish(topic, message, qos=qos, retain=retain)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    return True
                logger.warning(f"MQTT publish failed, attempt {attempt+1}/{max_retries}")
                time.sleep(0.5)  # Short delay before retry
            except Exception as e:
                logger.error(f"Error in MQTT publish attempt {attempt+1}: {e}")
        
        return False
    
    def subscribe(self, topic: str, qos: int = 0, callback: Optional[Callable] = None) -> bool:
        """
        Subscribe to an MQTT topic.
        
        Args:
            topic: The topic to subscribe to
            qos: Quality of Service level
            callback: Optional callback function to handle received messages
            
        Returns:
            bool: True if subscription was successful, False otherwise
        """
        if not self.connected:
            logger.warning("Cannot subscribe: MQTT client is not connected")
            return False
        
        # Set custom message callback if provided
        if callback:
            def on_message_wrapper(client, userdata, msg):
                callback(msg.topic, msg.payload)
            self.client.message_callback_add(topic, on_message_wrapper)
        
        try:
            result, mid = self.client.subscribe(topic, qos)
            if result != mqtt.MQTT_ERR_SUCCESS:
                logger.error(f"Failed to subscribe to {topic}: {mqtt.error_string(result)}")
                return False
            logger.info(f"Subscribed to topic: {topic} with QoS {qos}")
            return True
        except Exception as e:
            logger.error(f"Error subscribing to topic {topic}: {e}")
            return False
        
    def is_connected(self) -> bool:
        """
        Returns whether the MQTT client is currently connected.
        """
        return self.connected
