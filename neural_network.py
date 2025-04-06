import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# Set up logging
logger = logging.getLogger(__name__)

# Define possible operators and digits for a simple math model
DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
OPERATORS = ['+', '-', '*', '/', '=', 'x', 'y']
SYMBOLS = DIGITS + OPERATORS
SYMBOL_TO_IDX = {symbol: idx for idx, symbol in enumerate(SYMBOLS)}
IDX_TO_SYMBOL = {idx: symbol for idx, symbol in enumerate(SYMBOLS)}
NUM_CLASSES = len(SYMBOLS)

def create_math_recognition_model(input_shape=(224, 224, 3)):
    """
    Creates a CNN model for recognizing mathematical symbols.
    
    Args:
        input_shape: Input image shape, default (224, 224, 3)
    
    Returns:
        Compiled TensorFlow model
    """
    # Use MobileNetV2 as base model for transfer learning
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Build the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def predict_symbols(model, preprocessed_image):
    """
    Predict symbols from a preprocessed image.
    
    Args:
        model: Trained TensorFlow model
        preprocessed_image: Preprocessed input image
        
    Returns:
        List of predicted symbols with confidence scores
    """
    # Get raw predictions
    predictions = model.predict(preprocessed_image)
    
    # Get top predictions
    top_indices = np.argsort(predictions[0])[-5:][::-1]  # Get indices of top 5 predictions
    
    # Format results
    results = [
        {"symbol": IDX_TO_SYMBOL[idx], 
         "confidence": float(predictions[0][idx])}
        for idx in top_indices
    ]
    
    return results