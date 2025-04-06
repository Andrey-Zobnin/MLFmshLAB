import os
import io
import logging
import numpy as np
import tensorflow as tf
import sympy as sp
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from PIL import Image, ImageOps, ImageFilter
import cv2
from functools import wraps
# Import neural network module
from neural_network import create_math_recognition_model, predict_symbols, SYMBOLS, DIGITS, OPERATORS

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

# Define model variables - in-memory model without saving to file system
MATH_MODEL_PATH = None # Model will be created in memory directly

#####################################################################
# TRANSLATIONS
#####################################################################

TRANSLATIONS = {
    'en': {
        'Image Recognition App': 'Math Equation Solver',
        'Image Recognition': 'Math Equation Recognition',
        'Upload an image and our AI will identify what\'s in the picture.': 'Upload an image with a math equation and our AI will solve it.',
        'We use a pre-trained machine learning model to recognize common objects, animals, and scenes.': 'We use a neural network to recognize and solve mathematical equations from images.',
        'Preview': 'Preview',
        'Remove': 'Remove',
        'Drag and drop an image or click to browse': 'Drag and drop an image with a math equation or click to browse',
        'Browse Files': 'Browse Files',
        'Analyze Image': 'Solve Equation',
        'Recognition Results': 'Solution Results',
        'Loading...': 'Loading...',
        'Analyzing image...': 'Solving equation...',
        'Language': 'Language',
        'Dark Mode': 'Dark Mode',
        'Light Mode': 'Light Mode',
        'Please select an image file (JPG, PNG, etc.)': 'Please select an image file (JPG, PNG, etc.)',
        'Error': 'Error',
        'No file uploaded': 'No file uploaded',
        'Empty filename': 'Empty filename',
        'Model not loaded': 'Model not loaded',
        'Math recognition model not loaded': 'Math recognition model not loaded',
        'No symbols detected in the image': 'No mathematical symbols detected in the image',
        'Equation': 'Equation',
        'Solution': 'Solution',
    },
    'ru': {
        'Image Recognition App': 'Решатель Математических Уравнений',
        'Image Recognition': 'Распознавание Математических Уравнений',
        'Upload an image and our AI will identify what\'s in the picture.': 'Загрузите изображение с математическим уравнением, и наш ИИ решит его.',
        'We use a pre-trained machine learning model to recognize common objects, animals, and scenes.': 'Мы используем нейронную сеть для распознавания и решения математических уравнений из изображений.',
        'Preview': 'Предпросмотр',
        'Remove': 'Удалить',
        'Drag and drop an image or click to browse': 'Перетащите изображение с уравнением или нажмите для выбора',
        'Browse Files': 'Обзор Файлов',
        'Analyze Image': 'Решить Уравнение',
        'Recognition Results': 'Результаты Решения',
        'Loading...': 'Загрузка...',
        'Analyzing image...': 'Решение уравнения...',
        'Language': 'Язык',
        'Dark Mode': 'Темный Режим',
        'Light Mode': 'Светлый Режим',
        'Please select an image file (JPG, PNG, etc.)': 'Пожалуйста, выберите файл изображения (JPG, PNG и т.д.)',
        'Error': 'Ошибка',
        'No file uploaded': 'Файл не загружен',
        'Empty filename': 'Пустое имя файла',
        'Model not loaded': 'Модель не загружена',
        'Math recognition model not loaded': 'Модель распознавания математических уравнений не загружена',
        'No symbols detected in the image': 'В изображении не обнаружено математических символов',
        'Equation': 'Уравнение',
        'Solution': 'Решение',
    }
}

#####################################################################
# MATH MODEL FUNCTIONS 
#####################################################################

# Use symbols imported from ai module
from neural_network import SYMBOL_TO_IDX, IDX_TO_SYMBOL, NUM_CLASSES

def create_equation_solver_model():
    """
    Creates a simple model for equation solving.
    
    Returns:
        A function that can solve equations given recognized symbols
    """
    def solve_equation(symbols):
        """
        Solves a mathematical equation.
        
        Args:
            symbols: List of recognized symbols
        
        Returns:
            Dictionary with equation and solution
        """
        try:
            # Join the symbols to form an equation string
            equation_str = ''.join(symbols)
            
            # Handle different equation types
            if '=' in equation_str:
                # Simple equation solving
                left_side, right_side = equation_str.split('=')
                
                if 'x' in equation_str:
                    # Use sympy to solve for x
                    x = sp.symbols('x')
                    try:
                        # Replace the '*' symbol if needed for sympy parsing
                        if '*' in left_side:
                            left_side = left_side.replace('*', '*')
                        if '*' in right_side:
                            right_side = right_side.replace('*', '*')
                            
                        # Create a sympy equation
                        eq = sp.Eq(sp.sympify(left_side), sp.sympify(right_side))
                        
                        # Solve the equation
                        solution = sp.solve(eq, x)
                        
                        if solution:
                            solution_str = f"x = {solution[0]}"
                        else:
                            solution_str = "No solution found"
                    except Exception as e:
                        # Fallback if sympy fails
                        logger.error(f"Sympy error: {str(e)}")
                        solution_str = f"Could not solve: {str(e)}"
                else:
                    # Simple arithmetic verification
                    try:
                        left_eval = eval(left_side)
                        right_eval = eval(right_side)
                        solution_str = f"{left_eval} {'=' if left_eval == right_eval else '≠'} {right_eval}"
                    except Exception as e:
                        solution_str = f"Error evaluating: {str(e)}"
            
            else:
                # Just arithmetic expression
                try:
                    result = eval(equation_str)
                    solution_str = f"{result}"
                except Exception as e:
                    solution_str = f"Error: {str(e)}"
                
            return {
                "equation": equation_str,
                "solution": solution_str
            }
            
        except Exception as e:
            logger.error(f"Error solving equation: {e}")
            return {
                "equation": ''.join(symbols),
                "error": str(e)
            }
    
    return solve_equation

# predict_symbols function moved to neural_network.py

#####################################################################
# IMAGE PROCESSING FUNCTIONS
#####################################################################

def preprocess_image(file, for_math_recognition=True):
    """
    Preprocess image for either math recognition or general image classification:
    1. Convert to grayscale (for math) or RGB (for general classification)
    2. Resize to 224x224 pixels
    3. Apply preprocessing specific to the task
    4. Convert to numpy array and normalize
    5. Add batch dimension
    
    Args:
        file: The uploaded file object
        for_math_recognition: If True, apply preprocessing for math equation recognition
        
    Returns:
        Preprocessed image tensor
    """
    try:
        # Read image from uploaded file
        image_stream = file.read()
        image = Image.open(io.BytesIO(image_stream))
        
        if for_math_recognition:
            # Convert to grayscale for math recognition
            image = image.convert("L")
            
            # Apply image enhancements for math equation recognition
            image = ImageOps.autocontrast(image)  # Improve contrast
            image = image.filter(ImageFilter.SHARPEN)  # Sharpen edges
            
            # Convert PIL image to OpenCV format for advanced processing
            img_np = np.array(image)
            
            # Apply thresholding to create binary image
            _, binary_img = cv2.threshold(img_np, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            
            # Noise reduction
            binary_img = cv2.medianBlur(binary_img, 3)
            
            # Convert back to PIL
            image = Image.fromarray(binary_img)
            
            # Resize to model input size
            image = image.resize((224, 224))
            
            # Convert to numpy array and normalize
            image_array = np.array(image) / 255.0
            
            # Expand to 3 channels (grayscale repeated in each channel)
            image_array = np.stack([image_array, image_array, image_array], axis=-1)
        else:
            # For general image classification, use RGB
            image = image.convert("RGB")
            
            # Resize to 224x224 (standard size for MobileNetV2)
            image = image.resize((224, 224))
            
            # Convert to numpy array and normalize (0-1)
            image_array = np.array(image) / 255.0
            
            # Preprocess for MobileNetV2
            image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        logger.debug(f"Preprocessed image shape: {image_array.shape}")
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def segment_equation(image_array):
    """
    Segment a preprocessed image into individual symbols.
    This is a simplified implementation - for production, this would
    use more sophisticated segmentation techniques.
    
    Args:
        image_array: Preprocessed image array
        
    Returns:
        List of individual symbols and their positions
    """
    try:
        # Convert to single-channel if it's RGB
        if len(image_array.shape) == 4:  # has batch dimension
            img = image_array[0]  # remove batch dimension
            if img.shape[-1] == 3:  # it's RGB
                img_gray = np.mean(img, axis=-1)
            else:
                img_gray = img
        else:
            img_gray = image_array
        
        # Convert to uint8 for OpenCV
        img_uint8 = (img_gray * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours left-to-right (for reading equation properly)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        # Extract individual symbols
        symbols = []
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip very small contours (likely noise)
            if w < 5 or h < 5:
                continue
            
            # Extract the symbol
            symbol_img = img_uint8[y:y+h, x:x+w]
            
            # Pad the symbol image to make it square
            max_dim = max(w, h)
            padded_img = np.zeros((max_dim, max_dim), dtype=np.uint8)
            
            # Calculate padding
            pad_x = (max_dim - w) // 2
            pad_y = (max_dim - h) // 2
            
            # Place the symbol in the padded image
            padded_img[pad_y:pad_y+h, pad_x:pad_x+w] = symbol_img
            
            # Resize to a standard size (28x28 is common for digit recognition)
            resized_img = cv2.resize(padded_img, (28, 28))
            
            # Store the symbol and its position
            symbols.append({
                'image': resized_img / 255.0,  # normalize to 0-1
                'position': (x, y, w, h)
            })
        
        return symbols
        
    except Exception as e:
        logger.error(f"Error segmenting equation: {e}")
        return []

#####################################################################
# FLASK ROUTES
#####################################################################

# Global variables for models
math_model = None
equation_solver = None

# Create a new model in memory
try:
    logger.info("Creating math recognition model...")
    
    # Create a new model
    math_model = create_math_recognition_model()
    
    # Create equation solver
    equation_solver = create_equation_solver_model()
    
    logger.info("Math recognition model ready!")
    
except Exception as e:
    logger.error(f"Error creating math recognition model: {e}")
    math_model = None
    equation_solver = None

# Translation function
def _(text):
    """Translate text based on current language."""
    lang = session.get('lang', 'en')
    if lang in TRANSLATIONS and text in TRANSLATIONS[lang]:
        return TRANSLATIONS[lang][text]
    return text

# Make the translation function available in templates
@app.context_processor
def inject_translation_function():
    return {'_': _}

@app.route('/set_language/<lang>')
def set_language(lang):
    """Set the language for the user."""
    if lang in TRANSLATIONS:
        session['lang'] = lang
    return redirect(request.referrer or url_for('index'))

@app.route('/toggle_theme')
def toggle_theme():
    """Toggle between light and dark theme."""
    current_theme = session.get('theme', 'dark')
    if current_theme == 'dark':
        session['theme'] = 'light'
    else:
        session['theme'] = 'dark'
    return redirect(request.referrer or url_for('index'))

@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    """
    Process the uploaded image, recognize math equation, and solve it.
    """
    if "file" not in request.files:
        error_msg = _("No file uploaded")
        return jsonify({"error": error_msg}), 400

    file = request.files["file"]
    if file.filename == "":
        error_msg = _("Empty filename")
        return jsonify({"error": error_msg}), 400

    try:
        # Preprocess the image for math equation recognition
        processed_image = preprocess_image(file, for_math_recognition=True)
        
        # Process and recognize the math equation
        if math_model is not None and equation_solver is not None:
            # Segment the equation into individual symbols
            segmented_symbols = segment_equation(processed_image)
            
            if not segmented_symbols:
                return jsonify({
                    "success": False, 
                    "error": _("No symbols detected in the image")
                }), 400
            
            # Recognize symbols using our math model
            recognized_symbols = []
            symbol_predictions = []
            
            # Process each segmented symbol
            for symbol_data in segmented_symbols:
                # Extract the symbol image
                symbol_img = symbol_data['image']
                
                # Convert to proper format for model
                symbol_img_expanded = np.expand_dims(
                    np.stack([symbol_img, symbol_img, symbol_img], axis=-1), 
                    axis=0
                )
                
                # Predict the symbol
                symbol_preds = predict_symbols(math_model, symbol_img_expanded)
                
                # Add the top prediction to our list
                if symbol_preds:
                    # Extract top predicted symbol
                    top_symbol = symbol_preds[0]['symbol']
                    recognized_symbols.append(top_symbol)
                    
                    # Add more details for debugging/display
                    symbol_predictions.append({
                        'symbol': top_symbol,
                        'position': symbol_data['position'],
                        'confidence': symbol_preds[0]['confidence']
                    })
            
            # Join symbols to form equation
            equation_str = ''.join(recognized_symbols)
            
            # Solve the equation
            solution = equation_solver(recognized_symbols)
            
            return jsonify({
                "success": True,
                "equation": equation_str,
                "solution": solution.get('solution', 'Could not solve'),
                "error": solution.get('error', None),
                "details": {
                    "recognized_symbols": symbol_predictions,
                    "symbol_count": len(recognized_symbols),
                    "available_symbols": SYMBOLS
                }
            })
        else:
            error_msg = _("Math recognition model not loaded")
            return jsonify({"error": error_msg}), 500
            
    except Exception as e:
        logger.error(f"Error processing math equation: {e}")
        error_prefix = _("Error")
        return jsonify({"error": f"{error_prefix}: {str(e)}"}), 500

@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy", 
        "math_model_loaded": math_model is not None,
        "equation_solver_loaded": equation_solver is not None
    })


if __name__ == "__main__":
    # The following line will be executed when the file is imported by another script
    # This block will only be executed when the file is run directly
    # It won't be executed when imported by Gunicorn
    app.run(host='0.0.0.0', port=5000, debug=True)