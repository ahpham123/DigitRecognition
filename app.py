from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
from neuralnetwork import load_parameters, forward_prop, make_prediction
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

try:
    W1, b1, W2, b2 = load_parameters()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    W1, b1, W2, b2 = None, None, None, None

def preprocess_for_mnist(pixel_data, return_debug=False):
    """
    Preprocess drawing to match MNIST format with debugging capability
    """
    img_array = np.array(pixel_data, dtype=np.float32)
    
    debug_images = {}
    if return_debug:
        debug_images['original'] = img_array.copy()
    
    # Find bounding box of drawn content
    rows = np.any(img_array > 0, axis=1)
    cols = np.any(img_array > 0, axis=0)
    
    # If empty image, return zeros
    if not rows.any() or not cols.any():
        return np.zeros((784, 1))
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Add small padding
    padding = 2
    rmin = max(0, rmin - padding)
    rmax = min(27, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(27, cmax + padding)
    
    # Crop to bounding box
    cropped = img_array[rmin:rmax+1, cmin:cmax+1]
    
    if return_debug:
        debug_images['cropped'] = cropped.copy()
    
    # Calculate new size to fit in 20x20 while maintaining aspect ratio
    h, w = cropped.shape
    
    if h > w:
        new_h = 20
        new_w = max(1, int(20 * w / h))
    else:
        new_w = 20
        new_h = max(1, int(20 * h / w))
    
    # Resize using zoom
    if h > 0 and w > 0:
        scale_h = new_h / h
        scale_w = new_w / w
        resized = zoom(cropped, (scale_h, scale_w), order=1)
    else:
        resized = cropped
    
    if return_debug:
        debug_images['resized'] = resized.copy()
    
    # Calculate center of mass
    if resized.sum() > 0:
        rows_mass = np.sum(resized, axis=1)
        cols_mass = np.sum(resized, axis=0)
        
        row_indices = np.arange(resized.shape[0])
        col_indices = np.arange(resized.shape[1])
        
        cy = int(np.sum(row_indices * rows_mass) / rows_mass.sum())
        cx = int(np.sum(col_indices * cols_mass) / cols_mass.sum())
    else:
        cy, cx = resized.shape[0] // 2, resized.shape[1] // 2
    
    # Create 28x28 black canvas
    centered = np.zeros((28, 28))
    
    # Center based on center of mass
    y_offset = 14 - cy
    x_offset = 14 - cx
    
    # Calculate where to place the resized image
    y_start = max(0, y_offset)
    x_start = max(0, x_offset)
    y_end = min(28, y_offset + resized.shape[0])
    x_end = min(28, x_offset + resized.shape[1])
    
    # Calculate corresponding region in resized image
    resized_y_start = max(0, -y_offset)
    resized_x_start = max(0, -x_offset)
    resized_y_end = resized_y_start + (y_end - y_start)
    resized_x_end = resized_x_start + (x_end - x_start)
    
    # Place the image
    centered[y_start:y_end, x_start:x_end] = \
        resized[resized_y_start:resized_y_end, resized_x_start:resized_x_end]
    
    if return_debug:
        debug_images['before_blur'] = centered.copy()
    
    # Apply Gaussian blur
    centered = gaussian_filter(centered, sigma=0.5)
    
    if return_debug:
        debug_images['final'] = centered.copy()
    
    # Normalize to 0-1
    normalized = centered.flatten().reshape(784, 1) / 255.0
    
    if return_debug:
        return normalized, debug_images
    return normalized

def array_to_base64(arr):
    """Convert numpy array to base64 image string"""
    # Normalize to 0-255
    img_normalized = ((arr / arr.max()) * 255).astype(np.uint8) if arr.max() > 0 else arr.astype(np.uint8)
    img = Image.fromarray(img_normalized, mode='L')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': W1 is not None})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if W1 is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        pixel_data = data.get('pixels')
        debug_mode = data.get('debug', False)
        
        if not pixel_data:
            return jsonify({'error': 'No pixel data provided'}), 400
        
        # Preprocess to match MNIST format
        if debug_mode:
            img_array, debug_images = preprocess_for_mnist(pixel_data, return_debug=True)
            # Convert debug images to base64
            debug_b64 = {key: array_to_base64(img) for key, img in debug_images.items()}
        else:
            img_array = preprocess_for_mnist(pixel_data)
            debug_b64 = None
        
        # Make prediction
        prediction, probabilities = make_prediction(img_array, W1, b1, W2, b2)
        probs_dict = {str(i): float(probabilities[i][0]) for i in range(10)}
        
        # Get top 3 predictions
        top_3 = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"Prediction: {prediction[0]}, Confidence: {probabilities[prediction[0]][0]:.4f}")
        print(f"Top 3 predictions: {top_3}")
        
        response = {
            'prediction': int(prediction[0]),
            'probabilities': probs_dict,
            'confidence': float(probabilities[prediction[0]][0])
        }
        
        if debug_mode:
            response['debug_images'] = debug_b64
            response['top_3'] = [(int(label), float(prob)) for label, prob in top_3]
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)