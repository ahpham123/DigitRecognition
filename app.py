from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from neuralnetwork import load_parameters, forward_prop, make_prediction

app = Flask(__name__)
CORS(app)

try:
    W1, b1, W2, b2 = load_parameters()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    W1, b1, W2, b2 = None, None, None, None

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
        
        if not pixel_data:
            return jsonify({'error': 'No pixel data provided'}), 400
        
        img_array = np.array(pixel_data)
        img_array = img_array.flatten().reshape(784, 1) / 255.0
        
        prediction, probabilities = make_prediction(img_array, W1, b1, W2, b2)
        probs_dict = {str(i): float(probabilities[i][0]) for i in range(10)}
        
        print(f"Prediction: {prediction[0]}, Confidence: {probabilities[prediction[0]][0]:.4f}")
        print(f"Top 3 predictions: {sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)[:3]}")
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probabilities': probs_dict,
            'confidence': float(probabilities[prediction[0]][0])
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)