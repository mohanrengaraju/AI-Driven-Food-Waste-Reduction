from flask import Blueprint, request, jsonify
from app.services.ai_service import predict_food

ai_bp = Blueprint('ai', __name__)

@ai_bp.route('/predict', methods=['POST'])
def predict():
    print("Request method:", request.method)
    print("Request content type:", request.content_type)
    print("Request files:", request.files)

    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400
    
    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    prediction = predict_food(image)
    return jsonify({'prediction': prediction})
