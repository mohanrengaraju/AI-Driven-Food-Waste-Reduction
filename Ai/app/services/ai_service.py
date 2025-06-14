# server/app/services/ai_service.py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import io

# OLD LINE:
# model = load_model(r'F:\Mohan files\Sixth sem\NM\model1.h5')
# NEW LINE:
model = load_model(r'F:\Mohan files\Sixth sem\NM\model1_finetuned.h5') # <--- UPDATE THIS PATH

# Make sure this list is still accurate for the 9 classes your model was fine-tuned on.
# The order should correspond to the subdirectories found by train_generator.class_indices
# You can print train_generator.class_indices in model_train.py to confirm the order.
# Example: if class_indices was {'Biryani':0, 'Chole-Bhature':1, ...} then class_names[0] must be 'Biryani' etc.
# The ImageDataGenerator sorts folder names alphabetically to assign indices.
class_names = sorted(['Biryani', 'Pani Puri', 'Vada Pav', 'Dal', 'Naan', 'Dosa', 'Paneer-Tikka', 'Pav-Bhaji', 'Chole-Bhature'])
# It's safer to get this directly from the training script's output of train_generator.class_indices
# For example if train_generator.class_indices was:
# {'Biriyani': 0, 'Chole-Bhature': 1, 'Dal': 2, 'Dosa': 3, 'Naan': 4, 'Paneer-Tikka': 5, 'Pav-Bhaji': 6, 'Pani Puri': 7, 'Vada Pav': 8}
# Then your class_names should be:
# class_names = ['Biriyani', 'Chole-Bhature', 'Dal', 'Dosa', 'Naan', 'Paneer-Tikka', 'Pav-Bhaji', 'Pani Puri', 'Vada Pav']
# Double check your model_train.py output for the exact `Class indices` map.

CONFIDENCE_THRESHOLD = 0.60 # You can adjust this

def predict_food(image_file):
    image_file.seek(0)
    image_bytes = image_file.read()
    image_stream = io.BytesIO(image_bytes)
    
    image = load_img(image_stream, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    predictions_array = model.predict(image)[0]
    predicted_class_index = np.argmax(predictions_array)
    confidence = float(np.max(predictions_array))

    # Ensure class_names list is correctly ordered according to the model's training
    # If you printed train_generator.class_indices during training, use that order.
    # For example, if it was {'Biriyani': 0, 'Dal': 1, 'Dosa': 2, ...}, then:
    # class_names_from_training = ['Biriyani', 'Dal', 'Dosa', ...] # Based on your actual training output
    
    # The safest way is to hardcode the class_names in the exact order the model was trained on.
    # Refer to the output of `train_generator.class_indices` from your successful training run.
    # Example, if the output was:
    # Class indices: {'Biriyani': 0, 'Chole-Bhature': 1, 'Dal': 2, 'Dosa': 3, 'Naan': 4, 'Paneer-Tikka': 5, 'Pav-Bhaji': 6, 'Pani Puri': 7, 'Vada Pav': 8}
    # Then:
    class_names = ['Biriyani', 'Chole-Bhature', 'Dal', 'Dosa', 'Naan', 'Paneer-Tikka', 'Pav-Bhaji', 'Pani Puri', 'Vada Pav']
    # ^^^ UPDATE THIS LIST BASED ON YOUR `model_train.py` OUTPUT FOR `train_generator.class_indices`


    if confidence >= CONFIDENCE_THRESHOLD:
        label = class_names[predicted_class_index]
        expiry_risk = "Low" if confidence > 0.8 else "Medium"
    else:
        label = "Uncertain (low confidence)"
        expiry_risk = "High (prediction uncertain)"

    quantity = "1 plate"

    return {
        "food_type": label,
        "confidence": round(confidence, 2),
        "quantity": quantity,
        "expiry_risk": expiry_risk,
        "predicted_class_raw": class_names[predicted_class_index]
    }