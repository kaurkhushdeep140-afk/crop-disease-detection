import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = load_model("crop_disease_model.keras")
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]
treatments = {
    "Apple___Apple_scab": "Spray Mancozeb or Captan fungicide. Remove infected leaves.",
    "Apple___Black_rot": "Remove infected fruits and use a recommended fungicide.",
    "Apple___healthy": "Plant is healthy. No treatment required.",
    "Potato___Early_blight": "Spray Chlorothalonil or Mancozeb.",
    "Potato___Late_blight": "Use Metalaxyl-based fungicide immediately.",
    "Tomato___Early_blight": "Use Copper fungicide and remove infected leaves.",
    "Tomato___Late_blight": "Avoid overhead watering and spray fungicide.",
    "Tomato___healthy": "Plant is healthy. No treatment required."
}
disease_info = {
    "Apple___Apple_scab": {
        "description": "Apple Scab is a fungal disease that causes dark, scabby spots on leaves and fruits. It reduces fruit quality and yield.",
        "prevention": "Remove infected leaves, prune trees for better airflow, and spray fungicide during the growing season."
    },

    "Apple___Black_rot": {
        "description": "Black Rot is a fungal disease that affects leaves, branches, and fruits, causing black lesions.",
        "prevention": "Remove infected fruits and branches, maintain orchard hygiene, and use recommended fungicides."
    },

    "Apple___healthy": {
        "description": "The plant is healthy with no visible disease symptoms.",
        "prevention": "Continue proper watering, balanced fertilization, and regular monitoring."
    },

    "Tomato___Early_blight": {
        "description": "Early Blight is a fungal disease causing brown spots with concentric rings on leaves.",
        "prevention": "Avoid overhead watering, rotate crops, and remove infected leaves."
    },

    "Tomato___Late_blight": {
        "description": "Late Blight spreads rapidly in cool, wet weather and damages leaves and fruits.",
        "prevention": "Use certified seeds, improve air circulation, and apply fungicides."
    },

    "Tomato___healthy": {
        "description": "The tomato plant is healthy.",
        "prevention": "Maintain proper irrigation and balanced nutrients."
    }
}
@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["image"]

    if image:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
        image.save(filepath)

        # Load and preprocess image
        img = Image.open(filepath).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        disease = class_names[predicted_index]
        recommendation = treatments.get(disease, "No recommendation available.")
        description = disease_info.get(disease, {}).get(
           "description",
            "No description available.")
        prevention = disease_info.get(
            disease, 
            {}
            ).get(
                "prevention",
                "No prevention tips available."
                )
        return render_template(
    "index.html",
    disease=disease,
    confidence=f"{confidence:.2f}",
    image=image.filename,
    recommendation=recommendation,
     description=description,
    prevention=prevention
)
    

    return "No image selected."



if __name__ == "__main__":
    app.run(debug=True)