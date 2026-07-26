from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import json
from PIL import Image

app = Flask(__name__)

model = load_model("model/best_model.keras")
model.summary()

print("INPUT SHAPE:", model.input_shape)

with open("model/class_indices.json","r") as f:
    class_indices = json.load(f)

classes = class_indices
disease_info = {

    "Tomato_healthy": {
        "severity": "None",
        "description": "The tomato plant is healthy with no visible signs of disease.",
        "cause": "No disease detected.",
        "treatments": [
            "No treatment is required.",
            "Continue proper irrigation and fertilization.",
            "Inspect plants regularly for early symptoms."
        ],
        "prevention": "Maintain balanced nutrition and good field hygiene.",
        "urgency": "Low"
    },

    "Tomato_Bacterial_spot": {
        "severity": "Medium",
        "description": "Bacterial Spot causes dark brown to black spots on leaves and fruits.",
        "cause": "Caused by Xanthomonas bacteria.",
        "treatments": [
            "Apply copper-based bactericides.",
            "Remove infected leaves.",
            "Avoid working in wet fields."
        ],
        "prevention": "Use disease-free seeds and practice crop rotation.",
        "urgency": "Medium"
    },

    "Tomato_Early_blight": {
        "severity": "Medium",
        "description": "Early Blight produces brown concentric-ring lesions on older leaves.",
        "cause": "Caused by Alternaria solani fungus.",
        "treatments": [
            "Apply recommended fungicides.",
            "Remove infected leaves.",
            "Improve air circulation."
        ],
        "prevention": "Avoid overhead watering and rotate crops.",
        "urgency": "Medium"
    },

    "Tomato_Late_blight": {
        "severity": "High",
        "description": "Late Blight spreads rapidly and can destroy entire tomato crops.",
        "cause": "Caused by Phytophthora infestans.",
        "treatments": [
            "Apply systemic fungicides immediately.",
            "Destroy severely infected plants.",
            "Avoid excess moisture."
        ],
        "prevention": "Ensure proper spacing and monitor fields during humid weather.",
        "urgency": "High"
    },

    "Tomato_Leaf_Mold": {
        "severity": "Medium",
        "description": "Leaf Mold causes yellow patches on leaves with olive-green fungal growth underneath.",
        "cause": "Caused by Passalora fulva fungus.",
        "treatments": [
            "Remove infected foliage.",
            "Apply fungicide.",
            "Increase ventilation and reduce humidity."
        ],
        "prevention": "Avoid prolonged leaf wetness and maintain airflow.",
        "urgency": "Medium"
    },

    "Tomato_Septoria_leaf_spot": {
        "severity": "Medium",
        "description": "Septoria Leaf Spot appears as many small circular spots with gray centers.",
        "cause": "Caused by Septoria lycopersici fungus.",
        "treatments": [
            "Apply fungicides.",
            "Remove infected leaves.",
            "Avoid splashing water onto foliage."
        ],
        "prevention": "Rotate crops and keep the field clean.",
        "urgency": "Medium"
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "severity": "Medium",
        "description": "Spider mites suck plant sap, causing yellow speckling and webbing.",
        "cause": "Infestation by two-spotted spider mites.",
        "treatments": [
            "Use miticides if infestation is severe.",
            "Spray water on leaf undersides.",
            "Encourage natural predators."
        ],
        "prevention": "Maintain proper irrigation and monitor plants regularly.",
        "urgency": "Medium"
    },

    "Tomato_Target_Spot": {
        "severity": "Medium",
        "description": "Target Spot creates circular lesions that reduce plant productivity.",
        "cause": "Caused by Corynespora cassiicola fungus.",
        "treatments": [
            "Apply fungicides.",
            "Remove infected debris.",
            "Improve field sanitation."
        ],
        "prevention": "Ensure adequate plant spacing.",
        "urgency": "Medium"
    },

    "Tomato_Tomato_YellowLeaf_Curl_Virus": {
        "severity": "High",
        "description": "Leaves curl upward and turn yellow, greatly reducing fruit production.",
        "cause": "Spread by whiteflies carrying Tomato Yellow Leaf Curl Virus.",
        "treatments": [
            "Control whiteflies.",
            "Remove infected plants.",
            "Use resistant varieties."
        ],
        "prevention": "Monitor whitefly populations and use insect-proof nets.",
        "urgency": "High"
    },

    "Tomato_Tomato_mosaic_virus": {
        "severity": "High",
        "description": "Tomato Mosaic Virus causes mottled leaves, distortion, and poor fruit quality.",
        "cause": "Caused by Tomato Mosaic Virus (ToMV).",
        "treatments": [
            "Remove infected plants.",
            "Disinfect tools.",
            "Avoid handling healthy plants after infected ones."
        ],
        "prevention": "Use certified virus-free seeds and maintain sanitation.",
        "urgency": "High"
    }
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    img = Image.open(file.stream).convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img, dtype=np.float32)

    print("Image Shape:", img.shape)
    print("Minimum Pixel:", img.min())
    print("Maximum Pixel:", img.max())

    img = img / 255.0

    img = np.expand_dims(img, axis=0)     

    prediction = model.predict(img)

    print("\nPredictions:")

    for i, p in enumerate(prediction[0]):
        print(classes[str(i)], ":", p)

    print("Predicted class:", np.argmax(prediction))
    print("Class mapping:", classes)

    predicted_class = np.argmax(prediction)

    confidence = float(np.max(prediction)*100)

    disease = classes[str(predicted_class)]
    
    print("Disease:", disease)
    print("Available Keys:", disease_info.keys())

    info = disease_info.get(disease)

    return jsonify({
    "disease": disease.replace("_", " "),
    "confidence": round(confidence, 2),
    "severity": info["severity"],
    "description": info["description"],
    "cause": info["cause"],
    "treatments": info["treatments"],
    "prevention": info["prevention"],
    "urgency": info["urgency"],
    "govt_scheme": "PM Fasal Bima Yojana"
})

if __name__=="__main__":
    app.run(debug=True)