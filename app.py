import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gradio as gr

from model import Plant_Disease_Model2

# Initialize and load the trained model
model = Plant_Disease_Model2()
model_path = "plantDiseaseDetection.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define the image transformation (must match training transform)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # If you used normalization during training, add it here:
    # transforms.Normalize(mean=[...], std=[...])
])

# List of class names as provided
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def predict(image: Image.Image):
    """
    Takes an image input, processes it, and returns the predicted plant disease class.
    """
    if image is None:
        return "No image provided."
    
    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    return f"Predicted: {predicted_class} (Confidence: {confidence.item():.2f})"

# Create a Gradio interface for the API
iface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(type="pil"), 
    outputs="text", 
    title="Plant Disease Detection", 
    description="Upload a plant image to detect disease."
)

if __name__ == "__main__":
    iface.launch(share=True)

