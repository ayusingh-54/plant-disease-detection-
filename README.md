# 🌾 KisanMitra Backend

> A PyTorch-based plant disease detection API that uses deep learning to identify diseases in plant images

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com)
[![Gradio](https://img.shields.io/badge/Gradio-Latest-orange.svg)](https://gradio.app)

KisanMitra Backend provides both a Gradio web interface and FastAPI endpoints for plant disease classification, empowering farmers with AI-driven agricultural insights.

## 🌱 Overview

KisanMitra Backend is a machine learning service that can detect diseases in plants across 38 different categories, covering various crops including:

- **Fruits**: Apple, Blueberry, Cherry, Grape, Orange, Peach, Strawberry
- **Vegetables**: Corn, Pepper, Potato, Tomato, Squash
- **Other crops**: Raspberry, Soybean

The model can identify both healthy plants and various disease conditions with confidence scores.

## 🏗️ Architecture

The backend consists of several key components:

- **`model.py`**: Contains the Plant_Disease_Model2 class based on ResNet34 architecture
- **`app.py`**: Gradio interface for interactive web-based predictions
- **`main.py`**: FastAPI server setup with CORS configuration
- **`plantDiseaseDetection.pth`**: Pre-trained model weights

## 📋 Supported Disease Classes

The model can detect **38 different plant conditions** across multiple crop categories:

<details>
<summary><strong>🍎 Apple (4 classes)</strong></summary>

- Apple Scab
- Black Rot
- Cedar Apple Rust
- Healthy

</details>

<details>
<summary><strong>🌽 Corn/Maize (4 classes)</strong></summary>

- Cercospora Leaf Spot (Gray Leaf Spot)
- Common Rust
- Northern Leaf Blight
- Healthy

</details>

<details>
<summary><strong>🍅 Tomato (10 classes)</strong></summary>

- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites (Two-spotted Spider Mite)
- Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato Mosaic Virus
- Healthy

</details>

<details>
<summary><strong>🌱 Other Crops (20 classes)</strong></summary>

- **Blueberry**: Healthy
- **Cherry**: Powdery Mildew, Healthy
- **Grape**: Black Rot, Esca, Leaf Blight, Healthy
- **Orange**: Huanglongbing (Citrus Greening)
- **Peach**: Bacterial Spot, Healthy
- **Pepper Bell**: Bacterial Spot, Healthy
- **Potato**: Early Blight, Late Blight, Healthy
- **Raspberry**: Healthy
- **Soybean**: Healthy
- **Squash**: Powdery Mildew
- **Strawberry**: Leaf Scorch, Healthy

</details>

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ayusingh-54/plant-disease-detection-.git
   cd plant-disease-detection-
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model file exists**:
   Make sure `plantDiseaseDetection.pth` is present in the project directory.

### Running the Application

#### Option 1: Gradio Interface (Recommended for testing)

```bash
python app.py
```

This will launch a Gradio web interface accessible at `http://localhost:7860` where you can:

- Upload plant images
- Get instant disease predictions
- View confidence scores

#### Option 2: FastAPI Server (For production/frontend integration)

```bash
python main.py
```

This starts the FastAPI server at `http://localhost:8000` with:

- RESTful API endpoints
- CORS enabled for frontend integration
- Interactive API documentation at `/docs`

## 📡 API Endpoints

### FastAPI Endpoints

- **GET `/`**: Health check endpoint

  ```json
  {
    "message": "Plant Disease Detection API is running!"
  }
  ```

- **POST `/predict`**: Image prediction endpoint
  - **Input**: Multipart form data with image file
  - **Output**: JSON with prediction and confidence score

### Gradio Interface

The Gradio interface provides:

- **Input**: Image upload (JPG, PNG formats supported)
- **Output**: Predicted disease class with confidence score
- **Format**: `"Predicted: {class_name} (Confidence: {score:.2f})"`

## 🔧 Model Details

### Architecture

- **Base Model**: ResNet34
- **Input Size**: 128x128 pixels
- **Output Classes**: 38 disease categories
- **Framework**: PyTorch

### Preprocessing Pipeline

```python
transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # Note: Normalization parameters should match training
])
```

### Model Loading

```python
model = Plant_Disease_Model2()
model.load_state_dict(torch.load("plantDiseaseDetection.pth", map_location='cpu'))
model.eval()
```

## 📁 File Structure

```
plant-disease-detection-/
├── app.py                      # Gradio interface
├── main.py                     # FastAPI server
├── model.py                    # Model architecture
├── plantDiseaseDetection.pth   # Trained model weights
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## 🛠️ Development

### Model Architecture Details

The `Plant_Disease_Model2` class extends `ImageClassificationBase` and includes:

- **Training methods**: `training_step()`, `validation_step()`, `validation_epoch_end()`
- **Forward pass**: ResNet34 backbone with custom final layer (38 classes)
- **Loss function**: Cross-entropy loss
- **Metrics**: Accuracy calculation

### Adding New Features

1. **New disease classes**: Update `class_names` list in `app.py`
2. **Different preprocessing**: Modify `transform` pipeline
3. **Model improvements**: Extend `Plant_Disease_Model2` class in `model.py`

## 🚨 Important Notes

### Performance Considerations

- **CPU vs GPU**: Model loads with CPU by default (`map_location='cpu'`)
- **Image size**: Input images are resized to 128x128 pixels
- **Batch processing**: Current implementation processes single images

### Model Limitations

- **Input format**: Accepts PIL Images only
- **Confidence threshold**: No built-in confidence filtering
- **Preprocessing**: Must match training preprocessing exactly

## 🔧 Configuration

### Environment Variables

Consider setting up environment variables for:

- Model path location
- API host/port settings
- CORS origins for production

### Production Deployment

For production deployment:

1. Update CORS origins in `main.py`
2. Set up proper error handling
3. Configure logging
4. Add authentication if needed
5. Set up model versioning

## 📊 Model Performance

The model provides confidence scores for predictions. Higher confidence (closer to 1.0) indicates more certain predictions.

**Example output:**

```json
{
  "prediction": "Tomato___Bacterial_spot",
  "confidence": 0.95,
  "formatted": "Predicted: Tomato Bacterial Spot (Confidence: 95%)"
}
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

<details>
<summary><strong>Common Issues & Solutions</strong></summary>

### Model Loading Issues

- **Problem**: `FileNotFoundError: plantDiseaseDetection.pth`
- **Solution**: Ensure the model file exists in the project directory

### CUDA/GPU Issues

- **Problem**: CUDA out of memory or unavailable
- **Solution**: Model loads on CPU by default (`map_location='cpu'`)

### Dependency Issues

- **Problem**: Import errors or missing packages
- **Solution**: Run `pip install -r requirements.txt`

### Port Conflicts

- **Problem**: Port already in use
- **Solution**:
  - Gradio: Default port 7860
  - FastAPI: Default port 8000
  - Change ports in respective files if needed

</details>

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/ayusingh-54/plant-disease-detection-/issues)
- **Email**: [Contact Developer](mailto:your-email@example.com)
- **Documentation**: Check the code comments in `model.py` and `app.py`

---

## 👨‍💻 Developer

**Ayushi Singh** - Lead Developer and Creator of KisanMitra

- GitHub: [@ayusingh-54](https://github.com/ayusingh-54)

---

<div align="center">

**Built with ❤️ using PyTorch, Gradio, and FastAPI**

_KisanMitra - Empowering farmers with AI-driven plant disease detection_

[![GitHub stars](https://img.shields.io/github/stars/ayusingh-54/plant-disease-detection-.svg?style=social&label=Star)](https://github.com/ayusingh-54/plant-disease-detection-)
[![GitHub forks](https://img.shields.io/github/forks/ayusingh-54/plant-disease-detection-.svg?style=social&label=Fork)](https://github.com/ayusingh-54/plant-disease-detection-)

</div>

