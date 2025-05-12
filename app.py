from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import torch.nn as nn
from torchvision import models
import time
from azure.storage.blob import BlobServiceClient, BlobClient

app = Flask(__name__)
CORS(app)  # This allows cross-origin requests

# Define the class names
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Define Azure Storage settings
AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=octvisionmodell;AccountKey=p8Q2YuoCnA1AV9EOlMftfXD+avNj9uGv78YKZ3i771vk4Kpj7VPDMaBP/ZhzA9YN79UKNz6ZywkE+AStEDCCLw==;EndpointSuffix=core.windows.net"
AZURE_CONTAINER_NAME = "model"
AZURE_BLOB_NAME = "VGG16_v2-OCT_Retina_half_dataset.pt"

# Function to load model directly from Azure Blob Storage
def load_model_from_blob():
    try:
        print("Starting model loading from Azure Blob Storage...")
        start_time = time.time()
        
        # Create the BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        
        # Get a client to interact with the specific container
        container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
        
        # Get a client to interact with the specific blob
        blob_client = container_client.get_blob_client(AZURE_BLOB_NAME)
        
        print(f"Connected to blob: {AZURE_BLOB_NAME}")
        
        # Download the blob content directly to memory
        blob_data = blob_client.download_blob()
        model_bytes = blob_data.readall()
        
        print(f"Model data loaded from blob storage: {len(model_bytes)/1024/1024:.2f} MB")
        
        # Create the model architecture
        vgg16 = models.vgg16_bn(pretrained=False)
        
        # Modify the classifier
        num_features = vgg16.classifier[6].in_features
        features = list(vgg16.classifier.children())[:-1]  # Remove last layer
        features.extend([nn.Linear(num_features, len(class_names))])  # Add our layer with 4 outputs
        vgg16.classifier = nn.Sequential(*features)
        
        # Load the trained weights from the blob data
        vgg16.load_state_dict(torch.load(
            io.BytesIO(model_bytes), 
            map_location=torch.device('cpu')
        ))
        
        elapsed = time.time() - start_time
        print(f"Model loaded successfully in {elapsed:.2f} seconds")
        
        # Set the model to evaluation mode
        vgg16.eval()
        return vgg16
    
    except Exception as e:
        print(f"Error loading model from blob storage: {str(e)}")
        raise

# Function to load model with retry logic
def load_model_with_retry(max_retries=3):
    """Load model with retries to handle potential transient network issues"""
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            return load_model_from_blob()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Model loading failed, retrying in {retry_delay} seconds... ({attempt+1}/{max_retries})")
                print(f"Error: {str(e)}")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"All {max_retries} retries failed. Last error: {str(e)}")
                raise

# Transform for input images
def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

# Initialize model and transform
model = None
transform = get_transform()

@app.route('/predict', methods=['POST'])
def predict():
    global model
    
    # Lazy load the model on first request
    if model is None:
        try:
            model = load_model_with_retry()
        except Exception as e:
            return jsonify({'error': f"Failed to load model: {str(e)}"}), 500

    # Check if image was sent in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read and process the image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)
            pred_class = class_names[preds.item()]
            
            # Get probabilities for all classes
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            class_probs = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
            
        return jsonify({
            'prediction': pred_class,
            'probabilities': class_probs
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/model-info', methods=['GET'])
def model_info():
    """Return information about the model without loading it"""
    return jsonify({
        'model_name': 'VGG16_v2-OCT_Retina',
        'storage_location': f"{AZURE_CONTAINER_NAME}/{AZURE_BLOB_NAME}",
        'classes': class_names,
        'status': 'loaded' if model is not None else 'not_loaded'
    })

@app.route('/', methods=['GET'])
def home():
    return """
    <html>
        <head>
            <title>OCT Retina Classification API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>OCT Retina Classification API</h1>
            <p>This API provides retina OCT image classification into four categories:</p>
            <ul>
                <li>CNV (Choroidal Neovascularization)</li>
                <li>DME (Diabetic Macular Edema)</li>
                <li>DRUSEN</li>
                <li>NORMAL</li>
            </ul>
            <h2>Endpoints:</h2>
            <ul>
                <li><code>POST /predict</code> - Upload an image for classification</li>
                <li><code>GET /health</code> - Health check endpoint</li>
                <li><code>GET /model-info</code> - Information about the model</li>
            </ul>
            <p>To use this API, send a POST request to /predict with an image file.</p>
        </body>
    </html>
    """

if __name__ == '__main__':
    # Use the PORT environment variable provided by Azure App Service or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
