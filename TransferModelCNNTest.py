import io
from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model class
class PaddyDiseaseModel(nn.Module):
    def __init__(self, num_classes=13):
        super(PaddyDiseaseModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Instantiate the model
model = PaddyDiseaseModel()

# Load the model weights
PATH = "models/transfer_cnn_model.pth"
model.load_state_dict(torch.load(PATH, map_location=device))
model.to(device)
model.eval()

# Define the image transformation function
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0).to(device)

# Define the prediction function
def get_prediction(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, prediction = torch.max(outputs, 1)
    return prediction

# Load Image -----------------------------------+
image_path = 'images/<file-name>.<extention>' # |
# ----------------------------------------------+

# Transform the image
with open(image_path, 'rb') as image_file:
    image_bytes = image_file.read()

im_tensor = transform_image(image_bytes)

# Get the prediction
pred = get_prediction(im_tensor)

# Print the prediction
print(pred)
