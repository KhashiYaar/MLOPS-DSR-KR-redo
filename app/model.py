import os
import wandb
from loadotenv import load_env # removed in GCP deployment
from pathlib import Path
import torch
from torchvision.models import resnet18, ResNet
from torch import nn
from torchvision.transforms import v2 as transforms



load_env()
wandb_api_key = os.environ.get('WANDB_API_KEY')

MODELS_DIR = 'models'
CATEGORIES = ["freshapple", "freshbanana", "freshorange", 
              "rottenapple", "rottenbanana", "rottenorange"]

os.makedirs(MODELS_DIR, exist_ok=True)

def download_artifact():
    assert 'WANDB_API_KEY' in os.environ, 'Please enter the wandb API key'

    wandb_org = os.environ.get('WANDB_ORG')
    wandb_project = os.environ.get('WANDB_PROJECT')
    wandb_model_name = os.environ.get('WANDB_MODEL_NAME')
    wandb_model_version = os.environ.get('WANDB_MODEL_VERSION')

    artifact_path = f"{wandb_org}/{wandb_project}/{wandb_model_name}:{wandb_model_version}"

    wandb.login()
    print(f"Downloading artifact {artifact_path} to {MODELS_DIR}")
    artifact = wandb.Api().artifact(artifact_path, type='model')
    artifact.download(root=MODELS_DIR)



# model should have the same architecture as the one that we have on Kaggle, 
# but without any weights
import torch
from torch import nn
from torchvision.models import resnet18, ResNet

def get_raw_model() -> ResNet: # 
    architecture = resnet18(weights=None)
    architecture.fc = nn.Sequential(
        nn.Linear(512, 512), #be exact as your kaggel model architecture
        nn.ReLU(),
        nn.Linear(512, 6)
    )

    return architecture 


# return model with the weights from the Wandb Artifact
def load_model() -> ResNet:
     download_artifact()
     model = get_raw_model()
     # Get the trained model weights
     model_state_dict_path = Path(MODELS_DIR) / 'best_model.oth'
     model_state_dict = torch.load(model_state_dict_path)
     model.load_state_dict(model_state_dict, strict=True)
     model.eval()
     return model
     # Assign the trained model weights to model, this will fail for incomplete files 
     # Check the file size on wandb.ai, the resnet18 artifact should have 45.8 MB in size
     # Turn off BatchNorm and Dropout

from torchvision.transforms import v2 as transforms

def load_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(224, 224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225])
    ])


download_artifact()