import torch
import io

from pydantic import BaseModel # its not our ML model, 
# BaseModel is just our parent class for all pydantic models


# pydantic is about data model (not ML) with strict data types
class result(BaseModel):
    lable: str #category
    probability: float #confidence

#create the FastAPI instance
from fastapi import FastAPI, Depends, UploadFile, File
app = FastAPI()


from torchvision.models import ResNet
from app.model import load_model, load_transforms, CATEGORIES

#create a get handle
@app.get('/')
def read_root():
    return{'message': 'Call predict instead of root, this is an ML endpoint'}

@app.post('/predcit', response_model=Result)
async def predict(
        input_image: UploadFile = File(...)),  # need to import file and upload file from fastapi
        model: ResNet = Depends(load_model), # fastapi, need ot have same Architecture as what we use, WandB gives us weights and biases not the Archietcture... how to get it? import resnet from torchvision.modes ... what depends dp? it gets it from load-model
        transforms: transforms.Compose = Depends(load_transforms)
) -> Result:
    image = Image.open(io.BytesIO(await input_image.read()))


