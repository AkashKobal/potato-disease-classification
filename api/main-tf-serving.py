from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import requests
import tensorflow as tf
import uvicorn
from io import BytesIO
from PIL import Image





app = FastAPI()
endpoint = "http://localhost:8080/v1/models/potatoes_model/predict"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"



def read_file_as_image(data) -> np.ndarray:
   image = np.array(Image.open(BytesIO(data)))
   return image
   pass

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": image.tolist()
    }
    response =  requests.post(endpoint,json=json_data)
    prediction = response.json()["prediction"][0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction[0])
    return{
        "class": predicted_class,
        "confidence": float(confidence)
    }

    pass

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8080)
