from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import uvicorn
from io import BytesIO
from PIL import Image
import os


app = FastAPI()

# MODEL =tf.keras.models.load_model("../training/models/1")
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL =tf.keras.models.load_model("training/models/1")

# MODEL = tf.keras.models.load_model("../training/models/1")


# prod_model = tf.keras.models.load_model("../training/models/1")
# beta_model = tf.keras.models.load_model("../training/models/2")


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

    prediction = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction[0])
    print(predicted_class, confidence)
    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

    pass

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8080)

