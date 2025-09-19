import os
import logging
# from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

from model.OtimazedModel import OtimizedModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_dir = os.getenv('LAMBDA_TASK_ROOT', '.')
model_path = os.path.join(base_dir, 'model', 'best.onnx')

logger.info(f"Model path: {model_path}")

# Define the global variable for your model
yolo = OtimizedModel(model_path)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def hello():
    return JSONResponse("Hello")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if yolo is None:
        return JSONResponse(
            content={"error": "Model not loaded."},
            status_code=500
        )

    try:
        logger.info(
            f"Request received on /analyze with file: {file.filename}")

        image_bytes = await file.read()
        results = yolo.analyze(image_bytes)

        return JSONResponse(
            content={"results": results},
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/health")
async def health_check():
    status = "healthy" if yolo else "initializing"
    return JSONResponse(
        content={"status": status,
                 "timestamp": datetime.datetime.now().isoformat()},
        headers={"Access-Control-Allow-Origin": "*"}
    )
