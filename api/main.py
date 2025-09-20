import os
import logging
from fastapi.middleware.cors import CORSMiddleware
import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from model.OtimazedModel import OtimizedModel

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define os caminhos para o modelo e o JSON
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model", "best.onnx")
class_names_path = os.path.join(base_dir, "model", "class_names.json")

logger.info(f"Model path: {model_path}")
logger.info(f"Class names path: {class_names_path}")

# Carrega o modelo globalmente
try:
    yolo = OtimizedModel(model_path, class_names_path)
    logger.info("Modelo carregado com sucesso.")
except Exception as e:
    logger.error(f"Erro ao carregar o modelo: {str(e)}")
    yolo = None

app = FastAPI()

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def hello():
    return JSONResponse(content={"message": "Hello"}, headers={"Access-Control-Allow-Origin": "*"})


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if yolo is None:
        logger.error("Modelo não carregado.")
        return JSONResponse(
            content={"error": "Model not loaded."},
            status_code=500,
            headers={"Access-Control-Allow-Origin": "*"}
        )

    try:
        logger.info(f"Request received on /analyze with file: {file.filename}")
        image_bytes = await file.read()
        results = yolo.analyze(image_bytes)
        return JSONResponse(
            content={"results": results},
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500,
            headers={"Access-Control-Allow-Origin": "*"}
        )


@app.get("/health")
async def health_check():
    status = "healthy" if yolo is not None else "failed"
    return JSONResponse(
        content={"status": status,
                 "timestamp": datetime.datetime.now().isoformat()},
        headers={"Access-Control-Allow-Origin": "*"}
    )
