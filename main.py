from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os

from model.ModelYOLO import ModelYOLO


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global yolo
#     print("Inicializando modelo YOLO...")
#     yolo = ModelYOLO()  # inicializa apenas uma vez
#     yield
#     print("Finalizando API...")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Diretório para salvar uploads
SAVE_DIR = "uploads"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Inicializando modelo YOLO...")
yolo = ModelYOLO()


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    print("[call POST /analyze]")

    image_bytes = await file.read()

    # Passa pro YOLO sem salvar
    results = yolo.analyze(image_bytes)
    print(f"results: {results}\n")

    return JSONResponse(
        content={"results": results},
        headers={"Access-Control-Allow-Origin": "https://antwills.github.io"}
    )


@app.post("/author")
async def author():
    existYolo = False

    if yolo:
        existYolo = True

    return JSONResponse({
        "name": "Wills",
        "yolo_exists": existYolo
    })


@app.post("/test")
async def analyze_image_test(file: UploadFile = File(...)):
    # Lê a imagem como bytes
    image_bytes = await file.read()

    # save_path = os.path.join(SAVE_DIR, file.filename)
    # with open(save_path, "wb") as f:
    #     f.write(image_bytes)

    # Converte para PIL Image se necessário
    image = Image.open(io.BytesIO(image_bytes))

    # Aqui você roda seu modelo de IA
    # Exemplo fictício: resultado = model.predict(image)
    resultado = {"classe": "abelha", "conf": 0.95}

    # Retorna JSON com informações da análise, não os bytes
    return JSONResponse(content=resultado)


@app.get("/health")
async def health_check():
    return JSONResponse(
        content={"status": "healthy",
                 "timestamp": datetime.datetime.now().isoformat()},
        # garante CORS
        headers={"Access-Control-Allow-Origin": "https://antwills.github.io"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
print("Finalizando API...")
