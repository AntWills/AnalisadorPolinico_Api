import onnxruntime as ort
import numpy as np
from PIL import Image
from io import BytesIO
import json
import os


class OtimizedModel:
    def __init__(self, path: str, class_names_path: str = "class_names.json"):
        # Carrega o modelo ONNX
        self.ort_session = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )

        # Carrega os nomes das classes do arquivo JSON
        if not os.path.exists(class_names_path):
            raise FileNotFoundError(
                f"Arquivo de classes não encontrado: {class_names_path}")
        with open(class_names_path, "r") as f:
            data = json.load(f)
            self.class_names = data["names"]

        # Verifica se o número de classes corresponde à saída do modelo
        output_shape = self.ort_session.get_outputs()[0].shape
        expected_classes = output_shape[-1] if len(
            output_shape) > 1 else output_shape[0]
        if len(self.class_names) != expected_classes:
            raise ValueError(
                f"O número de classes no JSON ({len(self.class_names)}) não corresponde "
                f"à saída do modelo ({expected_classes})"
            )

    def analyze(self, image_bytes: bytes):
        # Converte bytes da imagem para um objeto Pillow
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # O modelo espera um tensor (1, 3, 640, 640)
        image_size = (224, 224)  # Corrigido para 640x640
        processed_image = image.resize(image_size)

        # Converte para numpy array
        input_data = np.array(processed_image, dtype=np.float32)

        # Normaliza os valores dos pixels para o intervalo de 0 a 1
        input_data = input_data / 255.0

        # Transpõe para o formato (1, 3, 640, 640)
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)

        # Roda a inferência
        input_name = self.ort_session.get_inputs()[0].name
        output_name = self.ort_session.get_outputs()[0].name
        outputs = self.ort_session.run([output_name], {input_name: input_data})

        # Processa e formata o resultado
        probabilities = outputs[0].squeeze()

        response = []
        for class_id, prob in enumerate(probabilities):
            if prob > 0.1:
                response.append({
                    "class": self.class_names[class_id],
                    "probability": float(prob)
                })

        response.sort(key=lambda x: x["probability"], reverse=True)
        return response
