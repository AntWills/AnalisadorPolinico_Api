import onnxruntime as ort
import numpy as np
from PIL import Image
from io import BytesIO


class OtimizedModel:
    def __init__(self):
        # Carrega o modelo ONNX
        self.ort_session = ort.InferenceSession(
            "best.onnx", providers=["CPUExecutionProvider"])

        # A ordem das classes agora está CORRETA
        self.class_names = [
            "anadenanthera", "arecaceae", "arrabidaea", "cecropia", "chromolaena",
            "combretum", "croton", "dipteryx", "eucalipto", "faramea",
            "hyptis", "mabea", "matayba", "mimosa", "myrcia",
            "protium", "qualea", "schinus", "senegalia", "serjania",
            "syagrus", "tridax", "urochloa"
        ]

    def analyze(self, image_bytes: bytes):
        # Converte bytes da imagem para um objeto Pillow
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # O modelo espera um tensor (1, 3, 224, 224)
        image_size = (224, 224)
        processed_image = image.resize(image_size)

        # Converte para numpy array
        input_data = np.array(processed_image, dtype=np.float32)

        # PASSO CORRIGIDO: Normaliza os valores dos pixels para o intervalo de 0 a 1
        input_data = input_data / 255.0

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
