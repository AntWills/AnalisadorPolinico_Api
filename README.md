# Analisador Polínico - Backend

![Licence](https://img.shields.io/badge/license-%20%20GNU%20GPLv3%20-green?style=plastic)

## Descrição

O Analisador Polínico é uma ferramenta desenvolvida no âmbito do Programa Institucional de Bolsas de Iniciação em Desenvolvimento Tecnológico e Inovação (PIBITI) do Instituto Federal de Educação, Ciência e Tecnologia do Maranhão (IFMA) - Campus Caxias, em parceria com FAPEMA e CNPq (Edital PRPGI Nº 14/2024, vigência 2024/2025).

O projeto consiste em um sistema de inteligência artificial que visa auxiliar apicultores e meliponicultores na identificação rápida e eficiente do espectro polínico presente no mel. A análise é realizada por meio de uma rede neural convolucional (CNN) otimizada, treinada para classificar imagens de amostras de mel e identificar os tipos de pólen presentes.

Este repositório contém o backend da aplicação, responsável por servir o modelo de IA através de uma API REST.

## Funcionalidades

- **Análise de Imagens:** Envio de uma imagem de amostra de mel para análise.
- **Identificação de Pólen:** Retorna uma lista das classes de pólen identificadas na imagem, com suas respectivas probabilidades.
- **API RESTful:** Interface de fácil integração para consumo por aplicações frontend ou outros serviços.
- **Health Check:** Endpoint para verificar o status e a disponibilidade do serviço.

## Tecnologias Utilizadas

- **Python 3:** Linguagem de programação principal.
- **FastAPI:** Framework web para construção da API.
- **ONNX Runtime:** Para execução do modelo de machine learning otimizado.
- **Pillow & NumPy:** Para manipulação e processamento de imagens.
- **Uvicorn:** Servidor ASGI para rodar a aplicação FastAPI.
- **Vercel:** Plataforma de deploy da aplicação.

## Classes de Pólen Identificadas

O modelo foi treinado para identificar as seguintes 23 classes de pólen:

- anadenanthera
- arecaceae
- arrabidaea
- cecropia
- chromolaena
- combretum
- croton
- dipteryx
- eucalipto
- faramea
- hyptis
- mabea
- matayba
- mimosa
- myrcia
- protium
- qualea
- schinus
- senegalia
- serjania
- syagrus
- tridax
- urochloa

## Endpoints da API

A API possui os seguintes endpoints:

### 1. Analisar Imagem

- **URL:** `/analyze`
- **Método:** `POST`
- **Descrição:** Recebe uma imagem e retorna a análise polínica.
- **Corpo da Requisição:** `multipart/form-data` com um campo `file` contendo a imagem.
- **Exemplo de Resposta (Sucesso):**
  ```json
  {
    "results": [
      {
        "class": "mimosa",
        "probability": 0.98
      },
      {
        "class": "syagrus",
        "probability": 0.15
      }
    ]
  }
  ```
- **Resposta (Erro):**
  ```json
  {
    "error": "Mensagem de erro detalhada."
  }
  ```

### 2. Health Check

- **URL:** `/health`
- **Método:** `GET`
- **Descrição:** Verifica o status do serviço.
- **Exemplo de Resposta:**
  ```json
  {
    "status": "healthy",
    "timestamp": "2024-09-19T12:00:00.000000"
  }
  ```

## Como Executar Localmente

Siga os passos abaixo para executar o projeto em seu ambiente local.

### 1. Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes do Python)

### 2. Clone o Repositório

```bash
git clone https://github.com/example/analisador-polinico-backend.git
cd analisador-polinico-backend
```

### 3. Crie e Ative um Ambiente Virtual

É uma boa prática usar um ambiente virtual para isolar as dependências do projeto.

```bash
# Criar o ambiente virtual
python -m venv venv

# Ativar no Windows
venv\\Scripts\\activate

# Ativar no macOS/Linux
source venv/bin/activate
```

### 4. Instale as Dependências

```bash
pip install -r requirements.txt
```

### 5. Execute a Aplicação

```bash
uvicorn api.main:app --reload
```

A aplicação estará disponível em `http://127.0.0.1:8000`. Você pode acessar a documentação interativa da API (gerada pelo Swagger UI) em `http://127.0.0.1:8000/docs`.

## Licença

Este projeto está licenciado sob a licença GNU GPLv3. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
