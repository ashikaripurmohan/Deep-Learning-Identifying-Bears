# Identifying Bears

A bear image classifier built on **ResNet-18**, trained using the [fastai](https://docs.fast.ai/) library. The model was trained on a bear dataset from [Kaggle](https://www.kaggle.com/) and can distinguish between **grizzly bears**, **black bears**, and **teddy bears**.

Upload any image and the model will classify it with a confidence percentage for each category.

## How it works

The trained model is exported as `bear_model.pkl`. When you upload an image, the server loads the model, runs the image through ResNet-18, and returns the prediction along with confidence scores — for example, 92.4% grizzly, 5.1% black, 2.5% teddy.

## Run locally

**Requirements:** Python 3.10+

1. Clone this repo:
   ```
   git clone https://github.com/ashikaripurmohan/identifying-bears.git
   cd identifying-bears
   ```

2. Install dependencies:
   ```
   pip install fastai fastapi uvicorn python-multipart
   ```

3. Start the server:
   ```
   python3 server.py
   ```

4. Open **http://localhost:4000** in your browser.

## Project structure

| File | Description |
|---|---|
| `server.py` | Python backend — loads the model and serves predictions via FastAPI |
| `index.html` | Frontend UI — drag-and-drop image upload with confidence bars |
| `bear_model.pkl` | Trained ResNet-18 model exported from fastai (45 MB) |

## Author

Avinash Mohan
