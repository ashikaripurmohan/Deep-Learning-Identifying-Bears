"""
Bear Classifier — Local Server
Loads bear_model.pkl (trained on Kaggle with fastai)
and serves predictions via a simple API on port 4000.
"""

from fastai.vision.all import *
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
from io import BytesIO
import torch
import pickle

# ── Fix version mismatch between Kaggle and local ──────
# The model was exported on Kaggle where plum-dispatch had
# internal modules named plum._resolver, plum._function etc.
# The current version renamed them. We create a custom
# unpickler that redirects old names to current classes.

from plum.resolver import Resolver as _OrigResolver

class Resolver(_OrigResolver):
    """Resolver subclass that stores attrs in __dict__
    (old plum used __dict__, new plum uses __slots__)."""
    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(name)

class _Method:
    """Stub for old plum Method objects."""
    def __init__(self, *a, **kw):
        for k, v in kw.items():
            object.__setattr__(self, k, v)

class _Missing:
    """Stub for old plum Missing sentinel."""
    pass

class CompatUnpickler(pickle.Unpickler):
    """Redirects old plum module paths to current ones."""
    _REMAP = {
        "plum._resolver": "plum.resolver",
        "plum._function": "plum.function",
        "plum._signature": "plum.signature",
        "plum._util": "plum.util",
        "plum._method": "plum.function",
    }

    def find_class(self, module, name):
        if name == "Resolver" and "plum" in module:
            return Resolver
        if (module, name) == ("plum._method", "MethodList"):
            return list
        if name == "Method" and module == "plum._method":
            return _Method
        if name == "Missing" and "plum" in module:
            return _Missing
        if module in self._REMAP:
            try:
                return super().find_class(self._REMAP[module], name)
            except (AttributeError, ImportError):
                return _Missing
        return super().find_class(module, name)

# ── Load the trained model ──────────────────────────────
_pickle_compat = type('P', (), {
    'Unpickler': CompatUnpickler,
    'load': staticmethod(lambda f, **kw: CompatUnpickler(f).load()),
    '__name__': 'pickle',
})()

model_path = Path(__file__).parent / "bear_model.pkl"
with open(model_path, "rb") as f:
    learn = torch.load(
        f, map_location="cpu", weights_only=False,
        pickle_module=_pickle_compat,
    )

print(f"  Model loaded: categories = {learn.dls.vocab}")

# ── Create the web server ───────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = PILImage.create(BytesIO(image_bytes))

    pred, pred_idx, probs = learn.predict(img)

    categories = learn.dls.vocab
    results = {}
    for i, category in enumerate(categories):
        results[category] = float(probs[i])

    return JSONResponse({
        "prediction": str(pred),
        "confidence": float(probs[pred_idx]),
        "probabilities": results,
    })

if __name__ == "__main__":
    print("\n  Bear Classifier running at http://localhost:4000\n")
    uvicorn.run(app, host="0.0.0.0", port=4000)
