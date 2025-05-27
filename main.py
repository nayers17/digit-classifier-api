from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import io, pickle, numpy as np, PIL.ImageOps, PIL.Image

with open('mnist_model.pkl','rb') as f:
    model = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. serve anything in ./static (but not at "/")
app.mount(
  "/static",
  StaticFiles(directory="static"),
  name="static"
)

# 2. GET / → returns index.html
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# 3. POST /predict-image/  (note trailing slash)
@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    pil = PIL.Image.open(io.BytesIO(contents)).convert('L')
    pil = PIL.ImageOps.invert(pil)
    pil = pil.resize((28, 28), PIL.Image.Resampling.LANCZOS)
    arr = np.array(pil).reshape(1, -1)
    pred = model.predict(arr)[0]
    return {"prediction": int(pred)}
