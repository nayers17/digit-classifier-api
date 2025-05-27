# Digit Classifier API

A simple FastAPI service that uses a pre-trained MNIST model to classify handwritten digits.

## Project Structure

```
mlapiproject/           # ← root of this repo
├── static/
│   └── index.html      # frontend: upload image & display prediction
├── mnist_model.pkl     # pre-trained sklearn model (pickle)
├── main.py             # FastAPI app (serves static + predict endpoint)
├── train_model.py      # script to train & pickle the MNIST model
├── venv/               # Python virtual environment
└── README.md           # this file
```

## Prerequisites

* Python 3.8+ installed
* Git (to clone the repo)

## Setup (in `root` directory)

1. **Clone** repository:

   ```bash
   git clone https://github.com/nayers17/digit-classifier-api.git
   cd digit-classifier-api
   ```
2. **Create & activate** a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install** dependencies:

   ```bash
   pip install fastapi uvicorn scikit-learn pillow numpy
   ```

## Running the App

1. **Ensure** `static/index.html` exists (already included).
2. **Start** the server from the repo root:

   ```bash
   uvicorn main:app --reload
   ```
3. **Open** your browser at `http://127.0.0.1:8000/` to see the upload form.
4. **Choose** a digit image and click **Classify**. The prediction will appear below the button.

## (Re)Training the Model

If you want to retrain on MNIST yourself:

1. In the root directory, run:

   ```bash
   python train_model.py
   ```
2. This will overwrite `mnist_model.pkl` with a newly trained model.

## Notes

* The FastAPI endpoint is mounted at `/predict-image/` and expects a `POST` with form-data `file`.
* Static assets (including `index.html`) are served from the `static/` folder.
