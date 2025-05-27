import io
import requests
from PIL import Image
from sklearn.datasets import fetch_openml

# Fetch a small set of MNIST samples
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
for idx in [0, 1, 2]:  # try first three samples
    img_data = X[idx].reshape(28, 28).astype('uint8')
    pil = Image.fromarray(img_data, mode='L')
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    buf.seek(0)

    # POST to your API
    resp = requests.post(
        'http://127.0.0.1:8000/predict-image/',
        files={'file': ('mnist.png', buf, 'image/png')}
    )
    print(f"True: {y[idx]}, Predicted: {resp.json()['prediction']}")