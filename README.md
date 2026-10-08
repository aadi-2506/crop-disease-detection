# Crop Disease Detection Using Machine Learning

CropCare is a Python and Streamlit project that predicts a crop leaf's disease
from a photo. It uses transfer learning with an ImageNet-pretrained MobileNetV2
model. Users can upload a JPG/JPEG/PNG image or capture one with a device camera.

> **Educational use:** PlantVillage contains controlled-background images. A
> prediction is not a confirmed diagnosis or a substitute for local agricultural
> advice. Field photos may produce less reliable results.

## Features

- Automatically discovers disease classes from image-containing dataset folders.
- Resizes input images to 224 × 224 and includes MobileNetV2 input scaling.
- Augments training images, then trains a frozen MobileNetV2 feature extractor.
- Makes a reproducible, per-class train/validation/test split.
- Saves a `.keras` model, class labels, held-out test manifest and evaluation results.
- Reports accuracy, weighted precision, weighted recall, weighted F1 and confusion matrix.
- Provides upload and camera prediction, confidence and simple disease guidance.
- Caches the model in Streamlit rather than training at application startup.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit dashboard, image upload/camera flow, prediction and metrics UI. |
| `train.py` | Dataset discovery, train/validation/test splitting, MobileNetV2 training and model saving. |
| `evaluate.py` | Re-evaluates the saved model against the exact test images recorded at training time. |
| `predict.py` | Optional command-line prediction for a single image. |
| `utils/preprocessing.py` | Image size, class-folder discovery and MobileNetV2 model definition. |
| `utils/disease_info.py` | Crop/disease label parsing and plain-language care guidance. |
| `requirements.txt` | Python package dependencies. |
| `dataset/` | Extracted PlantVillage class folders (download separately). |
| `models/` | Created by training; stores the trained model and evaluation metadata. |

## Requirements

- Windows, macOS or Linux.
- Python 3.10 (recommended with the pinned TensorFlow range in `requirements.txt`).
- Internet access for installing packages and the first download of ImageNet weights.
- Enough disk space and memory for PlantVillage and model training. A GPU is optional.

## Installation (Windows PowerShell)

From the project root in VS Code:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents local activation, use the environment's interpreter
directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Select `.venv` as the Python interpreter in VS Code (`Ctrl+Shift+P` →
**Python: Select Interpreter**).

## Dataset setup

1. Download the PlantVillage dataset from Kaggle (search for **PlantVillage
   Dataset**) and accept the dataset's current terms of use.
2. Extract the image class directories beneath this project's `dataset` folder.
   The accepted layouts include either:

   ```text
   dataset/
   ├── Apple___Apple_scab/
   ├── Apple___Black_rot/
   └── Tomato___healthy/
   ```

   or an extra wrapper directory:

   ```text
   dataset/
   └── PlantVillage/
       ├── Apple___Apple_scab/
       └── Tomato___healthy/
   ```

3. Each class folder should contain `.jpg`, `.jpeg` or `.png` leaf images.
   Folder names are the class labels; no class count is hardcoded. At least
   three images per class are required so each class can be represented in
   training, validation and test splits.

## Train

Run this from the project root after installing requirements and setting up
the dataset:

```powershell
python train.py
```

Defaults: 15 epochs, batch size 32, seed 42, dataset `./dataset`, model output
`./models`. Customize them if required:

```powershell
python train.py --epochs 20 --batch-size 16
python train.py --dataset-dir "D:\datasets\PlantVillage" --model-dir .\models
```

The first run downloads pretrained ImageNet weights from Keras. Training writes:

- `models/crop_disease_model.keras` — best model selected by validation accuracy.
- `models/test_manifest.json` — relative paths and labels for the held-out test set.
- `models/evaluation_results.json` — classes and held-out metrics, used by the app.

Do not rename or move test images after training if you plan to use `evaluate.py`.

## Evaluate

```powershell
python evaluate.py
```

Evaluation uses the saved test manifest, prints weighted precision/recall/F1,
accuracy and a per-class report, then updates `models/evaluation_results.json`.
The dashboard reads this file and renders the confusion matrix. Evaluation does
not retrain the model.

## Run the application

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal (normally
`http://localhost:8501`). Select **Disease Detection**, choose **Upload image**
or **Use camera**, then select **Detect disease**. A trained model must exist
under `models/` first.

Optional command-line prediction:

```powershell
python predict.py "path\to\leaf.jpg"
```

## Deploying the Streamlit application

1. Train the model and include both `models/crop_disease_model.keras` and
   `models/evaluation_results.json` with the deployment. These generated model
   artifacts are not included in source control by default in most workflows.
2. Push the project and trained model artifacts to a private GitHub repository
   or another deployment source. Check the dataset's license/terms before
   distributing any dataset images.
3. Create an app on [Streamlit Community Cloud](https://share.streamlit.io/)
   (or use another service that supports Streamlit), select the repository,
   branch and `app.py`, and set Python to 3.10 if the service permits.
4. Configure the dependency file as `requirements.txt` and deploy. For Community
   Cloud, keep the model artifacts in the repo only if they fit the service's
   limits; otherwise store them in an authorized model-artifact store and adapt
   deployment to retrieve them securely.
5. Test both upload and camera flows on the hosted URL. Camera availability
   depends on the browser, device permissions and hosting environment.

For larger deployments, use a host with sufficient memory/disk and external
model storage. Keep credentials out of source code and use the platform's
secrets mechanism for any private artifact access.
