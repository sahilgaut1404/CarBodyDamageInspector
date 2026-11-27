# Car Body Damage Inspector

## First exercise (Windows, Python 3.11 or 3.12)

Extract the project zip. Open the extracted CarBodyDamageInspector folder in VS Code, then open Terminal → New Terminal.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install Pillow
.\.venv\Scripts\python.exe first_mask.py
```

Open practice_door.png and practice_outline.png side by side. The outline comes from four known points on a drawing. The model's job later is to estimate these regions from a new photo.

## Train in Google Colab

Request the COCO version of CarDD through https://cardd-ustc.github.io/ and follow the authors' use terms. Extract it to Google Drive. The notebook looks for the folder containing annotations/instances_train2017.json and train2017/, val2017/, test2017/. Put the ORIGINAL project zip in My Drive as CarBodyDamageInspector.zip.

Open Train_CarDD.ipynb in Colab, select Runtime → Change runtime type → GPU, and run the cells in order. Inspect the real ground-truth overlay first. The quick run checks the pipeline with 16 training images. The separate full run trains a fresh model using the full training split. Both use COCO-pretrained starting weights, downloaded on first use. Colab GPU availability and memory vary. The full run writes its latest finished epoch to My Drive. Restarting the training command starts a fresh run.

Validation outputs mask AP and per-class precision, recall and F1. AP uses COCO mask overlap thresholds from 0.50 to 0.95. Counts use same-class one-to-one mask matches at IoU 0.50 and the supplied confidence threshold. An undefined ratio is saved as null. Choose the threshold on validation and use the untouched test split once for final reporting. The metric files contain results for your actual checkpoint.

## Run the local inspection app

Download damage.pt from the CarInspector folder in My Drive and place it next to app.py. From the same Windows terminal run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Upload a JPEG or PNG. Compare the outlines with the original, confirm or reject each finding, enter its body panel and download the JSON review. Panel names are entered by the reviewer. A photo with no predicted findings may still contain visible damage. The scope is dents, scratches and cracks.

Model implementation reference: https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
