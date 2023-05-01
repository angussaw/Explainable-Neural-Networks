# Explainable-Neural-Networks

Following is the initial file structure. 
# Folders/File structure 
```bash
├── data
│   ├── test
│   │   ├── BACTERIA
│   │   └── NORMAL
│   │   └── VIRUS
│   └── train
│       ├── BACTERIA
│       ├── NORMAL
│       └── VIRUS
```

```bash
├── conf
│   ├── logging.yaml
│   └── train_config.yaml
```

```bash
├── src
│   ├── models
│   ├── xray_classifier
│   │   ├── __init__.py
│   │   ├── utils.py
│   │   └── modeling
│   │       ├── __init__.py
│   │       ├── evaluation.py
│   │       ├── model.py
│   │       └── training.py
│   ├── streamlit.py
│   └── training_pipeline.py
```


1. Environment

Please ensure dependencies adhere to python 3.10
```bash
conda env create -f conda-env.yaml
conda activate xnn
```

Starting MLflow on localhost
```cmd
mlflow server
```

2. Training

```cmd
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
set MODE=training
python src/training_pipeline.py
```

3. Fine-tuning

```cmd
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
set MODE=fine-tuning
python src/training_pipeline.py
```

4. Streamlit deployment

```cmd
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
set RUN_ID=434536936096303142
set MODEL_URI=96eb2ef9dd4d4c2db56e348cfd6e9cef
set MODEL_NAME=finalized_model
streamlit run src/streamlit.py
```