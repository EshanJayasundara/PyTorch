### 1. Watch the video tutorial

How To Deploy ML Models With Google Cloud Run: [Video Guide by Mr. Patrick Loeber](https://www.youtube.com/watch?v=vieoHqt7pxo&t=314s).

### 2. Write App (Flask and PyTorch, TensorFlow or whatever)

The code to build, train, and save the model is `TransferModelCNNTrain.py`.
I have run it and able to locate in `developments/transfer_model_cnn/transfer_cnn_model.pth`
and used to implement the app in `main.py`

Dataset can be found inside `data/paddy-doctor-diseases-small-400-split` directory

_Optional Information: `resnet18` is the transfer model utilized to make CNN in `TransferModelCNNTrain.py`_

### 3. Setup Google Cloud

Create new project
Activate Cloud Run API and Cloud Build API

### 4. Install and init Google Cloud SDK

_Reference:_
*https://cloud.google.com/sdk/docs/install*

### 5. Dockerfile, requirements.txt, .dockerignore

Creating `Dockerfile`, `requirements.txt` and `.dockerignore` are described in the [video guide.](https://www.youtube.com/watch?v=vieoHqt7pxo&t=314s)

_Reference:_
*https://cloud.google.com/run/docs/quickstarts/build-and-deploy#containerizing*

### 6. Cloud build & deploy

```
gcloud builds submit --tag gcr.io/<project_id>/<function_name>
```

```
gcloud run deploy getprediction --image gcr.io/<project_id>/<function_name> --platform managed --region asia-east1 --memory <memory-size>
```

### 7. Test

Test the code with `test/test.py`
