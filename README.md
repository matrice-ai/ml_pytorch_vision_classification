# Model Repository Documentation

This repository contains scripts for training, evaluating, exporting, and deploying machine learning models.

## train.py

### Usage:

1. Obtain running parameters using `job_params = actionTracker.get_job_params()`.
2. Fetch the prepared dataset path with `actionTracker.get_dataset_path()`.
3. If applicable, add index-to-category mapping for class prediction with `actionTracker.add_index_to_category({'1':"class_1",..})`.
4. Log epoch results using `actionTracker.log_epoch_results([{"splitType": "train", "metricName": "loss", "metricValue": loss_train}, {...}])`.
5. Update action status in the frontend with `actionTracker.update_status(stepCode, status, status_description)`.
6. Save the best model weights file to S3 using `actionTracker.upload_checkpoint('model_best.pt')`.

## eval.py

### Usage:

1. Obtain running parameters using `job_params = actionTracker.get_job_params()`.
2. Download the model with `model_path = actionTracker.download_model('model_best.pt')`.
3. Fetch the prepared dataset path with `actionTracker.get_dataset_path()`.
4. Update action status in the frontend with `actionTracker.update_status(stepCode, status, status_description)`.
5. Save evaluation results using `actionTracker.save_evaluation_results([{"category": "all", "splitType": "val", "metricName": "acc_1", "metricValue": float(acc1)}, {...}])`.

## export.py

### Usage:

1. Obtain running parameters using `job_params = actionTracker.get_job_params()`.
2. Download the model with `model_path = actionTracker.download_model('model_best.pt')`.
3. Update action status in the frontend with `actionTracker.update_status(stepCode, status, status_description)`.
4. Save the exported model file to S3 using `actionTracker.upload_checkpoint('model_best.onnx')`.

## predict.py

### Usage:

Include two functions:

1. `load_model(model_id)`: Download and load the model using the `model_id`.
2. `predict(model, image)`: Predict given the model and the image opened as a bytes object.

## deploy.py

### Usage:

1. Create a script that utilizes the `MatriceModelDeploy()` object.
2. Implement functions like `load_model(model_id)` and `predict(model, image)`.
3. Finally, invoke `model.run_server()` to deploy the model.
