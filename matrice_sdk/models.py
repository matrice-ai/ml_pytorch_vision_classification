import sys
import requests
from matrice_sdk.dataset import Dataset
from matrice_sdk.matrice import Session
from matrice_sdk.projects import Projects
from matrice_sdk.model_store import ModelStore

class Model:

    def __init__(self, session, model_id=None):
        self.project_id = session.project_id
        self.model_id = model_id
        self.rpc = session.rpc

    def handle_response(self, response, success_message, failure_message):
        """Handle API response and return a standardized tuple"""
        if response.get("success"):
            result = response.get("data")
            error = None
            message = success_message
        else:
            result = None
            error = response.get("message")
            message = failure_message

        return result, error, message
    

    #GET REQUESTS
    def get_models_summary(self):
        path = f"/v1/model/summary?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Model Summary fetched successfully",
                                    "Could not fetch models summary")

    def list_completed_model_train(self):
        path = f"/v1/model/model_train_completed?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Model train list fetched successfully",
                                    "Could not fetch models train list")

    def list_model_train_paginated(self):
        path = f"/v1/model/model_train?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Model train list fetched successfully",
                                    "Could not fetch models train list")

    def get_model_train_by_id(self, model_train_id):
        path = f"/v1/model/model_train/{model_train_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Model train by ID fetched successfully",
                                    "Could not fetch model train by ID")

    #POST REQUESTS
    def get_eval_result(
            self,
            dataset_id,
            dataset_version,
            split_type
    ):
        S = Session(self.project_id)
        D = Dataset(S, dataset_id)
        dataset_info,_,_=D.get_processed_versions()
        if dataset_info is None:
            print("No datasets found")
            sys.exit(0)

        flag = False
        for data_info in dataset_info:
            if dataset_id==data_info["_id"]:
                if dataset_version in data_info["processedVersions"]:
                    flag = True
                    break
        
        if flag == False:
            print("Dataset or Dataset version does not exist. Can not use this dataset version to get/create a eval.")
            sys.exit(0)

        if self.model_id is None:
            print("Model Id is required for this operation")
            sys.exit(0)

        path = "/v1/model/get_eval_result"
        headers = {'Content-Type': 'application/json'}
        model_payload = {
            "_idDataset": dataset_id,
            "_idModel": self.model_id,
            "datasetVersion": dataset_version,
            "splitType": split_type
        }   
        resp = self.rpc.post(path=path, headers=headers, payload=model_payload)

        return self.handle_response(resp, "Eval result fetched successfully",
                                    "An error occured while fetching Eval result")

    def add_model_train_list(
            self,
            request_header,
    ):
        path = f"/v1/model/add_model_train_list?projectId={self.project_id}"
        headers = {'Content-Type': 'application/json'}
        model_payload=[]
        for req in request_header:
            payload = {
                        "modelKey": req['model_key'], #"resnet152",
                        "autoML": req['is_autoML'], #false,
                        "tuningType": req['tuning_type'], #"manual",
                        "modelCheckpoint": req['model_checkpoint'], #"auto"
                        "checkpointType": req['checkpoint_type'], #"predefined"
                        "primaryMetric": req['primary_metric'], #"acc@1",
                        "datasetName": req['dataset_name'], #"test",
                        "paramsMillions": req['params_millions'], #0,
                        "experimentName": req['experiment_name'], #"testingKhushi",
                        "modelName": req['model_name'], #"ResNet-152",
                        "modelInputs": req['model_inputs'], #["image"],
                        "modelOutputs": req['model_outputs'], #["classification"],
                        "targetRuntime": req['target_runtime'], #["PyTorch"],
                        "_idDataset": req['id_dataset'], #"65cb14fbd8db675a46b0e375",
                        "datasetVersion": req['dataset_version'], #"v1.0",
                        "_idModelInfo": req['id_model_info'], #"65b8d07e47ac273ab8fe515d",
                        "_idExperiment": req['id_experiment'], #"65d6fb8c0643dfdd17b6b0a7",
                        "actionConfig": req['action_config'], #{},
                        "modelConfig": req['model_config'], #{"min_delta":[0.0001],"lr_gamma":[0.1],"lr_min":[0.00001],"learning_rate":[0.001],"patience":[5],"lr_step_size":[10],"lr_scheduler":["StepLR"],"optimizer":["AdamW"],"weight_decay":[0.0001],"momentum":[0.95],"epochs":[50],"batch_size":[4]}
                    }
            model_payload.append(payload)
            
        resp = self.rpc.post(path=path, headers=headers, payload=model_payload)

        return self.handle_response(resp, "Model training list added successfully",
                                    "An error occured while adding model training list")

    def create_experiment(
            self,
            name,
            dataset_id,
            target_run_time,
            dataset_version,
            primary_metric,
            matrice_compute,
            models_trained=[],
            performance_trade_off=-1,
            project_id=None
        ):
            if project_id is None and self.project_id is None:
                print("Project ID is required for this operation.")
                sys.exit(0)
            
            if project_id is None:
                project_id=self.project_id

            session=Session(project_id)
            dataset=Dataset(session, dataset_id)
            project=Projects(session)
            model_store=ModelStore(session)

            project_info,_,_=project.get_a_project_by_id(project_id)
            if project_info is None:
                print("No project found.")
                sys.exit(0)

            dataset_info,_,_=dataset.get_processed_versions()
            if dataset_info is None:
                print("No datasets found")
                sys.exit(0)
            
            model_information = ''
            for data_info in dataset_info:
                if dataset_id==data_info["_id"]:
                    if dataset_version in data_info["processedVersions"]:
                        model_information=data_info
                        break
            
            if model_information == '':
                print("Dataset or Dataset version does not exist. Can not use this dataset version to create a model.")
                sys.exit(0)
            
            if model_information is None:
                print("Dataset not found.")
                sys.exit(0)
            
            model_inputs = [project_info["data"]["inputType"]]
            model_outputs = [project_info["data"]["outputType"]]

            runtime_metrics,_,_=model_store.fetch_supported_runtimes_metrics(
                model_inputs, model_outputs, project_id)
            
            if runtime_metrics is None:
                print("No primary metric and target runtime found.")
                sys.exit(0)
            
            if target_run_time not in runtime_metrics["data"]["supportedRuntimes"]:
                print("Target runtime provided does not exist.")
                sys.exit(0)
            
            if primary_metric not in runtime_metrics["data"]["supportedMetrics"]:
                print("Primary metric not available in the existing runtime Metrics.")
                #print(f"Following metrics available: {runtime_metrics["data"]["supportedMetrics"]}")
                sys.exit(0)

            path = f"/v1/model/create_experiment?projectId={project_id}"
            headers = {'Content-Type': 'application/json'}

            if matrice_compute == False:
                model_payload = {
                    "experimentName": name,
                    "_idProject": project_id,
                    "matriceCompute": matrice_compute
                }
            else:
                model_payload = {
                    "experimentName": name,
                    "_idDataset": dataset_id,
                    "modelInputs": [project_info["data"]["inputType"]],
                    "modelOutputs": [project_info["data"]["outputType"]],
                    "targetRuntime": [target_run_time],
                    "datasetVersion": dataset_version,
                    "performanceTradeoff": performance_trade_off,
                    "primaryMetric":primary_metric,
                    "modelsTrained": models_trained,
                    "matriceCompute": matrice_compute,
                    "baseModelStoragePath":"",
                    "storageCloudCredentials":[]
                }
            
            resp = self.rpc.post(path=path, headers=headers, payload=model_payload)  

            return self.handle_response(resp, "Experiment successfully created",
                                    "An error occured while creating an experiment")

    def add_model_eval(
        self,
        id_dataset,
        dataset_version,
        split_types,
        export_format=None,
        is_optimized=False,
        is_pruned=False,
        is_gpu_required=False,
    ):
        if self.model_id is None:
            print("Set Model Id for model object")
            sys.exit(0)

        model_by_id_resp, _, _ = self.get_model_train_by_id(self.model_id)
        path = "/v1/model/add_model_eval"
        headers = {'Content-Type': 'application/json'}
        model_payload = {
            "_idModel": self.model_id,
            "_idProject": self.project_id,
            "isOptimized": is_optimized,
            "isPruned": is_pruned,
            "runtimeFramework": model_by_id_resp["targetRuntime"][0],
            "_idDataset": id_dataset,
            "_idExperiment": model_by_id_resp["_idExperiment"],
            "datasetVersion": dataset_version,
            "gpuRequired": is_gpu_required,
            "splitTypes": split_types,
            "modelType": model_by_id_resp["status"],
            "exportFormat" : export_format
        }
        resp = self.rpc.post(path=path, headers=headers, payload=model_payload)

        return self.handle_response(resp, "Model eval added successfully",
                                    "An error occured while adding model eval")

    def get_model_download_path(
            self,
            model_type
    ):
        if self.model_id is None:
            print("Model id not set for this model. Cannot perform the operation for model without model id")
            sys.exit(0)

        path = "/v1/model/get_model_download_path"
        headers = {'Content-Type': 'application/json'}
        model_payload = {
            "modelID": self.model_id,
            "modelType": model_type,
            "expiryTimeInMinutes": 15
        }  
        resp = self.rpc.post(path=path, headers=headers, payload=model_payload)

        return self.handle_response(resp, "Model download path fetched successfully and it will expire in 15 mins",
                                    "An error occured while downloading the model")

    #PUT REQUESTS
    def update_model_train_name(self, updated_name):
        if self.model_id is None:
            print("Model id not set for this model. Cannot perform the operation for model without model id")
            sys.exit(0)

        body = {
            "modelTrainId": self.model_id,
            "name": updated_name,
            }
        
        headers = {"Content-Type": "application/json"}
        path = f"/v1/model/{self.model_id}/update_modelTrain_name"
        resp = self.rpc.put(path=path, headers=headers, payload=body)

        return self.handle_response(resp, f"Model name updated to {updated_name}",
                                    "Could not update the model name")

    #DELETE REQUESTS
    def delete_model_train(self, model_train_id):
        
        path = f"/v1/model/delete_model_train/{model_train_id}"
        resp = self.rpc.delete(path=path)

        return self.handle_response(resp, f"Model deleted",
                                    "Could not delete the model")


    # Need to handle if want to download exported model for eval or deploy
    def download_model(self,model_save_path, presigned_url,model_type="trained"):

        response = requests.get(presigned_url)

        if response.status_code == 200:
            with open(model_save_path, 'wb') as file:
                file.write(response.content)
            print("Download Successful")
            return True
        else:
            print(f"Download failed with status code: {response.status_code}")
            return False