import sys
from matrice_sdk.matrice import Session
from matrice_sdk.models import Model

class InferenceOptimization:

    def __init__(self, session, inference_optim_id=None):
        self.project_id = session.project_id
        self.inference_id = inference_optim_id
        self.rpc = session.rpc

    def handle_response(self, resp, success_message, error_message):
        """Helper function to handle API response"""
        if resp.get("success"):
            error = None
            message = success_message
        else:
            error = resp.get("message")
            message = error_message

        return resp, error, message
    
    #GET REQUESTS
    def get_exports_summary(self):
        path = f"/v1/model/summaryExported?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Model Export Summary fetched successfully",
                                    "Could not fetch models export summary")

    def list_exported_models(self):
        path = f"/v1/model/model_exported?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Models exported list fetched successfully",
                                    "Could not fetch models exported list")
    
    def get_model_exports(self):
        path = f"/v1/model/get_model_exports?projectId={self.project_id}"
        print(path)
        resp = self.rpc.get(path=path)
        print(resp)
        return self.handle_response(resp, "Model exports fetched successfully",
                                    "Could not fetch model exports")
    
    def get_model_export_by_id(self, model_export_id):
        path = f"/v1/model/get_model_export_by_id?modelExportId={model_export_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Model export by ID fetched successfully",
                                    "Could not fetch model export by ID")
    
    def get_model_train_by_export_id(self, model_export_id):
        path = f"/v1/model/get_model_train_by_export_id?exportId={model_export_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Model train by export ID fetched successfully",
                                    "Could not fetch model train by export ID")
    
    #POST REQUESTS
    def add_model_eval(
        self,
        is_optimized,
        is_pruned,
        runtime_framework,
        id_dataset,
        id_experiment,
        dataset_version,
        is_gpu_required,
        split_types,
        model_type,
    ):
        path = "/v1/model/add_model_eval"
        headers = {'Content-Type': 'application/json'}
        model_payload = {
            "_idModel": self.inference_id,
            "_idProject": self.project_id,
            "isOptimized": is_optimized,
            "isPruned": is_pruned,
            "runtimeFramework": runtime_framework,
            "_idDataset": id_dataset,
            "_idExperiment": id_experiment,
            "datasetVersion": dataset_version,
            "gpuRequired": is_gpu_required,
            "splitTypes": split_types,
            "modelType": model_type,
            "computeAlias" : "" 
        }
        resp = self.rpc.post(path=path, headers=headers, payload=model_payload)

        return self.handle_response(resp, "Model eval added successfully",
                                    "An error occured while adding model eval")
    
    def add_model_export(
        self,
        model_train_id,
        export_formats,
        model_config,
        is_gpu_required=False
    ):
        S = Session(self.project_id)
        M = Model(S)
        model_train_id_resp, _, _ = M.get_model_train_by_id(model_train_id)
        if model_train_id_resp["createdAt"] == '0001-01-01T00:00:00Z':
            print("No model exist with the given model train id")
            sys.exit(0)

        path = f"/v1/model/{model_train_id}/add_model_export?projectId={self.project_id}"
        headers = {'Content-Type': 'application/json'}
        model_payload = {
            "_idProject": self.project_id,
            "_idModelTrain": model_train_id,
            "modelName": model_train_id_resp["modelName"],
            "modelInputs": model_train_id_resp["modelInputs"],
            "_idModelInfo": model_train_id_resp["_idModelInfo"],
            "modelOutputs": model_train_id_resp["modelOutputs"],
            "exportFormats": export_formats,
            "_idDataset": model_train_id_resp["_idDataset"],
            "datasetVersion": model_train_id_resp["datasetVersion"],
            "gpuRequired": is_gpu_required,
            "actionConfig": model_train_id_resp["actionConfig"],
            "modelConfig": model_config,
        }
        resp = self.rpc.post(path=path, headers=headers, payload=model_payload)

        return self.handle_response(resp, "Model export eval added successfully",
                                    "An error occured while adding model export eval")
    
    #PUT REQUEST
    def update_model_export_name(self, model_export_id, updated_name):

        body = {
            "modelExportId": self.inference_id,
            "name": updated_name,
            }
        
        headers = {"Content-Type": "application/json"}
        path = f"/v1/model/{model_export_id}/update_modelExport_name"
        resp = self.rpc.put(path=path, headers=headers, payload=body)

        return self.handle_response(resp, f"Model export name updated to {updated_name}",
                                    "Could not update the model export name")
    
    #DELETE REQUEST
    def delete_model_export(self, model_export_id):
        path = f"/v1/model/model_export/{model_export_id}"
        resp = self.rpc.delete(path=path)

        return self.handle_response(resp, f"Model export deleted",
                                    "Could not delete the model export")
    