import sys

class ModelStore:

    def __init__(self,session):
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

    # To fetch a model family
    def get_model_family(self, model_family_id):
        
        path = f"/v1/model_store/model_family/{model_family_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Successfully fetched the model family",
                                    "An error occured while fetching the model family")

    # To fetch model info
    def get_model_info(self, model_info_id):

        path = f"/v1/model_store/model_info/{model_info_id}"
        resp = self.rpc.get(path=path)
       
        return self.handle_response(resp, "Successfully fetched the model info",
                                    "An error occured while fetching the model info")
    
    # To fetch model action config
    def get_model_action_config(self, model_action_config_id):

        path = f"/v1/model_store/model_action_config/{model_action_config_id}"
        resp = self.rpc.get(path=path)
        
        return self.handle_response(resp, "Successfully fetched the model action config",
                                    "An error occured while fetching the model action config")

    def get_all_models(self, project_id):
        path = f"/v1/model_store/get_all_models?projectId={project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Successfully fetched all model infos",
                                    "An error occured while fetching the model family")
    
    def get_all_model_families(self, project_id):
        path = f"/v1/model_store/get_all_model_families?projectId={project_id}"
        resp = self.rpc.get(path=path)
    
        return self.handle_response(resp, "Successfully fetched all model family",
                                    "An error occured while fetching the model family")
    
    def get_models_by_modelfamily(self, model_family_id):
        path = f"/v1/model_store/get_models_by_modelfamily?modelFamilyId={model_family_id}"
        resp = self.rpc.get(path=path)
       
        return self.handle_response(resp, "Successfully fetched all model family",
                                    "An error occured while fetching the model family")
    
    def get_export_formats(self, model_info_id):
        path = f"/v1/model_store/get_export_formats?modelInfoId={model_info_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Successfully fetched all model family",
                                    "An error occured while fetching the model family")
   
    def get_action_config_for_model_export(self, model_info_id, export_format):
        path = f"/v1/model_store/get_action_config_for_model_export?modelInfoId={model_info_id}&exportFormat={export_format}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Successfully fetched all model family",
                                    "An error occured while fetching the model family")
            
    def fetch_supported_runtimes_metrics(self, model_inputs, model_outputs, project_id):
        path = f"/v1/model_store/fetch_supported_runtimes_metrics?projectId={project_id}"
        payload = {
            "modelInputs": model_inputs,
            "modelOutputs": model_outputs,
        }
        headers = {'Content-Type': 'application/json'}
        resp = self.rpc.post(path=path, headers=headers, payload=payload)

        return self.handle_response(resp, "Successfully fetched all model family",
                                    "An error occured while fetching the model family")
    
    # To delete a model family
    def delete_model_family(self, model_family_id):
        
        path = f"/v1/model_store/model_family/{model_family_id}"
        resp = self.rpc.delete(path=path)
    
        return self.handle_response(resp, "Successfully deleted the model family",
                                    "An error occured while deleting the model family")
    
    # To delete model info
    def delete_model_info(self, model_info_id):

        path = f"/v1/model_store/model_info/{model_info_id}"
        resp = self.rpc.delete(path=path)
        
        return self.handle_response(resp, "Successfully deleted the model family",
                                    "An error occured while deleting the model family")
    
    # To delete model action config
    def delete_model_action_config(self, model_action_config_id):

        path = f"/v1/model_store/model_action_config/{model_action_config_id}"
        resp = self.rpc.delete(path=path)
        
        return self.handle_response(resp, "Successfully deleted the model action config",
                                    "An error occured while deleting the model action config")
    
    # To add a new entry into model family
    def add_model_family(
        self, 
        project_id, 
        model_family, 
        model_inputs, 
        model_outputs, 
        models, 
        description, 
        training_framework, 
        supported_runtimes, 
        benchmark_datasets, 
        supported_metrics, 
        pruning_support, 
        code_repository, 
        training_docker_container, 
        input_format, 
        data_loader_class_definition, 
        data_loader_call_signature, 
        references, 
        is_private
    ):

        path = "/v1/model_store/add_model_family"
        model_store_payload = {
            "modelFamily": model_family,
            "modelInputs": model_inputs,
            "modelOutputs": model_outputs,
            "models": models,
            "description": description,
            "trainingFramework": training_framework,
            "supportedRuntimes": supported_runtimes,
            "benchmarkDatasets": benchmark_datasets,
            "supportedMetrics": supported_metrics,
            "pruningSupport": pruning_support,
            "codeRepository": code_repository,
            "trainingDockerContainer": training_docker_container,
            "dataProcessing": {
                "inputFormat": input_format,
                "dataLoaderClassDefinition": data_loader_class_definition,
                "dataLoaderCallSignature": data_loader_call_signature
            },
            "references": references,
            "isPrivate": is_private,
            "projectId": project_id
        }
        headers = {'Content-Type': 'application/json'}
        resp = self.rpc.post(path=path, headers=headers, payload=model_store_payload) 

        return self.handle_response(resp, "New model family created",
                                    "An error occured while creating model family")
    
    # To add a new entry into model info
    def add_model_info(
        self,
        model_key,
        model_name,
        model_family_id,
        params_millions,
        recommended_run_time,
        benchmark_results,
        run_time_results
    ):
        path = "/v1/model_store/add_model_info"
        model_store_payload = {
            "modelKey": model_key,
            "modelName": model_name,
            "_idModelFamily": model_family_id,
            "paramsMillions": params_millions,
            "recommendedRuntime": recommended_run_time,
            "benchmarkResults": benchmark_results,
            "runtimeResults": run_time_results
        }
        headers = {'Content-Type': 'application/json'}
        resp = self.rpc.post(path=path, headers=headers, payload=model_store_payload)

        return self.handle_response(resp, "New model family created",
                                    "An error occured while creating model info")
    
    # To add a new entry into model action config
    def add_model_action_config(
        self,
        model_info_id,
        action_type,
        action_config,
        docker_container_for_action,
        docker_container_for_evaluation,
        docker_credentials,
        private_docker,
        model_checkpoint,
        action_call_signature,
        export_format
    ):
        path = "/v1/model_store/add_model_action_config"
        model_store_payload = {
            "_idModelInfo": model_info_id,
            "actionType": action_type,
            "actionConfig": action_config,
            "dockerContainerForAction": docker_container_for_action,
            "dockerContainerForEvaluation": docker_container_for_evaluation,
            "dockerCredentials": docker_credentials,
            "privateDocker": private_docker,
            "modelCheckpoint": model_checkpoint,
            "actionCallSignature": action_call_signature,
            "exportFormat": export_format
        }
        headers = {'Content-Type': 'application/json'}
        resp = self.rpc.post(path=path, headers=headers, payload=model_store_payload)

        return self.handle_response(resp, "New model action config created",
                                    "An error occured while creating model action config")
    
    # To update model family
    def update_model_family(
        self,
        model_family_id,
        project_id,
        model_family,
        model_inputs,
        model_outputs,
        model_keys,
        description,
        training_framework,
        supported_runtimes,
        benchmark_datasets,
        supported_metrics,
        pruning_support,
        code_repository,
        training_docker_container,
        input_format, 
        data_loader_class_definition, 
        data_loader_call_signature,
        references,
        is_private
    ):
        path = f"/v1/model_store/model_family/{model_family_id}"
        model_store_payload = {
            "modelFamily": model_family,
            "modelInputs": model_inputs,
            "modelOutputs": model_outputs,
            "modelKeys": model_keys,
            "description": description,
            "trainingFramework": training_framework,
            "supportedRuntimes": supported_runtimes,
            "benchmarkDatasets": benchmark_datasets,
            "supportedMetrics": supported_metrics,
            "pruningSupport": pruning_support,
            "codeRepository": code_repository,
            "trainingDockerContainer": training_docker_container,
            "dataProcessing": {
                "inputFormat": input_format,
                "dataLoaderClassDefinition": data_loader_class_definition,
                "dataLoaderCallSignature": data_loader_call_signature
            },
            "references": references,
            "isPrivate": is_private
        }
        headers = {'Content-Type': 'application/json'}
        resp = self.rpc.put(path=path, headers=headers, payload=model_store_payload)

        return self.handle_response(resp, "Model family successfully updated",
                                    "An error occured while updating model family")
    
    # To update model info
    def update_model_info(
        self,
        model_info_id,
        model_key,
        model_name,
        model_family_id,
        params_millions,
        recommended_runtime,
        benchmark_results,
        runtime_results
    ):
        path = f"/v1/model_store/model_info/{model_info_id}"
        model_store_payload = {
            "modelKey": model_key,
            "modelName": model_name,
            "_idModelFamily": model_family_id,
            "paramsMillions": params_millions,
            "recommendedRuntime": recommended_runtime,
            "benchmarkResults": benchmark_results,
            "runtimeResults": runtime_results
        }
        headers = {'Content-Type': 'application/json'}
        resp = self.rpc.put(path=path, headers=headers, payload=model_store_payload)

        return self.handle_response(resp, "Model family successfully updated",
                                    "An error occured while updating model family")
    
    # To update model action config
    def update_model_action_config(
        self,
        model_action_config_id,
        model_info_id,
        action_type,
        action_config,
        docker_container_for_action,
        docker_container_for_evaluation,
        docker_credentials,
        private_docker,
        model_checkpoint,
        action_call_signature,
        export_format
    ):
        path = f"/v1/model_store/model_action_config/{model_action_config_id}"
        model_store_payload = {
            "_idModelInfo": model_info_id,
            "actionType": action_type,
            "actionConfig": action_config,
            "dockerContainerForAction": docker_container_for_action,
            "dockerContainerForEvaluation": docker_container_for_evaluation,
            "dockerCredentials": docker_credentials,
            "privateDocker": private_docker,
            "modelCheckpoint": model_checkpoint,
            "actionCallSignature": action_call_signature,
            "export_format": export_format
        }
        headers = {'Content-Type': 'application/json'}
        resp = self.rpc.put(path=path, headers=headers, payload=model_store_payload)

        return self.handle_response(resp, "Model family successfully updated",
                                    "An error occured while updating model family")
    
    def add_model_family_action_config(
        self, 
        model_family_id,
        action_type, 
        action_config, 
        docker_container_for_action,
        docker_container_for_evaluation,
        model_checkpoint,
        docker_credentials, 
        private_docker,
        action_call_signature,
        export_format
        
    ):
        path = f"/v1/model_store/add_model_family_config"
        model_store_payload = {
            "_idModelFamily": model_family_id,
            "actionType": action_type,
            "actionConfigs": action_config,
            "dockerCredentials": docker_credentials,
            "privateDocker": private_docker,
            "actionCallSignature": action_call_signature,
            "dockerContainerForAction": docker_container_for_action,
            "dockerContainerForEvaluation": docker_container_for_evaluation,
            "modelCheckpoint": model_checkpoint,
            "exportFormat": export_format,

        }
        headers = {'Content-Type': 'application/json'}
        resp = self.rpc.post(path=path, headers=headers, payload=model_store_payload)

        return self.handle_response(resp, "Modelfamily action config successfully added",
                                    "An error occured while adding model family action config")