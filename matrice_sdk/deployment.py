import sys
from matrice_sdk.models import Model
from matrice_sdk.matrice import Session

class Deployment:

    def __init__(self, session, deployment_id=None):
        """Initialize Deployment instance with the given session and optional deployment_id."""
        self.project_id = session.project_id
        self.deployment_id = deployment_id
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
    def get_deployment(self):
        path = f"/v1/deployment/{self.deployment_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(
            resp,
            "Deployment fetched successfully",
            "An error occurred while trying to fetch deployment.",
        )
    
    def get_deployment_summary(self):
        path = f"/v1/deployment/summary?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(
            resp,
            "Deployment summary fetched successfully",
            "An error occurred while trying to fetch deployment summary.",
        )
    
    def list_deployments(self):
        """List all deployments inside the project."""
        path = f"/v1/deployment/list_deployments?projectId={self.project_id}"
        resp = self.rpc.get(path=path)
        return self.handle_response(
                    resp,
                    "Deployment list fetched successfully",
                    "An error occurred while trying to fetch deployment list.",
                ) 

    def get_deployment_server(self,model_train_id,model_type):
        path = f"/v1/deployment/get_deploy_server/{model_train_id}/{model_type}"
        resp = self.rpc.get(path=path)
        return self.handle_response(
                    resp,
                    "Deployment server fetched successfully",
                    "An error occurred while trying to fetch deployment server.",
                )
    
    def wakeup_deployment_server(self):
        path = f"/v1/deployment/wake_up_deploy_server/{self.deployment_id}"
        resp = self.rpc.get(path=path)
        return self.handle_response(
                    resp,
                    "Deployment server has been successfully awakened",
                    "An error occurred while attempting to wake up the deployment server.",
                )
    
    def get_deployment_status_cards(self):
        path = f"/v1/deployment/status_cards?projectId={self.project_id}"
        resp = self.rpc.get(path=path)
        return self.handle_response(
                    resp,
                    "Deployment status cards fetched successfully.",
                    "An error occurred while trying to fetch deployment status cards.",
                )

    #POST REQUESTS
    def create_deployment(
            self,
            deployment_name,
            model_id,
            gpu_required=True,
            auto_scale=False,
            auto_shutdown=True,
            shutdown_threshold=5,
            image_store_confidence_threshold=0.9,
            image_store_count_threshold=50
            ):
        """Create a new deployment for the specified model."""
        session=Session(self.project_id)
        model=Model(session)
        trained_model_list,err,_=model.list_completed_model_train()
        if err is not None:
            print(
                "Models not found."
            )
            sys.exit(0)
        
        model_info=None
        for model_item in trained_model_list:
            if model_item["_id"] == model_id:
                model_info= model_item
        
        if(model_info is None):
            print(
                "Models info for the corresponding model id not found."
            )
            sys.exit(0)
        
        body = {
                "deploymentName":deployment_name,
                "_idModel":model_id,
                "runtimeFramework":model_info["targetRuntime"][0],
                "deploymentType":"regular",
                "modelType":model_info["status"],
                "modelInput":model_info["modelInputs"][0],
                "modelOutput":model_info["modelOutputs"][0],
                "autoShutdown":auto_shutdown,
                "autoScale":auto_scale,
                "gpuRequired":gpu_required,
                "shutdownThreshold":shutdown_threshold,
                "imageStoreConfidenceThreshold":image_store_confidence_threshold,
                "imageStoreCountThreshold":image_store_count_threshold,
                "bucketAlias": "",
                "computeAlias": "",
                "credentialAlias": ""
        } 

        headers = {'Content-Type': 'application/json'}
        path = f"/v1/deployment?projectId={self.project_id}"
       
        resp=self.rpc.post(path=path, headers=headers, payload=body)
        if resp.get("success"):
            resp_data = resp.get("data")
            self.deployment_id=resp_data
        return self.handle_response(
                                    resp,
                                    "Deployment created successfully.",
                                    "An error occurred while trying to create deployment.",
                                )    
    
    def create_auth_key(self, expiry_days):
        """Create a new deployment for the specified model."""
        body = {"expiryDays":expiry_days,
                "authType":"external"}

        headers = {'Content-Type': 'application/json'}
        path = f"/v1/deployment/add_auth_key/{self.deployment_id}?projectId={self.project_id}"
       
        resp=self.rpc.post(path=path, headers=headers, payload=body)
        return self.handle_response(
                            resp,
                            "Auth Key created successfully.",
                            "An error occurred while trying to create auth key.",
                        )
    
    #PUT REQUESTS
    def update_deployment_name(self, updated_name):
        if self.deployment_id is None:
            print("Deployment id not set for this model.")
            sys.exit(0)

        body = {"deploymentName":updated_name}
            
        
        headers = {"Content-Type": "application/json"}
        path = f"/v1/deployment/update_deployment_name/{self.deployment_id}"
        resp = self.rpc.put(path=path, headers=headers, payload=body)

        return self.handle_response(resp, f"Deployment name updated to {updated_name}",
                                    "Could not update the deployment name")

    #DELETE REQUESTS
    def delete_deployment(self,deployment_id):
        """Delete the current deployment."""
        path = f"/v1/deployment/delete_deployment/{deployment_id}"

        resp = self.rpc.delete(path=path)
        return self.handle_response(
                    resp,
                    "Deployment deleted successfully.",
                    "An error occurred while trying to delete the deployment.",
                )
    
    def delete_auth_key(self,auth_key):
        """Delete the current deployment."""
        if self.deployment_id is None:
            print(
                "Deployment id not set for this deployment."
            )
            sys.exit(0)

        path = f"/v1/deployment/delete_auth_key/{self.deployment_id}/{auth_key}"

        resp = self.rpc.delete(path=path)
        return self.handle_response(
                    resp,
                    "Auth key deleted successfully.",
                    "An error occurred while trying to delete the auth key.",
                )
