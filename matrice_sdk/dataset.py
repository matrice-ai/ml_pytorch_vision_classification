import sys


class Dataset:
    def __init__(self, session, dataset_id=None):
        self.project_id = session.project_id
        self.dataset_id = dataset_id
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

    # GET REQUESTS

    def list_datasets(self):
        path = f"/v1/dataset?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Dataset list fetched successfully",
                                    "Could not fetch dataset list")

    def list_complete_dataset(self):
        path = f"/v1/dataset/complete?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Completed dataset list fetched successfully",
                                    "Could not fetch completed dataset list")

    def get_summary(self, dataset_version):
        if self.dataset_id is None:
            print("Dataset id not set for this dataset. Cannot perform the operation for dataset without dataset id")
            sys.exit(0)

        path = f"/v1/dataset/{self.dataset_id}/version/{dataset_version}/summary?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Dataset summary fetched successfully",
                                    "Could not fetch dataset summary")

    def get_dataset(self):
        if self.dataset_id is None:
            print("Dataset id not set for this dataset. Cannot perform the operation for dataset without dataset id")
            sys.exit(0)

        path = f"/v1/dataset/{self.dataset_id}?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Dataset fetched successfully",
                                    "Could not fetch dataset")

    def get_categories(self, dataset_version):
        if self.dataset_id is None:
            print("Dataset id not set for this dataset. Cannot perform the operation for dataset without dataset id")
            sys.exit(0)

        path = f"/v1/dataset/{self.dataset_id}/version/{dataset_version}/categories?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, f"Dataset categories for version - {dataset_version} fetched successfully",
                                    "Could not fetch dataset categories")

    def list_items_V2(self, dataset_version):
        if self.dataset_id is None:
            print("Dataset id not set for this dataset. Cannot perform the operation for dataset without dataset id")
            sys.exit(0)

        path = f"/v1/dataset/{self.dataset_id}/version/{dataset_version}/v2/item?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, f"Dataset items for version - {dataset_version} fetched successfully",
                                    "Could not fetch dataset items")

    def list_items(self, dataset_version):
        if self.dataset_id is None:
            print("Dataset id not set for this dataset. Cannot perform the operation for dataset without dataset id")
            sys.exit(0)

        path = f"/v1/dataset/{self.dataset_id}/version/{dataset_version}/item?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, f"Dataset items for version - {dataset_version} fetched successfully",
                                    "Could not fetch dataset items")

    def get_processed_versions(self):
        if self.dataset_id is None:
            print("Dataset id not set for this dataset. Cannot perform the operation for dataset without dataset id")
            sys.exit(0)

        path = f"/v1/dataset/get_processed_versions?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, f"Processed versions fetched successfully",
                                    "Could not fetch processed versions")

    # PUT REQUESTS

    def update_dataset(self, updated_name):
        if self.dataset_id is None:
            print("Dataset id not set for this dataset. Cannot perform the operation for dataset without dataset id")
            sys.exit(0)

        path = f"/v1/dataset/{self.dataset_id}?projectId={self.project_id}"
        headers = {"Content-Type": "application/json"}
        body = {"name": updated_name}
        resp = self.rpc.put(path=path, headers=headers, payload=body)

        return self.handle_response(resp, f"Successfully updated dataset name to {updated_name}",
                                    "Could not update datename")

    def update_data_item_label(self, dataset_version, item_id, label_id):
        if self.dataset_id is None:
            print("Dataset id not set for this dataset. Cannot perform the operation for dataset without dataset id")
            sys.exit(0)

        path = f"/v1/dataset/{self.dataset_id}/version/{dataset_version}/item/{item_id}/label?projectId={self.project_id}"
        headers = {"Content-Type": "application/json"}
        body = {"labelId": label_id}
        resp = self.rpc.put(path=path, headers=headers, payload=body)

        return self.handle_response(resp, "Update data item label in progress",
                                    "Could not update the date item label")

    # POST REQUESTS
    def create_dataset(
        self,
        dataset_name,
        source,
        source_url,
        cloud_provider,
        dataset_type,
        input_type,
        credential_alias="",
        bucket_alias="",
        compute_alias="",
        dataset_description="",
        version_description="",
        source_credential_alias="",
        bucket_alias_service_provider="auto",
    ):
        dataset_size, err, msg = self.get_dataset_size(source_url)
        if err:
            dataset_size = 0
        path = f"/v1/dataset?projectId={self.project_id}"
        headers = {"Content-Type": "application/json"}
        body = {
            "name": dataset_name,
            "isUnlabeled": False,
            "source": source,
            "sourceUrl": source_url,
            "cloudProvider": cloud_provider,
            "isCreateNew": True,
            "oldDatasetVersion": None,
            "newDatasetVersion": "v1.0",
            "datasetDescription": dataset_description,
            "description": dataset_description,
            "newVersionDescription": version_description,
            "isPublic": False,
            "computeAlias": compute_alias,
            "datasetSize": dataset_size,
            "bucketAliasServiceProvider": bucket_alias_service_provider,
            "_idProject": self.project_id,
            "type": dataset_type,
            "sourceCredentialAlias": source_credential_alias,
            "credentialAlias": credential_alias,
            "bucketAlias": bucket_alias,
            "inputType": input_type
        }
        resp = self.rpc.post(path=path, headers=headers, payload=body)

        return self.handle_response(resp, "Dataset creation in progress",
                                    "An error occured while trying to create new dataset")

    def create_dataset_import(
        self,
        source,
        source_url,
        new_dataset_version,
        old_dataset_version,
        dataset_description="",
        version_description="",
        compute_alias="",
    ):
        if self.dataset_id is None:
            print("Dataset id not set for this dataset. Cannot perform the operation for dataset without dataset id")
            sys.exit(0)

        dataset_resp, err, message = self.get_dataset()
        if err is not None:
            return dataset_resp, err, message

        stats = dataset_resp["stats"]
        if dataset_description == "":
            dataset_description = dataset_resp["datasetDesc"]

        for stat in stats:
            if stat["version"] != old_dataset_version:
                continue
            if stat["versionStatus"] != "processed":
                resp = {}
                err = None
                message = f"Only the dataset versions with complete status can be updated.Version {old_dataset_version} of the dataset doesn't have status complete."
                return resp, err, message

            if version_description == "" and old_dataset_version == new_dataset_version:
                version_description = stat["versionDescription"]
            break
        
        is_created_new = new_dataset_version == old_dataset_version
        path = f"v1/dataset/{self.dataset_id}/import?project={self.project_id}"
        headers = {"Content-Type": "application/json"}
        body = {
            "source": source,
            "sourceUrl": source_url,
            "isCreateNew": is_created_new,
            "isUnlabeled": False,
            "newDatasetVersion": new_dataset_version,
            "oldDatasetVersion": old_dataset_version,
            "newVersionDescription": version_description,
            "datasetDesc": dataset_description,
            "computeAlias": compute_alias,
        }
        resp = self.rpc.post(path=path, headers=headers, payload=body)

        return self.handle_response(resp, "New data item addition in progress",
                                    "An error occured while trying to add new data item.")

    def split_data(
        self,
        old_dataset_version,
        new_dataset_version,
        is_random_split,
        train_num=0,
        val_num=0,
        test_num=0,
        transfers=[{"source": "", "destination": "", "transferAmount": 1}],
        dataset_description="",
        version_description="",
        new_version_description="",
        compute_alias="",
    ):
        if self.dataset_id is None:
            print("Dataset id not set for this dataset. Cannot perform the operation for dataset without dataset id")
            sys.exit(0)

        dataset_resp, err, message = self.get_dataset()
        if err is not None:
            return dataset_resp, err, message

        stats = dataset_resp["stats"]
        if dataset_description == "":
            dataset_description = dataset_resp["datasetDesc"]

        for stat in stats:
            if stat["version"] != old_dataset_version:
                continue
            if stat["versionStatus"] != "processed":
                resp = {}
                err = None
                message = f"Only the dataset versions with complete status can be updated.Version {old_dataset_version} of the dataset doesn't have status complete."
                return resp, err, message

            if version_description == "" and old_dataset_version == new_dataset_version:
                version_description = stat["versionDescription"]
            break

        path = f"/v1/dataset/{self.dataset_id}/split_data?projectId={self.project_id}"
        headers = {"Content-Type": "application/json"}
        body = {
            "trainNum": train_num,
            "testNum": test_num,
            "valNum": val_num,
            "unassignedNum": 0,
            "oldDatasetVersion": old_dataset_version,
            "newDatasetVersion": new_dataset_version,
            "isRandomSplit": is_random_split,
            "datasetDesc": dataset_description,
            "newVersionDescription": new_version_description,
            "transfers": transfers,
            "computeAlias": compute_alias,
        }
        resp = self.rpc.post(path=path, headers=headers, payload=body)

        return self.handle_response(resp, "Dataset spliting in progress",
                                    "An error occured while trying to split the data.")
        
    def create_dataset_from_deployment(
            self,
            dataset_name,
            is_unlabeled,
            source,
            source_url,
            deployment_id,
            is_public,
            dataset_type,
            project_type,
            dataset_description="",
            version_description="",
    ):
        dataset_size,err,msg=self.get_dataset_size(source_url)
        print(f'dataset size is = {dataset_size}')
        path = f"/v1/dataset/deployment?projectId={self.project_id}"
        headers = {"Content-Type": "application/json"}
        body = {
            "name":dataset_name,
            "isUnlabeled": is_unlabeled, #false,
            "source": source, #"lu",
            "sourceUrl": source_url, #"https://s3.us-west-2.amazonaws.com/dev.dataset/test%2Fb34ea15a-1f52-48a3-9a70-d43688084441.zip",
            "_idDeployment": deployment_id,
            "cloudProvider":"AWS",
            "isCreateNew": True,
            "oldDatasetVersion": None,
            "newDatasetVersion": "v1.0",
            "datasetDescription": dataset_description,
            "newVersionDescription": version_description,
            "isPublic": is_public, #false,
            "computeAlias": "",
            "targetCloudStorage": "GCP",
            "inputType": "MSCOCO",
            "copyData": false,
            "isPrivateStorage": false,
            "cloudStoragePath": "",
            "urlType": "",
            "datasetSize": 0,
            "deleteDeploymentDataset": false,
            "_idProject": self.project_id,
            "type": project_type
        }
        resp = self.rpc.post(path=path, headers=headers, payload=body)

        print(resp)

        return self.handle_response(resp, "Dataset creation in progress",
                                    "An error occured while trying to create new dataset")

    # DELETE REQUESTS
    def delete_dataset_item_classification(self, dataset_version, dataset_item_ids):
        if self.dataset_id is None:
            print("Dataset id not set for this dataset. Cannot perform the operation for dataset without dataset id")
            sys.exit(0)

        path = f"/v1/dataset/version/{dataset_version}/dataset_item_classification?projectId={self.project_id}&datasetId={self.dataset_id}"
        requested_payload = {"datasetItemIds": dataset_item_ids}
        headers = {"Content-Type": "application/json"}
        resp = self.rpc.delete(path=path, headers=headers,
                               payload=requested_payload)

        return self.handle_response(resp, f"Given dataset items deleted successfully",
                                    "Could not delete the given dataset items")

    def delete_dataset_item_detection(self, dataset_version, dataset_item_ids):
        if self.dataset_id is None:
            print("Dataset id not set for this dataset. Cannot perform the operation for dataset without dataset id")
            sys.exit(0)

        path = f"/v1/dataset/version/{dataset_version}/dataset_item_detection?projectId={self.project_id}&datasetId={self.dataset_id}"
        requested_payload = {"datasetItemIds": dataset_item_ids}
        headers = {"Content-Type": "application/json"}
        resp = self.rpc.delete(path=path, headers=headers,
                               payload=requested_payload)

        return self.handle_response(resp, f"Given dataset items deleted successfully",
                                    "Could not delete the given dataset items")

    def delete_version(self, dataset_version):
        if self.dataset_id is None:
            print("Dataset id not set for this dataset. Cannot perform the operation for dataset without dataset id")
            sys.exit(0)

        path = f"/v1/dataset/{self.dataset_id}/version/{dataset_version}?projectId={self.project_id}"
        resp = self.rpc.delete(path=path)

        return self.handle_response(resp, f"Successfully deleted version - {dataset_version}",
                                    "Could not delete the said version")

    def delete_dataset(self):
        if self.dataset_id is None:
            print("Dataset id not set for this dataset. Cannot perform the operation for dataset without dataset id")
            sys.exit(0)

        path = f"/v1/dataset/{self.dataset_id}?projectId={self.project_id}"
        resp = self.rpc.delete(path=path)

        return self.handle_response(resp, f"Successfully deleted the dataset",
                                    "Could not delete the dataset")

    def get_dataset_size(self, url):
        path = f"/v1/dataset/get_dataset_size_in_mb_from_url?projectId={self.project_id}"
        requested_payload = {"datasetUrl": url}
        headers = {"Content-Type": "application/json"}
        resp = self.rpc.post(path=path, headers=headers,
                             payload=requested_payload)

        return self.handle_response(resp, f"Dataset size fetched successfully",
                                    "Could not fetch dataset size")
