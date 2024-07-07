import sys

class Annotation:

    def __init__(self, session, dataset_id = None, annotation_id = None):
        self.project_id = session.project_id
        self.dataset_id = dataset_id
        self.annotation_id = annotation_id
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
    def get_all_annotations(self):
        path=f"/v1/annotations?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Annotations fetched successfully",
                                    "Could not fetch annotations")

    def get_all_summary(self):
        path=f"/v1/annotations/summary?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Annotation summary fetched successfully",
                                    "Could not fetch annotation summary")
    
    def get_annotation_by_id(self):
        path=f"/v1/annotations/{self.annotation_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Annotation fetched successfully",
                                    "Could not fetch annotation")
    
    def get_categories_by_annotation_id(self):
        path=f"/v1/annotations/{self.annotation_id}/categories"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Categories fetched successfully",
                                    "Could not fetch categories")
    
    def get_annotation_files(self):
        path=f"/v1/annotations/{self.annotation_id}/files?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Annotation files fetched successfully",
                                    "Could not fetch annotation files")
    
    
    def get_annotation_history(self,annotation_item_id):
        path=f"/v1/annotations/{self.annotation_id}/{annotation_item_id}/annotation_history?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Annotation history fetched successfully",
                                    "Could not fetch annotation history")
    
    def get_annotation_summary(self):
        path=f"/v1/annotations/{self.annotation_id}/summary"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Annotation item summary fetched successfully",
                                    "Could not fetch annotation item summary")
    
    def get_annotation_files(self):
        if self.annotation_id is None:
            print(
                "Annotation id not set for this dataset. Cannot perform the operation for annotation without annotation id"
            )
            sys.exit(0)

        path = f"/v1/annotations/{self.annotation_id}/files?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Sucessfully fetched the annotation item files",
                                    "Could not fetch annotation item files")

   #PUT REQUESTS
    def rename_annotation(self, annotation_title):
        if self.annotation_id is None:
            print("Dataset id not set for this dataset. Cannot perform the operation for dataset without dataset id")
            sys.exit(0)
        
        path = f"/v1/annotations/{self.annotation_id}"
        headers = {"Content-Type": "application/json"}
        body = {"title": annotation_title}
        resp = self.rpc.put(path=path, headers=headers, payload=body)

        return self.handle_response(resp, "Update data item label in progress",
                                    "Could not update the date item label")


    #POST REQUESTS
    def create_annotation(
            self,
            project_type,
            ann_title,
            dataset_version,
            labels,
            only_unlabeled,
            is_ML_assisted,
            labellers,
            reviewers,
            guidelines):
        path=f"/v1/annotations?projectId={self.project_id}&projectType={project_type}"
        payload={
            "title":ann_title,
            "_idDataset":self.dataset_id,
            "datasetVersion":dataset_version,
            "labels":labels,
            "onlyUnlabeled":only_unlabeled,
            "isMLAssisted":is_ML_assisted,
            "labellers":labellers,  
            "reviewers":reviewers,
            "guidelines":guidelines,
            "type":project_type,
            "modelType": "",
            "modelId": ""
        }
        headers = {'Content-Type': 'application/json'}
        resp = self.rpc.post(path=path, headers=headers, payload=payload)

        return self.handle_response(resp, "Annotation added successfully",
                                    "An error occured while adding annotation")
    
    def annotate(
            self,
            file_id,
            annotation_item_id,
            updated_classification_label,
            labeller,
            reviewer,
            status,
            issues,
            label_time,
            review_time,
            ):
        path=f"/v1/annotations/{self.annotation_id}/files/{file_id}/annotate?projectId={self.project_id}"
        payload={
             "annotationId":self.annotation_id,
             "annotationItemId":annotation_item_id,
             "labeller":labeller,            #{"_idUser":"65c0a3e262384d7c949fd94c","name":"Anutosh Agrawal"},
             "reviewer":reviewer,            #{"_idUser":"65d46105b480d4879fc75c46","name":"Khushi Sahay"},
             "updatedClassificationLabel":updated_classification_label,   #{"_idCategory":"65dc690d97a85d665bdb3bff","categoryName":"dog"},
             "status":status,
             "issues":issues,
             "labelTime":label_time,
             "reviewTime":review_time
        }
        headers = {'Content-Type': 'application/json'}
        resp = self.rpc.post(path=path, headers=headers, payload=payload)

        return self.handle_response(resp, "Annotation added successfully",
                                    "An error occured while adding annotation")

    def create_dataset(
            self,
            is_create_new,
            old_dataset_version,
            new_dataset_version,
            new_version_description
        ):
        path=f"/v1/annotations/{self.annotation_id}/create_dataset?projectId={self.project_id}"

        payload={
            "annotationId":self.annotation_id,
            "isCreateNew":is_create_new,
            "oldDatasetVersion":old_dataset_version,
            "newDatasetVersion":new_dataset_version,
            "newVersionDescription":new_version_description,  
            "datasetDesc": ""
        }
        headers = {'Content-Type': 'application/json'}
        resp = self.rpc.post(path=path, headers=headers, payload=payload)

        return self.handle_response(resp, "Annotation added successfully",
                                    "An error occured while adding annotation")


    def create_category(self, labelname):
        if self.annotation_id is None:
            print(
                "Annotation id not set for this annotation. Cannot download without annotation id"
            )
            sys.exit(0)

        body = {
            "_idAnnotation": self.annotation_id,
            "name": labelname
        }
        headers = {"Content-Type": "application/json"}
        path = f"/v1/annotations/{self.annotation_id}/categories?projectId={self.project_id}"

        resp = self.rpc.post(path=path, headers=headers, payload=body)
        return self.handle_response(resp, "Category added successfully",
                                    "An error occured while adding Category")
    
    def delete_annotation(self):

        if self.annotation_id is None:
            print(
                "Annotation id not set for this dataset. Cannot perform the deletion for annotation without annotation id"
            )
            sys.exit(0)

        path = f"/v1/annotations/{self.annotation_id}?projectId={self.project_id}" # check

        resp = self.rpc.delete(path=path)
        return self.handle_response(resp, "Annotation deleted successfully",
                                    "An error occured while deleting annotation")
    
    def delete_annotation_item(self, annotation_item_id):

        if self.annotation_id is None:
            print(
                "Annotation id not set for this dataset. Cannot perform the deletion for annotation without annotation id"
            )
            sys.exit(0)

        path = f"/v1/annotations/{self.annotation_id}/files/{annotation_item_id}?projectId={self.project_id}" # check

        resp = self.rpc.delete(path=path)
        return self.handle_response(resp, "Annotation Item deleted successfully",
                                    "An error occured while deleting annotation item")
    