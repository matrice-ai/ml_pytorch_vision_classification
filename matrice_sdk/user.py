import sys
from matrice_sdk.rpc import RPC


class User:
    def __init__(self, session):
        """Initialize User object with an RPC session"""
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
    def get_account_subscription(self, account_number):
        path = f"/v1/user/get_account_subscription/{account_number}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Account aubscription details fetched successfully",
                                    "Could not fetch account aubscription details")

    def list_invites(self):
        path = "/v2/user/project/invite"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Invites list fetch successfully",
                                    "Could not fetch invite list")

    def list_project_invites(self, project_id):
        path = f"/v2/user/project/{project_id}/invites?projectId={project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Project invites list fetch successfully",
                                    "Could not fetch project invite list")

    def list_collaborators(self, project_id):
        path = f"/v2/user/project/{project_id}/collaborators?projectId={project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Collaborators fetched successfully",
                                    "Could not fetch collaborators")


    #POST REQUESTS
    def invite_user(
            self,
            project_id,
            user_id,
            project_name,
            permissions
        ):
        path = f"/v2/user/project/invite?projectId={project_id}"
        headers = {"Content-Type": "application/json"}
        body = {
            "_idProject": project_id,
            "_idUser": user_id,
            "projectName": project_name,
            "permissions": permissions
        }
        resp = self.rpc.post(path=path, headers=headers, payload=body)

        return self.handle_response(resp, "User invited to the project successfully",
                                    "Could not invite user to the project")

    def accept_invite(self, invite_id):
        path = f"/v2/user/project/invite/{invite_id}/accept"
        resp = self.rpc.post(path=path)

        return self.handle_response(resp, "Invite accepted successfully",
                                    "Could not accept invite")


    #PUT REQUESTS
    def update_permissions(self, project_id, collaborator_id, permissions):
        [
            version,
            is_project_admin,
            datasets_service,
            models_service,
            annotations_service,
            byom_service,
            deployment_service,
            inference_service
        ] = permissions
        path = f"/v2/user/project/{project_id}/collaborators/{collaborator_id}?projectId={project_id}"
        headers = {"Content-Type": "application/json"}
        body = {
            "version": version,
            "isProjectAdmin": is_project_admin,
            "datasetsService": datasets_service,
            "modelsService": models_service,
            "annotationService": annotations_service,
            "byomService": byom_service,
            "deploymentService": deployment_service,
            "inferenceService": inference_service
        }
        resp = self.rpc.put(path=path, headers=headers, payload=body)

        return self.handle_response(resp, "Collaborator permissions updated successfully",
                                    "Could not update collaborator permissions")


    #DELETE REQUESTS
    def delete_invite(self, project_id):
        path = f"/v2/user/project/invite/{project_id}"
        resp = self.rpc.delete(path=path)

        return self.handle_response(resp, "Invite deleted successfully",
                                    "Could not delete invite")

    def delete_collaborator(self, project_id, collaborator_id):
        path = f"/v2/user/project/{project_id}/delete-collaborator/{collaborator_id}"
        resp = self.rpc.delete(path=path)

        return self.handle_response(resp, "Collaborator removed successfully",
                                    "Could not remove collaborator")
