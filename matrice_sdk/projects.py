"""Module for interacting with backend API to manage projects."""

import sys
class Projects:
    """Class for handling project-related operations."""

    def __init__(self, session):
        """Initialize Projects object with project_id and rpc session"""
        self.project_id = session.project_id
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

    # GET REQUESTS
    def list_projects_by_ids(self):
        """Fetch a list of projects by project IDs."""
        path = "/v1/project"
        resp = self.rpc.get(path=path)

        return self.handle_response(
            resp,
            "Projects fetched successfully",
            "An error occurred while trying to fetch projects",
        )

    def get_a_project_by_id(self, project_id):
        """Fetch project information by project ID."""
        path = f"/v1/project/{project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(
            resp,
            f"Project info fetched for project with id {project_id}",
            f"Could not fetch project info for project with id {project_id}",
        )

    def get_service_action_logs(self, service_id):
        """Fetch action logs for a specific service."""

        # User can fetch service id using the get method of respective
        # services, eg - to get logs of dataset use get_dataset method
        path = f"/v1/project/service/{service_id}/logs?projectId={self.project_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(
            resp, "Action logs fected succesfully", "Could not fetch action logs"
        )

    def action_logs_from_action_account_number(self, account_number):
        """Fetch action logs for a specific action account number."""
        path = f"/v1/project/actions/action_records/{account_number}"
        resp = self.rpc.get(path=path)

        return self.handle_response(
            resp, "Action logs fected succesfully", "Could not fetch action logs"
        )
    
    def get_latest_action_record(self, service_id):
        """Fetch latest action logs for a specific service ID."""

        path = f"/v1/project/get_latest_action_record/{service_id}"
        resp = self.rpc.get(path=path)

        return self.handle_response(
            resp, "Action logs fected succesfully", "Could not fetch action logs"
        )

    # POST REQUESTS
    def create_project(self, project_name, input_type, output_type, enabled_platforms, account_number):
        """Create a new project."""
        path = "/v1/project"
        headers = {"Content-Type": "application/json"}
        body = {
            "name": project_name,
            "inputType": input_type,
            "outputType": output_type,
            "enabledPlatforms": enabled_platforms,
            "accountType": "enterprise",
            'accountNumber':account_number, #"0598274624636012576617365"
            # {"matrice":true,"android":false,"gCloudGPU":false,"ios":false,
            # "tpu":false,"intelCPU":false,"gcloudGPU":false},
        }
        resp = self.rpc.post(path=path, headers=headers, payload=body)
        if resp.get("success"):
            resp_data = resp.get("data")
            self.project_id = resp_data.get("_id")

        return self.handle_response(
            resp,
            "Project creation in progress",
            "An error occurred while trying to create a new project",
        )
    
    def delete_project(self,project_id):
        # Delete a project by project ID.
        _, error, _ = self.get_a_project_by_id(project_id)
        if error:
            print("Project is not found")
            sys.exit(1)
        
        path = f"/v1/project/delete_project/{project_id}"
        resp = self.rpc.delete(path=path)
        return self.handle_response(
            resp,
            "Project deleted successfully",
            "An error occurred while trying to delete project",
        )
    
    def enable_disable_project(self, project_id,type):
        """Enable or disable a project."""

        _, error, _ = self.get_a_project_by_id(project_id)
        if error:
            print("Project is not found")
            sys.exit(1)

        path = f"/v1/project/enable-disable-project/{type}/{project_id}"
        resp = self.rpc.put(path=path)

        return self.handle_response(
            resp,
            f"Project {project_id} {type}d successfully",
            f"Could not {type} project {project_id}",
        )
