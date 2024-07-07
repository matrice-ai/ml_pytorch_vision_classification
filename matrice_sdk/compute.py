import sys

class Compute:
    def __init__(self, session):
        self.project_id = session.project_id
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
    
    def get_all_instance_type(self):
        path = f"/v1/scaling/getAllInstancesType"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Instance list fetched successfully",
                                    "Could not fetch instance list")
    
    def get_all_account_compute(self, account_number):
        path = f"/v1/scaling/getAllAccountCompute/{account_number}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, f"Instance list fetched successfully for the account number: {account_number}",
                                    f"Could not fetch instance list for the account number: {account_number}")
    
    def get_threshold_details(self, account_number, compute_alias):
        path = f"/v1/scaling/get_threshold_details/{account_number}/{compute_alias}"
        resp = self.rpc.get(path=path)

        return self.handle_response(resp, "Instance threshold detailed fetched successfully",
                                    "Could not fetch instance threshold detailed")
    
    def add_account_compute(
            self,
            account_number,
            compute_alias,
            user_id,
            service_provider,
            instance_type,
            shutdown_time,
            launch_duration,
            lease_type="",
        ):

        path = f"/v1/scaling/addAccountCompute"
        headers = {"Content-Type": "application/json"}
        body = {
            "accountNumber": account_number,
            "alias": compute_alias,
            "projectID": self.project_id,
            "userID": user_id,
            "serviceProvider": service_provider,
            "instanceType": instance_type,
            "shutDownTime": shutdown_time,
            "leaseType": lease_type,
            "launchDuration": launch_duration
        }
        resp = self.rpc.post(path=path, headers=headers, payload=body)

        return self.handle_response(resp, f"Compute added successfully for the account number: {account_number}",
                                    f"Could not add compute for the account number: {account_number}")
    
    def update_account_compute(
            self,
            account_number,
            compute_alias,
            launch_duration,
            shutdown_threshold
        ):

        path = f"/v1/scaling/updateAccountCompute"
        headers = {"Content-Type": "application/json"}
        body = {
            "accountNumber": account_number,
            "computeAlias": compute_alias,
            "launchDuration": launch_duration,
            "shutDownTime": shutdown_threshold,
        }

        resp = self.rpc.put(path=path, headers=headers, payload=body)

        return self.handle_response(resp, f"Successfully updated the given compute",
                                    "Error updating the given compute")
    
    def delete_account_compute(
            self,
            account_number,
            compute_alias,
        ):

        path = f"/v1/scaling/deleteAccountCompute/{account_number}/{compute_alias}"
        headers = {"Content-Type": "application/json"}
        resp = self.rpc.put(path=path, headers=headers)

        return self.handle_response(resp, f"Successfully deleted the given compute",
                                    "Error deleting the given compute")