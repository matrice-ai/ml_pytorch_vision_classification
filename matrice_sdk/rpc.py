"""Module for RPC client handling Matrice.ai backend API requests."""

import os
import sys
from datetime import datetime, timedelta, timezone
import requests
from matrice_sdk.token_auth import RefreshToken, AuthToken

ENV = os.environ["ENV"]
BASE_URL = f"https://{ENV}.backend.app.matrice.ai"

class RPC:
    """RPC class for handling backend API requests with token-based authentication."""
    def __init__(self, project_id=""):
        """Initialize the RPC client with optional project ID."""
        # print("ininitalizing rpc client")
        self.project_id = project_id
        self.BASE_URL = BASE_URL
        self.Refresh_Token = RefreshToken()
        # print(self.refresh_token.__dict__, "refesh dict")
        self.AUTH_TOKEN = AuthToken(self.Refresh_Token)
        # print(self.AUTH_TOKEN)
        self.url_projectID = f"projectId={self.project_id}" if self.project_id else ""

    def send_request(self, method, path, headers=None, payload={},files=None,data=None):
        """Send an HTTP request to the specified endpoint."""
        self.refresh_token()
        request_url = f"{self.BASE_URL}{path}"
        request_url = self.add_project_id(request_url)
        try:
            # print("Sending request", request_url)
            # print(self.AUTH_TOKEN.__dict__)

            response = requests.request(
                method, request_url, auth=self.AUTH_TOKEN, headers=headers, json=payload,
                data=data,
                files=files)
            response_data = response.json()

        except Exception as e:#pylint:disable=W0718
            print("Error: ", e)
            sys.exit(0)

        return response_data

    def get(self, path, params={}):
        """Send a GET request to the specified endpoint."""
        return self.send_request("GET", path, payload=params)

    def post(self, path, headers=None, payload={},files=None,data=None):
        """Send a POST request to the specified endpoint."""
        return self.send_request("POST", path, headers=headers, payload=payload,files=files,data=data)

    def put(self, path, headers=None, payload={}):
        """Send a PUT request to the specified endpoint."""
        return self.send_request("PUT", path, headers=headers, payload=payload)

    def delete(self, path, headers=None, payload={}):
        """Send a DELETE request to the specified endpoint."""
        return self.send_request("DELETE", path, headers=headers, payload=payload)

    def refresh_token(self):
        """Refresh the authentication token if expired."""
        time_difference = datetime.utcnow().replace(
            tzinfo=timezone.utc
        ) - self.AUTH_TOKEN.expiry_time.replace(tzinfo=timezone.utc)

        time_diff = time_difference - timedelta(0)
        if time_diff.total_seconds() >= 0:
            self.AUTH_TOKEN = AuthToken(self.Refresh_Token)
        return

    def add_project_id(self, url):
        """Add project ID to the URL if present and not already included."""
        if not self.url_projectID or "?projectId" in url or "&projectId" in url:
            return url
        if "?" in url:
            url = url + "&" + self.url_projectID
        else:
            url = url + "?" + self.url_projectID
        return url
