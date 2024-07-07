"""Module for custom authentication"""
import os
import sys
import json
from datetime import datetime
import requests
from requests.auth import AuthBase
from dateutil.parser import parse

ENV = os.environ['ENV']
ACCESS_KEY_URL = f"https://{ENV}.backend.app.matrice.ai/v1/user/validate_access_key"
AUTH_TOKEN_URL = f"https://{ENV}.backend.app.matrice.ai/v2/user/auth/refresh"


class RefreshToken(AuthBase):
    """Implements a custom authentication scheme."""

    def __init__(self):
        self.bearer_token = None

    def __call__(self, r):
        """Attach an API token to a custom auth header."""
        self.set_bearer_token()
        r.headers['Authorization'] = self.bearer_token
        return r

    def set_bearer_token(self):
        """Obtain a bearer token using the provided access key and secret key."""
        print("Setting bearer token...")

        payload_dict = {
            "accessKey": os.environ['MATRICE_ACCESS_KEY_ID'],
            "secretKey": os.environ['MATRICE_SECRET_ACCESS_KEY']
        }
        payload = json.dumps(payload_dict)

        headers = {'Content-Type': 'text/plain'}
        try:
            response = requests.request(
                "GET", ACCESS_KEY_URL, headers=headers, data=payload,timeout=10)
        except Exception as e:#pylint: disable=W0718
            print("Error while making request to the auth server")
            print(e)
            sys.exit(0)

        if response.status_code != 200:
            print("Error response from the auth server")
            print(response.text)
            sys.exit(0)

        res_dict = response.json()

        if res_dict["success"]:
            self.bearer_token = "Bearer " + res_dict["data"]["refreshToken"]
        else:
            print("The provided credentials are incorrect!!")
            sys.exit(0)


class AuthToken(AuthBase):
    """Implements a custom authentication scheme."""

    def __init__(self, refresh_token):
        self.bearer_token = None
        self.refresh_token = refresh_token
        self.expiry_time = datetime.utcnow()

    def __call__(self, r):
        """Attach an API token to a custom auth header."""
        self.set_bearer_token()
        r.headers['Authorization'] = self.bearer_token
        return r

    def set_bearer_token(self):
        """Obtain an authentication bearer token using the provided refresh token."""
        print("Getting Auth bearer token...")

        headers = {
            "Content-Type": "application/json"
        }

        try:
            response = requests.request(
                "POST", AUTH_TOKEN_URL, headers=headers, auth=self.refresh_token,timeout=10)
        except Exception as e:#pylint: disable=W0718
            print("Error while making request to the auth server")
            print(e)
            sys.exit(0)

        if response.status_code != 200:
            print("Error response from the auth server")
            print(response.text)
            sys.exit(0)

        res_dict = response.json()

        if res_dict["success"]:
            self.bearer_token = "Bearer " + res_dict["data"]["token"]
            self.expiry_time = parse(res_dict["data"]["expiresAt"])
        else:
            print("The provided credentials are incorrect!!")
            sys.exit(0)
