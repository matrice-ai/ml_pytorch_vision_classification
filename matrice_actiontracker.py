import json
import boto3
from botocore.exceptions import ClientError
secrets_dict = {}
client = None
import os

ENV = os.environ['ENV']
access_key=os.environ['AWS_ACCESS_KEY_ID']
secret_key=os.environ['AWS_SECRET_ACCESS_KEY']


def get_secret_from_aws(secret_name):
    global client
    region_name = "us-west-2"
    secret_name = f"{ENV}/{secret_name}"
    print(secret_name)
    if secret_name in secrets_dict:
        return secrets_dict[secret_name]
    if client == None:
        session = boto3.session.Session(aws_access_key_id=access_key,
                        aws_secret_access_key=secret_key)
        client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secret = get_secret_value_response['SecretString']
    secrets_dict[secret_name] = secret
    return secret

def get_internal_api_key():
    secret_name = "internal_api_keys"
    secret_value = get_secret_from_aws(secret_name)

    try:
        secret_map = json.loads(secret_value)
        internal_api_key = secret_map.get("key1")

        if not internal_api_key or not isinstance(internal_api_key, str):
            print("Error: 'key1' is not a valid string or does not exist in the secret manager data.")
            return
        
        return internal_api_key

    except Exception as e:
        print(f"Error: {e}")

class Rpc:

    def __init__(self):
        secret_name = "env"
        res = json.loads(get_secret_from_aws(secret_name))
        base_url = res["base_url"]
        self.BASE_URL = base_url

        secret_name = "internal_api_keys"
        res = json.loads(get_secret_from_aws(secret_name))
        api_keys = res["key1"]
        self.API_KEY = api_keys

    def get(self, url):
        url = self.BASE_URL + url
        headers = {
            'X-API-Key': self.API_KEY
        }
        response = requests.request("GET", url, headers=headers)
        return response.json()['data']

    def put(self, url, payload):
        url = self.BASE_URL + url
        headers = {
        'Content-Type': 'application/json',
        'X-API-Key': self.API_KEY
        }
        response = requests.put(url, headers=headers, json=payload)  # Pass payload as JSON in the request body
        res = response.json()
        if 'data' in res.keys():
            return res['data']
        return True

    def post(self, url, data):
        url = self.BASE_URL + url
        payload = json.dumps(data)
        headers = {
            'Content-Type': 'application/json',
            'X-API-Key': self.API_KEY
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json()['data']

import sys
import sys
import json
import requests
from requests.auth import AuthBase

LOGIN_URL = "https://dev.backend.app.matrice.ai/v2/user/auth/login"

class TokenAuth(AuthBase):
    """Implements a custom authentication scheme."""

    def __init__(self, email, passwd):
        self.email = email
        self.password = passwd
        self.bearer_token = None

    def __call__(self, r):
        """Attach an API token to a custom auth header."""
        self.set_bearer_token()
        r.headers['Authorization'] = self.bearer_token  # Python 3.6+
        return r

    def set_bearer_token(self):

        print("Setting bearer token...")

        payload_dict = {"email":self.email, "password":self.password}
        payload = json.dumps(payload_dict)

        headers = {'Content-Type': 'text/plain'}
        try:
            response = requests.request("POST", LOGIN_URL, headers=headers, data=payload)
        except Exception as e:
            print("Error while making request to the auth server")
            print(e)
            sys.exit(0) 

        if(response.status_code != 200):
            print("Error response from the auth server")
            print(response.text)
            sys.exit(0)    

        res_dict = response.json()
    
        if res_dict["success"]:
            self.bearer_token = "Bearer "+ res_dict["data"]["token"]
        else:
            print("The provided credentials are incorrect!!")   
            sys.exit(0)  

BASE_URL = "https://dev.backend.app.matrice.ai"

class RPC:

    def __init__(self, email, password):
        self.BASE_URL = BASE_URL
        self.email = email
        self.password = password
        self.API_KEY = get_internal_api_key()


    def get(self, path):
        request_url = self.BASE_URL+path
        try:
            print("Sending request")
            response = requests.request("GET", request_url, auth=TokenAuth(self.email, self.password))
            response_data = response.json()
            
        except Exception as e:
            print("Error: ",e)
            sys.exit(0)

        return response_data


    def post(self, path,  headers=None, payload={}):
        request_url = self.BASE_URL+path
        payload = json.dumps(payload)
        try: 
            print("Sending request")
            response = requests.request("POST", request_url, headers=headers, data=payload, auth=TokenAuth(self.email, self.password))
            response_data = response.json()
        except Exception as e:
            print("Error: ",e)
            sys.exit(0) 
        return response_data
    
    def internal_post(self, path,  headers=None, payload={}):
        request_url = self.BASE_URL+path
        headers = {"X-Api-Key": self.API_KEY}
        payload = json.dumps(payload)
        try: 
            print("Sending request")
            response = requests.request("POST", request_url, headers=headers, data=payload)
            response_data = response.json()
        except Exception as e:
            print("Error: ",e)
            sys.exit(0) 
        return response_data


    def put(self, path,  headers=None, payload={}):
        request_url = self.BASE_URL+path
        payload = json.dumps(payload)
        try: 
            print("Sending request")
            response = requests.request("PUT", request_url, headers=headers, data=payload, auth=TokenAuth(self.email, self.password))
            response_data = response.json()
        except Exception as e:
            print("Error: ",e)
            sys.exit(0) 
        return response_data    

    def delete(self, path):

        request_url = self.BASE_URL+path
        try:
            print("Sending request")
            response = requests.request("DELETE", request_url, auth=TokenAuth(self.email, self.password))
            response_data = response.json()
            
        except Exception as e:
            print("Error: ",e)
            sys.exit(0)

        return response_data

class ModelLogging:

    def __init__(self, model_id=None, email="", password=""):

        self.model_id = model_id
        self.rpc = RPC(email, password)

    # Insert model log
    def insert_model_log_to_queue(self,model_id, action_id, epoch,epochDetails):
        
        model_log_payload = {
                "_idModel":model_id,
                "_idAction":action_id,
                "epoch":epoch,
                "epochDetails":epochDetails,
            }

        headers = {'Content-Type': 'application/json'}
        path = f"/model_logging/v1/model/{self.model_id}/train_epoch_log"
       
        resp=self.rpc.internal_post(path=path, headers=headers, payload=model_log_payload)
        if resp.get("success"):
            error = None
            message = "Model Epoch Train log message inserted successfully to queue"  
        else:
            error = resp.get("message") 
            message = "An error occured while trying to insert model log to queue"   

        return resp, error, message

class Model:

    def __init__(self, project_id, model_id=None, email="", password=""):

        self.project_id = project_id
        self.model_id = model_id
        self.rpc = RPC(email, password)

    def get_model_train(self, action_status_id):

        path = f"/v1/model/action_status/{action_status_id}"
        resp = self.rpc.get(path=path)
        if resp.get("success"):
            error = None
            message = "Model Train fetched successfully"
        else:
            error = resp.get("message")
            message = "Could not fetch models train"

        return resp, error, message
    
    def get_experiment_storage_path(self, experiment_id):

        path = f"/v1/model/experiment/{experiment_id}/storage_path"
        resp = self.rpc.get(path=path)
        if resp.get("success"):
            error = None
            message = "Storage path fetched successfully"
        else:
            error = resp.get("message")
            message = "Could not fetch the storage path"
    
        return resp, error, message


import os
import requests
import sys
import time
import boto3
from bson import ObjectId


def get_s3_client():
    s3_client = boto3.client('s3', aws_access_key_id=access_key,
                             aws_secret_access_key=secret_key)
    return s3_client

def upload_to_s3(source: str, bucket: str, key: str):
    s3_client = get_s3_client()
    s3_client.upload_file(source, bucket, key)
    s3_location = f"https://{bucket}.s3.amazonaws.com/{key}"
    return s3_location



class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ActionTracker:
    def __init__(self,action_id,email,password):

        self.email , self.password= email,password
        self.rpc=Rpc()
        self.action_id = ObjectId(action_id)
        self.action_id_str=str(self.action_id)


        url = f"/internal/project/v1/action/{self.action_id_str}/details"
        self.action_details = self.rpc.get(url)['actionDetails']
        self._idModelTrain=self.action_details['_idModelTrain']
        self._idModelTrain_str=str(self._idModelTrain)
        self.experiment_id=self.action_details['_idExperiment']
        self.model_logger=ModelLogging(self._idModelTrain_str,email=self.email,password=self.password)
        

    def get_job_params(self):


        url = f"/internal/project/v1/action/{self.action_id_str}/details"
        self.jobParams = self.rpc.get(url)['jobParams']

        return dotdict(self.jobParams)


    def update_status(self, action,service_name,stepCode, status, status_description):

        url= "/internal/project/v1/action"

        payload = {
            "_id":self.action_id_str,
            "action"  : action,
            "serviceName": service_name,
            "stepCode": stepCode,
            "status":status,
            "statusDescription":status_description,
        }

        x=self.rpc.put(url, payload)


    def log_epoch_results(self,  epoch,epoch_result_list):

        self.model_logger.insert_model_log_to_queue(self._idModelTrain_str,self.action_id_str,epoch,epoch_result_list)



    def upload_checkpoint(self,checkpoint_path,best_epoch_num=0 ):

        url= 'model/v1/update_storage_path'

        self.baseModelStoragePath_bucket=self.matriceModel.get_experiment_storage_path(self.experiment_id)[0]['data'].split('.com/')[-1]
                                            # Hard coded self.baseModelStoragePath_bucket instead of 'dev.images.download.speed.test' beacuse problem in dev.models
        s3_location=upload_to_s3(checkpoint_path, 'dev.images.download.speed.test', f'{self._idModelTrain_str}/{checkpoint_path.split("/")[-1]}')

        Payload = {
            "_idModelTrain": self._idModelTrain,
            "bestEpoch": best_epoch_num,
            "storage_path":s3_location
        }

        print(s3_location)
        #self.rpc.put(url,Payload)


    # def save_evaluation_results(self, data_split, result_dict):
    #     url= 'model/v1/save_evaluation_results'
    #     Payload = {
    #         "_idModelTrain": self._idModelTrain,
    #         "_idDataset": self._idDataset, # What if NONE
    #         "datasetVersion": self.datasetVersion,
    #         "splitType": data_split,
    #         "results": result_dict
    #     }
    #     # "results": {
    #     # "accuracy" : 0.23,
    #     # "precision": 0.45,}
    #     self.rpc.put(url,Payload)
