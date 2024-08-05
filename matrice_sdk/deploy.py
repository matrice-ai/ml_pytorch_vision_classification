from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Body
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import Optional
import time
import sys
import os
import urllib.request
import uvicorn
import threading
import requests

from .actionTracker import ActionTracker

class MatriceDeploy:

    def __init__(self, load_model, predict ,action_id, port):
        
        self.action_id = action_id
        self.actionTracker = ActionTracker(action_id)
        self.rpc = self.actionTracker.session.rpc
        
        self.action_details=self.actionTracker.action_details 
        print(self.action_details)

        self._idDeploymentInstance=self.action_details['_idModelDeployInstance']
        self._idDeployment=self.action_details['_idDeployment']
        self.model_id=self.action_details['_idModelDeploy']
        
        self.load_model = lambda actionTracker : load_model(actionTracker)
        self.predict = lambda model, image : predict(model, image)

        self.model = None
        self.last_no_inference_time = -1
        self.shutdown_on_idle_threshold = int(self.action_details['shutdownThreshold']) *60 
        self.app = FastAPI()
        self.ip = self.get_ip()
        self.port=int(port)
        self.run_shutdown_checker()

        self.load_time=None
        self.prediction_time=[]
        @self.app.post("/inference/")
        async def serve_inference(image: UploadFile = File(...)):
            image_data = await image.read()
            results, ok = self.inference(image_data)

            if ok:
                return JSONResponse(content=jsonable_encoder({"status": 1, "message": "Request success", "result": results}))
            else:
                return JSONResponse(content=jsonable_encoder({"status": 0, "message": "Some error occurred"}), status_code=500)
        
        @self.app.post("/inference_from_url/")
        async def serve_inference_from_url(imageUrl: str = Body(embed=True)):
            if imageUrl:
                response = requests.get(imageUrl)
                if response.status_code == 200:
                    image_data = response.content
                else:
                    return JSONResponse(content=jsonable_encoder({"status": 0, "message": "Failed to fetch image from URL"}), status_code=400)
            else:
                return JSONResponse(content=jsonable_encoder({"status": 0, "message": "Please provide imageUrl"}), status_code=400)
                
            results, ok = self.inference(image_data)

            if ok:
                return JSONResponse(content=jsonable_encoder({"status": 1, "message": "Request success", "result": results}))
            else:
                return JSONResponse(content=jsonable_encoder({"status": 0, "message": "Some error occurred"}), status_code=500)


    def start_server(self):
        host="0.0.0.0"
        port=80
        self.update_deployment_address()
        try:
            self.actionTracker.update_status("MDL_DPL_STR", "OK", "Model deployment started")
            uvicorn.run(self.app, host=host, port=port)
        except:
            self.actionTracker.update_status("ERROR", "ERROR", "Model deployment ERROR")


    def get_ip(self):
        external_ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')
        print(f"YOUR PUBLIC IP IS: {external_ip}")
        return external_ip


    def inference(self, image):
        now=time.time()
        if self.model is None:
            self.model = self.load_model(self.actionTracker)
            self.load_time=time.time()-now
            print(f"Time to load the model is : {self.load_time} seconds")
            now=time.time()

        self.last_no_inference_time = -1

        try:
            results = self.predict(self.model, image)
            print("Successfully ran inference")
            inference_time = time.time()-now
            self.prediction_time.append(inference_time)
            print(f"Time to predict is : {inference_time} seconds")
            print(f"AVG time for prediction is : {sum(self.prediction_time)/len(self.prediction_time)} seconds")
            return results, True
        except Exception as e:
            print(f"ERROR: {e}")
            return None, False
            


    def trigger_shutdown_if_needed(self):
        if self.last_no_inference_time == -1:
            self.last_no_inference_time = time.time()
        else:
            elapsed_time = time.time() - self.last_no_inference_time
            if elapsed_time > int(self.shutdown_on_idle_threshold):
                try:
                    print('Shutting down due to idle time exceeding the threshold.')
                    self.rpc.delete(f"/v1/deployment/delete_deploy_instance/{self._idDeploymentInstance}")
                    self.actionTracker.update_status("MDL_DPL_STP", "SUCCESS", "Model deployment STOP")
                    time.sleep(10)
                    os._exit(0)
                except Exception as e:
                    print(f"Error during shutdown: {e}")
                os._exit(1)
            else:
                print('Time since last inference:', elapsed_time)
                print('Time left to shutdown:', int(self.shutdown_on_idle_threshold) - elapsed_time)


    def shutdown_checker(self):
        while True:
            self.trigger_shutdown_if_needed()
            time.sleep(10)


    def run_shutdown_checker(self):
        t1 = threading.Thread(target=self.shutdown_checker, args=())
        t1.setDaemon(True)
        t1.start()


    def update_deployment_address(self):
        ip = self.get_ip()
        port = self.port
        
        url='/v1/deployment/update_deploy_instance_address'

        payload={
            "port":port,
            "ipAddress":ip,
            "_idDeploymentInstance":self._idDeploymentInstance,
            "_idModelDeploy":self._idDeployment
        }
        
        self.rpc.put(path=url,payload=payload)
