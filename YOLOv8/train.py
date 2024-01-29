from python_sdk.src.actionTracker import ActionTracker
from python_sdk.matrice import Session
import sys
import os


def main(action_id):
    global actionTracker

    update_dataset_path()
    from ultralytics import YOLO

    session=Session()
    actionTracker = ActionTracker(session,action_id)

    model_config=actionTracker.get_job_params()

    stepCode='MDL_TRN_ACK'
    status='OK'
    status_description='Model Training has acknowledged'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)

    model_config = actionTracker.get_job_params()

    _idDataset = model_config['_idDataset']
    dataset_version = model_config['dataset_version']
    model_config.data = f'workspace/{str(_idDataset)}-{str(dataset_version).lower()}-yolo'

    model = YOLO(model_config.model_key+".pt")

    model.add_callback("on_fit_epoch_end", log_epoch_results)


    status="OK"
    status_description='Training Dataset is loaded'
    stepCode='MDL_TRN_DTL'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)

    stepCode='MDL_TRN_STRT'
    status='OK'
    status_description='Model Training has started'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)


    x=model.train(data=model_config.data+"/data.yaml",epochs=model_config.get("epochs",2))

    actionTracker.add_index_to_category(x.names)
    actionTracker.upload_checkpoint("runs/detect/train/weights/best.pt")

    save_eval_results(model_config,model,actionTracker)

def log_epoch_results(trainer):

    global actionTracker

    """training set metrics after each epoch."""
    tloss=trainer.tloss
    metrics=trainer.metrics

    try:
        epochDetails= [ {"splitType": "train", "metricName": "box_loss", "metricValue":tloss[0].item()},
                        {"splitType": "train", "metricName": "cls_loss", "metricValue": tloss[1].item()},
                        {"splitType": "train", "metricName": "dfl_loss", "metricValue": tloss[2].item()},
                        {"splitType": "val", "metricName": "box_loss", "metricValue":metrics['val/box_loss']},
                        {"splitType": "val", "metricName": "cls_loss", "metricValue": metrics['val/cls_loss']},
                        {"splitType": "val", "metricName": "dfl_loss", "metricValue": metrics['val/dfl_loss']},
                        {"splitType": "", "metricName": "precision", "metricValue": metrics['metrics/precision(B)']},
                        {"splitType": "", "metricName": "recall", "metricValue": metrics['metrics/recall(B)']},
                        {"splitType": "", "metricName": "mAP50", "metricValue": metrics['metrics/mAP50(B)']},
                        {"splitType": "", "metricName": "mAP50-95", "metricValue": metrics['metrics/mAP50-95(B)']}]

        actionTracker.log_epoch_results(trainer.epoch,epochDetails)

        print(epochDetails)
    except Exception as e:
        print(e)



from eval import get_metrics
def save_eval_results(model_config,model,actionTracker):

    payload=[]

    try:
        if  os.path.exists(os.path.join(model_config.data,"images/val")):
            eval=model.val(split="val")
            payload+=get_metrics(eval,'val')

        if  os.path.exists(os.path.join(model_config.data,"images/test")):
            eval=model.val(split="val")
            payload+=get_metrics(eval,'test')
        
        actionTracker.save_evaluation_results(payload)
        status = 'SUCCESS'
        status_description='Model Training is completed'

    except Exception as e:
        status = 'ERROR'
        status_description = 'Model training is completed but error in model saving or eval' + str(e)
            
    stepCode='MDL_TRN_CMPL'

    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)
    
def update_dataset_path():
  import yaml
  with open("/root/.config/Ultralytics/settings.yaml", 'r') as file:
      data = yaml.safe_load(file)
      data['datasets_dir']=data['datasets_dir'].split("/")
      if data['datasets_dir'][-1]== "datasets" : del data['datasets_dir'][-1]
      data['datasets_dir']="/".join(data['datasets_dir'])
      
      with open("/root/.config/Ultralytics/settings.yaml", 'w') as file:
          yaml.dump(data, file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 train.py <action_status_id>")
        sys.exit(1)
    action_status_id = sys.argv[1]
    main(action_status_id)