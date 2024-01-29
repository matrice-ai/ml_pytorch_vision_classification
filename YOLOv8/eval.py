from python_sdk.src.actionTracker import ActionTracker
from python_sdk.matrice import Session
import sys
import os

def main(action_id):

    from train import update_dataset_path
    update_dataset_path()  # Needs to be calles before importing YOLO
    from ultralytics import YOLO

    session=Session()
    actionTracker = ActionTracker(session,action_id)

    model_config=actionTracker.get_job_params()

    stepCode='MDL_EVL_ACK'
    status='OK'
    status_description='Model Training has acknowledged'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)

    model_config = actionTracker.get_job_params()

    _idDataset = model_config['_idDataset']
    dataset_version = model_config['dataset_version']
    model_config.data = f'workspace/{str(_idDataset)}-{str(dataset_version).lower()}-yolo'
    

    actionTracker.download_model("yolo.pt")
    model = YOLO("yolo.pt")

    stepCode='MDL_EVL_STR'
    status='OK'
    status_description='Model Training has started'
    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)
    
    save_eval_results(model_config,model,actionTracker)

def save_eval_results(model_config,model,actionTracker):

    payload=[]

    try:
        if 'val' in model_config.split_types and os.path.exists(os.path.join(model_config.data,"images/val")):
            metrics=model.val(split="val")
            payload+=get_metrics(metrics,'val')

        if  'test' in model_config.split_types and os.path.exists(os.path.exists(os.path.join(model_config.data,"images/test"))):
            metrics=model.val(split="test")
            payload+=get_metrics(metrics,'test')

        if 'train' in model_config.split_types and os.path.exists(os.path.exists(os.path.join(model_config.data,"images/train"))):
            metrics=model.val(split="train")
            payload+=get_metrics(metrics,'train')
        
        actionTracker.save_evaluation_results(payload)
        status = 'SUCCESS'
        status_description='Model Evaluation is completed'

    except Exception as e:
        status = 'ERROR'
        status_description = 'ERROR in Model Evaluation' + str(e)
            
    stepCode='MDL_EVL_CMPL'

    print(status_description)
    actionTracker.update_status(stepCode,status,status_description)

def get_metrics(x,split):
    results=[]
    results_for_all=x.results_dict

    results.append({
            "category":"all",
             "splitType":split,
             "metricName":"precision",
            "metricValue":results_for_all['metrics/precision(B)']
         })

    results.append({
        "category":"all",
            "splitType":split,
            "metricName":"recall",
        "metricValue":results_for_all['metrics/recall(B)']
        })

    results.append({
        "category":"all",
            "splitType":split,
            "metricName":"mAP50",
        "metricValue":results_for_all['metrics/mAP50(B)']
        })

    results.append({
        "category":"all",
            "splitType":split,
            "metricName":"mAP50-95",
        "metricValue":results_for_all['metrics/mAP50-95(B)']
        })

    results.append({
        "category":"all",
            "splitType":split,
            "metricName":"fitness",
        "metricValue":results_for_all['fitness']
        })

    precision =x.box.p
    recall =x.box.r
    f1_score=x.box.f1
    ap=x.box.ap
    map=x.box.maps

    index_to_category=x.names
    for i in range(len(x.box.ap_class_index)):

        results.append({
            "category":index_to_category[x.box.ap_class_index[i]],
            "splitType":split,
            "metricName":"precision",
            "metricValue":precision[i]
        })
        results.append({
            "category":index_to_category[x.box.ap_class_index[i]],
            "splitType":split,
            "metricName":"recall",
            "metricValue":recall[i]
        })
        results.append({
            "category":index_to_category[x.box.ap_class_index[i]],
            "splitType":split,
            "metricName":"f1_score",
            "metricValue":f1_score[i]
        })
        results.append({
            "category":index_to_category[x.box.ap_class_index[i]],
            "splitType":split,
            "metricName":"AP",
            "metricValue":ap[i]
        })
        results.append({
            "category":index_to_category[x.box.ap_class_index[i]],
            "splitType":split,
            "metricName":"mAP",
            "metricValue":map[i]
        })

    return results

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 eval.py <action_status_id>")
        sys.exit(1)
    action_status_id = sys.argv[1]
    main(action_status_id)