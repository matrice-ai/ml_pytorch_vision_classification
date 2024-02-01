from datetime import datetime
import traceback
from bson import ObjectId
from python_common.services.rpc import Rpc

def error_log(exception, _id,moreinfo={}):
    traceback_str = traceback.format_exc().rstrip()
    traceback_info = traceback.extract_tb(exception.__traceback__)[-1]
    filename, line_num, function_name, _ = traceback_info
    log_err = {
        "serviceName": "Job-Scheduler",
        "stackTrace": traceback_str,
        "errorType": "Internal",
        "description": str(exception),
        "fileName": filename,
        "functionName": function_name,
        "moreInfo":moreinfo
    }

    if _id!="":
        log_err["actionRecordID"] = ObjectId(_id)
    
    insert_error_log(log_err)

def insert_error_log(log_err):
    r=Rpc()
    error_logging_route="/internal/v1/system/log_error"
    r.post(url=error_logging_route, data=log_err)
    print("An exception occurred. Logging the exception information")