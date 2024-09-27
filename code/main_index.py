from engine import run_model_initiator
from constant import ADS, SPEC, OUTPUT, CPU_COUNT, SAMPLE, USE_STORE, FAST_OPTIMiZE, RESUME, GRANULAR_RESULT, GRANULAR_STATS, GRANULAR_SUPPORT,GRANULAR_CORRELATION, GRANULAR_VOL_AGG, GEOLEVEL_CONSTRAINT, UNIQUE_ID, EFS_MOUNT_DIR, ADS_FILE_NAME, EFS_ADS_DIR, EFS_SPEC_DIR, OUTPUT_DIR, EFS_WORKING_DIR, EFS_OUTPUT_DIR, EFS_SPLIT_DIR, IS_SPLIT_AVAILABLE, EFS_GRANULAR_SPEC_DIR, GRANULAR_SPEC_PATH, DYNAMIC_OPT_ITR
from efs import prepare_working_dirs, download_s3_file, log_efs_path, cleanup
import os
import traceback
import time
import datetime
from utility import copy_dir_to_s3_thrd
from db import insertEntry
import argparse
import json
from aws import is_s3_folder_exists

#Environ vars
if "aws" in os.environ:
    print("os.environ[aws]: ", os.environ["aws"])
else:
    print("aws not found on env")
IS_RUN_ON_BATCH = bool(os.environ["aws"]) if "aws" in os.environ else False
CPU = 6 if not IS_RUN_ON_BATCH else int(os.environ["cpu"]) if "cpu" in os.environ else 1
ARGS = True if IS_RUN_ON_BATCH and "args" in os.environ else False
#=======================================================================================

print("IS_RUN_ON_BATCH, CPU, ARGS: ", IS_RUN_ON_BATCH, CPU, ARGS)
S3_BUCKET = "immap-app"

def get_param():
    if ARGS:
        parser = argparse.ArgumentParser(description='AWS Driver Batch Job Runner')
        parser.add_argument('--model_param', dest='model_param', required=True, type=str, help='model param json')
        args = parser.parse_args()
        print('args: ', args.model_param)
        json_body = json.loads(args.model_param)
        # process = json_body["process"] if 'process' in json_body else ""
        spec_path = json_body["spec_path"] if 'spec_path' in json_body else ""
        ads_path = json_body["ads_path"] if 'ads_path' in json_body else ""
        output_path = json_body["output_path"] if 'output_path' in json_body else ""
        dynamic_opt_itr = json_body["dynamic_opt_itr"] if 'dynamic_opt_itr' in json_body else "200"
        # input_path = json_body["input_path"] if 'input_path' in json_body else "" 
        # file_name = json_body["file_name"] if 'file_name' in json_body else ""
        # cpu_count = json_body["cpu_count"] if 'cpu_count' in json_body else 0
        # is_main = json_body["is_main"] if 'is_main' in json_body else -1
        # store_count = json_body["store_count"] if 'store_count' in json_body else -1
        # chunks = json_body["chunks"] if 'chunks' in json_body else -1
        # geo_col_name = json_body['geo_col_name'] if 'geo_col_name' in json_body else ""
        # isMultiProcess = json_body['isMultiProcess'] if 'isMultiProcess' in json_body else 0 
        # isSplitCopyRequired = json_body["isSplitCopyRequired"] if 'isSplitCopyRequired' in json_body else 0
        model_id = json_body["model_id"] if 'model_id' in json_body else ""
        # is_split_on_efs = json_body["is_split_on_efs"] if 'is_split_on_efs' in json_body else 0
        is_split_available = json_body["is_split_available"] if 'is_split_available' in json_body else False
    else:    
        ads_path = r"/app/input/YTL_1.csv" if not IS_RUN_ON_BATCH else r"client/ahold/core-as/test_ads_upload/200_data/v3.csv"
        spec_path = r"/app/input/YTL_1.json" if not IS_RUN_ON_BATCH else r"client/ahold/core-as/test_ads_upload/200_data/v3.json"
        output_path = r"/app/output" if not IS_RUN_ON_BATCH else r"client/ahold/core-as/test_ads_upload/model_out/v7"
        model_id = "Test_MP_" + str(datetime.datetime.now().strftime('%H:%M:%S'))
        is_split_available = False
        dynamic_opt_itr = 200


    param = {        
        ADS: ads_path,
        SPEC: spec_path,
        OUTPUT: output_path,
        CPU_COUNT: CPU,
        SAMPLE : 0,
        USE_STORE: False,
        FAST_OPTIMiZE: False,
        RESUME: False,
        GRANULAR_RESULT: True,
        GRANULAR_STATS: True,
        GRANULAR_SUPPORT: True,
        GRANULAR_CORRELATION: False,
        GRANULAR_VOL_AGG:False,
        GEOLEVEL_CONSTRAINT:True,
        UNIQUE_ID: model_id,
        IS_SPLIT_AVAILABLE: is_split_available,
        EFS_SPLIT_DIR : '',
        DYNAMIC_OPT_ITR: int(dynamic_opt_itr)
    }
    return param

def file_getter(config, payload):
    payload[ADS] = os.path.join(config[EFS_ADS_DIR], os.path.split(payload[ADS])[1])
    payload[SPEC] = os.path.join(config[EFS_SPEC_DIR], os.path.split(payload[SPEC])[1])
    payload[OUTPUT] = config[EFS_OUTPUT_DIR]
    payload[EFS_SPLIT_DIR] = config[EFS_SPLIT_DIR]
    return payload

def input_manipulator(payload):
    model_id = payload[UNIQUE_ID]#"Test_MP_" + str(datetime.datetime.now().strftime('%H:%M:%S'))
    if "MODEL_ID" not in os.environ:
        os.environ["MODEL_ID"] = model_id
    config = {}
    config[UNIQUE_ID] = model_id
    config[EFS_MOUNT_DIR] = "/mount/efs/"
    if not prepare_working_dirs(config,True):
        raise Exception("Failed to create efs directories, exiting...")  
    
    #This process is done in ADS split engine, hence verifying
    if not os.path.exists(config[EFS_ADS_DIR]):#already done in ads split engine 
        download_s3_file(payload[ADS], config[EFS_ADS_DIR])
    else:
        print("Ads is already present")
    if not os.path.exists(config[EFS_SPLIT_DIR]):#already done in ads split engine 
        print("EFS_SPLIT_DIR not exists")
    else:
        print("EFS_SPLIT_DIR is already present")   
    #========================================================

    download_s3_file(payload[SPEC], config[EFS_SPEC_DIR])

    spec_folder = os.path.split(payload[SPEC])[0]
    spec_filename = os.path.split(payload[SPEC])[1].split(".")[0]
    payload[GRANULAR_SPEC_PATH] = spec_folder + "/" + spec_filename + "_" + 'geolevel_spec.xlsx'
    if is_s3_folder_exists(payload[GRANULAR_SPEC_PATH]):
        download_s3_file(payload[GRANULAR_SPEC_PATH], config[EFS_SPEC_DIR])
    
    return config, file_getter(config, payload)

if __name__ == "__main__":
    payload = get_param()
    model_out_path = payload[OUTPUT]
    if IS_RUN_ON_BATCH:
        config, payload = input_manipulator(payload)
    print("Payload in index: ", payload)
    try:
        start_time = time.time()
        if IS_RUN_ON_BATCH and not payload[IS_SPLIT_AVAILABLE]:
            insertEntry(config[UNIQUE_ID], 0)
        run_model_initiator(payload[ADS], payload[EFS_SPLIT_DIR], payload[SPEC], payload[OUTPUT], payload[CPU_COUNT], payload[SAMPLE], payload[USE_STORE], payload[FAST_OPTIMiZE], payload[RESUME], 
                        payload[GRANULAR_RESULT], payload[GRANULAR_STATS], payload[GRANULAR_SUPPORT], payload[GRANULAR_CORRELATION], payload[GRANULAR_VOL_AGG], payload[GEOLEVEL_CONSTRAINT], payload[DYNAMIC_OPT_ITR])
        print(f"Model completed {str(time.time()-start_time)}")
    except Exception as e:
        print("Index error: ", str(e))
        print("traceback: ", traceback.format_exc())
    finally:
        if IS_RUN_ON_BATCH:
            copy_dir_to_s3_thrd(S3_BUCKET, payload[OUTPUT], model_out_path)
            log_efs_path(config[EFS_WORKING_DIR])
            print(f"I am in the index finally block and cleaning up the unique dir {config[UNIQUE_ID]}")
            cleanup(config[EFS_WORKING_DIR])  
            log_efs_path(config[EFS_WORKING_DIR])  
        else:
            print("IS_RUN_ON_BATCH: ", IS_RUN_ON_BATCH)    
    
    
