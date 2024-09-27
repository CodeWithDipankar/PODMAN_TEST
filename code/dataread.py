import simplejson as json
import pandas as pd
import numpy as np
from aws import read_s3_file, read_s3_file_resource

IS_AWS_BATCH = False

def read_spec_json_raw(json_path):
    if not IS_AWS_BATCH:
        with open(json_path) as json_file:
            spec = json.load(json_file)
        return spec
    else:
        result = read_s3_file(json_path)
        spec = json.load(result)

def read_as_dataFrame(file, usecols = [], low_memory = False, memory_map = True, engine = 'c'):
    if IS_AWS_BATCH:
        file = read_s3_file_resource(file)
    df = pd.read_csv(file, usecols=usecols, low_memory=low_memory, memory_map=memory_map, engine=engine)
    return df

def read_csv_as_dataFrame_float(file, usecols = [], low_memory = False, memory_map = True, engine = 'c'):
    df = pd.read_csv(file, usecols=usecols,  dtype=np.float64, low_memory=low_memory, memory_map=memory_map, engine=engine)
    return df