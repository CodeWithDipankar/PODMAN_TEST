import pymssql 
import datetime
import traceback 
from auth import create_Dynamo_Client_Prod

dynamo_client = create_Dynamo_Client_Prod()
table_name = 'core_as'
key_name = 'job'

def getDBConnection():
    conn = pymssql.connect("AWSUS2-SQL02", "Coreappuser","anbF6Pug38Z4$3q", "MP_Core_Central")
    return conn

def updateModelDBRunTime(model_id, model_start_time, model_run_time):
    conn = getDBConnection()
    cursor = conn.cursor()
    try:
        cursor.execute(f"update models set ModelStartTime = '{str(model_start_time)}', ModelRunTime = {model_run_time} where id = '{model_id}'")#ModelStartTime = {model_start_time},
        conn.commit()
        print(f"{model_id} updated...")
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
    
def updateModelDBSplitTime(process, model_id, runtime = 0):
    status = -1 if runtime ==-1 else 1
    conn = getDBConnection()
    cursor = conn.cursor()
    if process == "splitStartTIme":
        splitStartTime = datetime.datetime.utcnow()
        cursor.execute(f"update models set SplitStartTime = '{splitStartTime}' where id = '{model_id}'")
        conn.commit()
        print(f"{model_id} updated with SplitStartTime: {splitStartTime}")
    elif process == "splitRunTIme":
        cursor.execute(f"update models set status = {status}, SplitRunTime = {runtime} where id = '{model_id}'")
        conn.commit()
        print(f"{model_id} updated with SplitRunTime: {runtime}") 
    elif process == "modelSubmitError":
        cursor.execute(f"update models set status = -1 where id = '{model_id}'")
        conn.commit()
        print(f"{model_id} updated with status: -1") 

def updateStatusModelDB(model_id, status):
    conn = getDBConnection()
    cursor = conn.cursor()
    cursor.execute(f"update models set status = {status} where id = '{model_id}'")
    conn.commit()
    if int(status) == 0:
        updateJob(model_id, 'Done', True)
    else:
        updateJob(model_id, 'Falied', False)    
    print(f"{model_id} updated with status {status}")

def updateADSDB(ads_id, count=0):
    conn = getDBConnection()
    cursor = conn.cursor()
    isSplitted = -1 if count == -1 else 0
    cursor.execute(f"update ads set StoreCount = {count}, isSplitted = {isSplitted} where id = '{ads_id}'")    
    conn.commit() 
    print(f"{ads_id} updated with store count {count}")   

# def incrementStore(key, dyn_count = 0):
def incrementStore(key):
    # incr = str(1) if dyn_count == 0 else str(dyn_count+1)
    try:
        response = dynamo_client.update_item(
            TableName=table_name,
            Key={
                key_name:{
                    'S':key
                }
            },
            UpdateExpression="set done_store_count = done_store_count + :val",
            ExpressionAttributeValues={
            ':val': {'N':'1'}
            },
            # ReturnValues="UPDATED_NEW"
        )
        # return 0
    except Exception as e:
        print('key',key)
        print(e)
        # return dyn_count + 1  
        # 
def incrementSplit(key):
    try:
        response = dynamo_client.update_item(
            TableName=table_name,
            Key={
                key_name:{
                    'S':key
                }
            },
            UpdateExpression="set split_store_count = split_store_count + :val",
            ExpressionAttributeValues={
            ':val': {'N':'1'}
            },
            # ReturnValues="UPDATED_NEW"
        )
        # return 0  
    except Exception as e:
        print('key',key)
        print(e)
        # return dyn_count + 1

def updateJob(key, progress = "RUNNING", job_status = False):
      response = dynamo_client.update_item(
        TableName=table_name,
        Key={
            key_name:{
                'S':key
            }
        },
        UpdateExpression="set progress = :val1, job_status = :val2, end_time = :val3 ",
        ExpressionAttributeValues={
        ':val1': {'S':progress},
        ':val2': {'BOOL':job_status},
        ':val3': {'S':str(datetime.datetime.utcnow())}
        },
        ReturnValues="UPDATED_NEW"
    )

def updateJobStore(key, store = 0):
      response = dynamo_client.update_item(
        TableName=table_name,
        Key={
            key_name:{
                'S':key
            }
        },
        UpdateExpression="set total_store_count = :val1",
        ExpressionAttributeValues={
        ':val1': {'N':str(store)}
        },
        ReturnValues="UPDATED_NEW"
    )

def insertEntry(key, store):
    response = dynamo_client.put_item(
        TableName=table_name,
        Item={
            key_name:{
                'S':key
            },
            'total_store_count':{
                'N':str(store)
            },
            'split_store_count':{
                'N':'0'
            },
            'done_store_count':{
                'N':'0'
            },
            'progress':{
               'S': "SPLIT",
            },
            'job_status':{
               'BOOL': False,
            },
            'start_time':{
                'S':str(datetime.datetime.utcnow())
            },
            'end_time':{
                'S':'NA'
            }
        },
    )