import boto3
import os

def set_creds_path_env():
    os.environ['AWS_SHARED_CREDENTIALS_FILE'] = "C:\\creds\\aws-profile"

def create_S3_Client_POC():
    set_creds_path_env()
    session = boto3.Session(profile_name="core-profile")
    return session.client('s3', verify=False)

def create_Batch_Client_POC():
    set_creds_path_env()
    session = boto3.Session(profile_name="core-profile")
    return session.client('batch', verify=False)
        
#=======================================================================
#For Docker Image pilot
def create_S3_Client_Image():
    access_key = os.environ['AWS_ACCESS_KEY_ID']
    secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
    region = os.environ['REGION']
    session = boto3.Session(aws_access_key_id = access_key,
                            aws_secret_access_key = secret_key,
                            region_name= region)
    return session.client('s3',verify=False)

def create_Batch_Client_Image():
    access_key = os.environ['AWS_ACCESS_KEY_ID']
    secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
    region = os.environ['REGION']
    session = boto3.Session(aws_access_key_id = access_key,
                            aws_secret_access_key = secret_key,
                            region_name= region)
    return session.client('batch', verify=False)

def create_S3_Resource_Image():
    access_key = os.environ['AWS_ACCESS_KEY_ID']
    secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
    region = os.environ['REGION']
    session = boto3.Session(aws_access_key_id = access_key,
                            aws_secret_access_key = secret_key,
                            region_name= region)
    return session.resource('s3', verify=False)

#==================PROD=================================================

def create_S3_Client_Image_Prod():
    session = boto3.Session()
    return session.client('s3',verify=False)

def create_Batch_Client_Image_Prod():
    session = boto3.Session()
    return session.client('batch', verify=False)

def create_S3_Resource_Image_Prod():
    session = boto3.Session()
    return session.resource('s3', verify=False)
def create_Dynamo_Client_Prod():
    session = boto3.Session()
    return session.client('dynamodb', region_name="us-east-1")

#==================PROD=================================================

def create_S3_Client_Image_Profile():
    session = boto3.Session()
    return session.client('s3',verify=False)

def create_Batch_Client_Image_Profile():
    session = boto3.Session()
    return session.client('batch', verify=False)

def create_S3_Resource_Image_Profile():
    session = boto3.Session()
    return session.resource('s3', verify=False)
def create_Dynamo_Client_Profile():
    session = boto3.Session()
    return session.resource('dynamodb', verify=False)