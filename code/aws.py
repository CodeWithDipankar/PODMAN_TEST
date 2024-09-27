import os
import traceback
import codecs
import re
from io import StringIO
import math
import multiprocessing as mp
from auth import create_S3_Client_Image, create_S3_Client_Image_Prod, create_S3_Client_POC, create_S3_Resource_Image, create_S3_Resource_Image_Prod

Is_Prod = True
S3_BUCKET_NAME = "immap-app"
client = create_S3_Client_Image() if not Is_Prod else create_S3_Client_Image_Prod() 
s3_resource = create_S3_Resource_Image() if not Is_Prod else create_S3_Resource_Image_Prod()
#======================================GENERIC====================================================================
def aws_communicator(info, status_path):
    try:
        client.put_object(Body=info, Bucket=S3_BUCKET_NAME, Key=status_path)
    except Exception as e: 
        msg = "Aws-ERROR|" + str(e) + "\n\nDev Details\n===========\n" + traceback.format_exc()
        print("aws_communicator Exception: ", str(msg))

def create_s3_folder(folder_path):
    try:
        client.put_object(Bucket = S3_BUCKET_NAME, Key = folder_path)
    except Exception as e: 
        msg = "Aws-ERROR|" + str(e) + "\n\nDev Details\n===========\n" + traceback.format_exc()
        print("create_s3_folder Exception: ", str(msg))
        # aws_communicator(msg, status_path)

def is_s3_folder_exists(folder_path):
    try:
        client.head_object(Bucket = S3_BUCKET_NAME, Key = folder_path)
        return True
    except Exception as e: 
        return False

def get_all_files_s3Folder(folder_path):
    s3_files_response = client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=folder_path)
    return [f['Key'].split('/')[len(f['Key'].split('/'))-1] for f in s3_files_response['Contents'] if '.csv' in f['Key'].split('/')[len(f['Key'].split('/'))-1]] #returing only .csv's

def get_all_files_s3Folder_paginator(folder_path):
    paginator = client.get_paginator('list_objects_v2')

    main_response = paginator.paginate(
        Bucket=S3_BUCKET_NAME,
        Delimiter='/',
        Prefix=folder_path,
    )
    return [f['Key'].split('/')[len(f['Key'].split('/'))-1] for s3_files_response in main_response for f in s3_files_response['Contents'] if '.csv' in f['Key'].split('/')[len(f['Key'].split('/'))-1]]
    # for page in response_iterator:
    #     for object in page['Contents']:
    #         print(object['Key'])

def get_all_json_files_s3Folder(folder_path):
    s3_files_response = client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=folder_path)
    return [f['Key'].split('/')[len(f['Key'].split('/'))-1] for f in s3_files_response['Contents'] if '.json' in f['Key'].split('/')[len(f['Key'].split('/'))-1]] #returing only .json's

def aws_path_creator(output_path, input_path, granular_result, granular_stats, granular_support, create_output, rand_mapping_dict = {}):
    split = True
    print('aws_path_creator: ', output_path, input_path, granular_result, granular_stats, granular_support, create_output)
    # exit()
    if create_output:
        is_folder_exist = is_s3_folder_exists(output_path)
        if not is_folder_exist:
            create_s3_folder(output_path + '/') 
            pass
    # try:
    #     is_folder_exist = is_s3_folder_exists(input_path +'/')
    #     if not is_folder_exist:
    #         print("split not exist")
    #         create_s3_folder(input_path + '/')
    #         split = True
    #     else: 
    #         s3_files = get_all_files_s3Folder(input_path + '/')  
    #         s3_files = [f for f in s3_files if '.csv' in f]
    #         split = True if len(s3_files) == 0 else False
    # except:
        # files = [f for f in os.listdir(input_path) if isfile(join(input_path, f))]
        # print(files)
        # split = True if len(files) == 0 else False
        # pass
    if granular_stats:
        try:
            create_s3_folder(output_path + '/stats/')
            if len(rand_mapping_dict) > 0:
                try:
                    create_s3_folder(output_path + '/randomization_stats/')
                    for key in rand_mapping_dict.keys():
                        create_s3_folder(output_path + '/randomization_stats/' + key + '_stats/')
                except:
                    pass
        except:
            pass
    if granular_result:
        try:
            create_s3_folder(output_path + '/granular_result/')
            if len(rand_mapping_dict) > 0:
                try:
                    create_s3_folder(output_path + '/randomization_result/')
                    for key in rand_mapping_dict.keys():
                        create_s3_folder(output_path + '/randomization_result/' + key + '_result/')
                except:
                    pass
        except:
            pass
    if granular_support:
        try:
            create_s3_folder(output_path + '/granular_support/')
        except:
            pass
    return split

def aws_delete_files(folder):
    response = client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=folder)
    for object in response['Contents']:
        try:
            if 'transformstions.csv' not in object['Key']:
                print('Deleting', object['Key'])
                client.delete_object(Bucket=S3_BUCKET_NAME, Key=object['Key'])
        except Exception as e:
            print('ERROR|Failed to delete %s. Reason: %s' % (object['Key'], e))
    
def read_s3_file(file_path):
    s3_object = client.get_object(Bucket = S3_BUCKET_NAME, Key = file_path)
    return s3_object['Body']

def read_s3_file_resource(file_path):
    s3_bucket = s3_resource.Bucket(S3_BUCKET_NAME)
    s3_object = s3_bucket.Object(file_path).get()
    return s3_object['Body']
    
def read_s3_file_stream(file_path):
    s3_object = client.get_object(Bucket = S3_BUCKET_NAME, Key = file_path)
    line_stream = codecs.getreader("utf-8")
    return line_stream(s3_object['Body'])   

def write_file_stream_s3(file_path, content):
    try:
        client.put_object(Bucket=S3_BUCKET_NAME, Key=file_path, Body=content)
    except Exception as e: 
        msg = "Aws-ERROR|" + str(e) + "\n\nDev Details\n===========\n" + traceback.format_exc()
        print("write_file_stream_s3 Exception: ", str(msg))

def write_csv_to_s3(file_path, df):
    try:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        client.put_object(Bucket=S3_BUCKET_NAME, Key=file_path, Body=csv_buffer.getvalue())    
    except Exception as e: 
        msg = "Aws-ERROR|" + str(e) + "\n\nDev Details\n===========\n" + traceback.format_exc()
        print("write_csv_to_s3 Exception: ", str(msg))    

def append_file_stream_s3(file_path, content):
    try:
        appended_data = read_s3_file_stream(file_path).read() + content
        client.put_object(Bucket=S3_BUCKET_NAME, Key=file_path, Body=appended_data)
    except Exception as e: 
        msg = "Aws-ERROR|" + str(e) + "\n\nDev Details\n===========\n" + traceback.format_exc()
        print("append_file_stream_s3 Exception: ", str(msg))

def aws_file_splitter(ads_file_fullpath, input_path, geo):
    mainFile = read_s3_file_stream(ads_file_fullpath)#open(ads_file_fullpath, 'r')
    header = mainFile.readline()
    headerList = header.split(",")
    if geo in headerList:
        geoIndex = headerList.index(geo)
        firstLine = mainFile.readline()
        firstLineSplit = firstLine.split(",")
        knownGeo = firstLineSplit[geoIndex]
        file_name = re.sub("[*:,./<>?\|]", "_", knownGeo)
        write_file_stream_s3(input_path + '/' + file_name + '.csv', header+firstLine)#open(input_path + '/' + file_name + '.csv', 'w')
        # fw.write(header + firstLine)
        knownGeo = knownGeo + ','
        fileS = ""
        count = 0
        no_of_lines = False
        while True:
            nextLine = mainFile.readline()
            if(knownGeo in nextLine):
                fileS += nextLine
                count += 1
            elif not nextLine:
                if(not no_of_lines): no_of_lines = count # Set this one time, if 1st file = last file
                if(count != no_of_lines): raise Exception("100: Error in Split - " + knownGeo + " current store / 1st store row count - " + str(count) + "/" + str(no_of_lines))
                # fw.write(fileS)
                # fw.close()
                append_file_stream_s3(input_path + '/' + file_name + '.csv', fileS)
                break
            else:
                if(not no_of_lines): no_of_lines = count # Set this one time
                if(count != no_of_lines): raise Exception("100: Error in Split - " + knownGeo + " current store / 1st store row count - " + str(count) + "/" + str(no_of_lines))
                # fw.write(fileS)
                append_file_stream_s3(input_path + '/' + file_name + '.csv', fileS)
                count = 0
                # fw.close()

                nextLineSplit = nextLine.split(",")
                knownGeo = nextLineSplit[geoIndex]
                file_name = re.sub("[*:,./<>?\|]", "_", knownGeo)
                write_file_stream_s3(input_path + '/' + file_name + '.csv', '')# fw = open(input_path + '/' + file_name + '.csv', 'w')
                knownGeo = knownGeo + ','
                fileS = header + nextLine

        mainFile.close() 
        return 0   
    else:
        return -1  

#==========================================ADS COPY===================================================================           
def get_range(param):
    job_index = int(os.environ['AWS_BATCH_JOB_ARRAY_INDEX'])
    store_count = int(param["store_count"])
    chunks = int(param["chunks"])
    stores_per_job = math.ceil(store_count/chunks)
    print("stores_per_job: ", stores_per_job)
    print("job_index: ", job_index)
    first_idx = job_index * stores_per_job
    print("first_idx: ", first_idx)
    last_idx = int(job_index * stores_per_job) + int(stores_per_job) if job_index != chunks-1 else store_count
    split_range = str(first_idx) + '|' + str(last_idx) 
    return split_range  

def files_to_core_split_batch(files, cpu):
    first_idx, last_idx = int(cpu.split('|')[0]), int(cpu.split('|')[1])  
    print('files_to_core_split_batch, first_idx, last_idx: ', first_idx, last_idx)
    split_files = files[first_idx: last_idx]
    return split_files

def files_to_core_split(files, cpu):
    splitBy = int(len(files) / cpu)
    splitList = [files[i:i + splitBy] for i in range(0, len(files), splitBy)]
    if len(splitList) > cpu:
        extra = splitList.pop()
        for i in range(len(extra)): # distribute extra from begining
            splitList[i].append(extra[i])
    return splitList

def uploadDirectory(_files, path, s3_path):
    print("_files: ", len(_files))
    for file in _files:
        # copy_source = {'Bucket': S3_BUCKET_NAME, 'Key': path+file}
        # client.copy(copy_source, S3_BUCKET_NAME, s3_path+file)  
        response = client.copy_object(
            CopySource=S3_BUCKET_NAME+'/'+path+file,  # /Bucket-name/path/filename
            Bucket=S3_BUCKET_NAME,                       # Destination bucket
            Key=s3_path+file                    # Destination path/filename
        )
        print("copy response: ", response)
    print("Done")

def copy_ads_split(param):
    FILE_PATH = param["input_path"]#src
    KEY_PATH = param["output_path"]#dest
    cpu_count = int(param["cpu_count"])#for mp
    procs = []
    cpu_count = 3 if cpu_count == 0 else cpu_count
    files = get_all_files_s3Folder_paginator(FILE_PATH) 
    
    split_range = get_range(param) 
    names_l = files_to_core_split_batch(files, split_range)
    names = files_to_core_split(names_l, cpu_count)
    print("MP started")
    # instantiating process with arguments
    for name in names:
        print("Name: ",name)
        proc = mp.Process(target=uploadDirectory, args=(name, FILE_PATH, KEY_PATH))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()
    print("Completed")