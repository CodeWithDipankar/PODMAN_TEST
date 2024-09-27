from os import path
from os.path import join, isfile
import os
import shutil
import re
import time
from aws import aws_path_creator, aws_file_splitter, get_all_files_s3Folder_paginator, aws_communicator
import s3fs
from auth import create_S3_Client_Image_Prod
from db import updateJob, updateJobStore, incrementSplit
import concurrent.futures

s3_client = create_S3_Client_Image_Prod()

# IS_AWS_BATCH = False
IS_RUN_ON_BATCH = bool(os.environ["aws"]) if "aws" in os.environ else False

def get_paths(ads_path, output_path, spec_path):
    ads_folder = path.split(ads_path)[0]
    ads_filename = path.split(ads_path)[1].split(".")[0]
    split_ads_path = 'split_' + ads_filename
    input_path = path.join(ads_folder, split_ads_path)
    status_path = path.join(output_path, 'status.txt')
    spec_folder = path.split(spec_path)[0]
    spec_filename = path.split(spec_path)[1].split(".")[0]
    spec_jsonpath = path.join(spec_folder, spec_filename + ".json")
    geolevel_spec = path.join(spec_folder, spec_filename + "_" + 'geolevel_spec.xlsx')
    return input_path, status_path, geolevel_spec, spec_jsonpath

def delete_files(folder):
    print("DELETE|")
    files = os.listdir(folder)
    if "transformstions.csv" in files:
        files.remove("transformstions.csv")
    for filename in files:
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('ERROR|Failed to delete %s. Reason: %s' % (file_path, e))

def file_splitter(ads_file_fullpath, input_path, geo):
    mainFile = open(ads_file_fullpath, 'r')
    header = mainFile.readline()
    headerList = header.split(",")
    geoIndex = headerList.index(geo)

    firstLine = mainFile.readline()
    firstLineSplit = firstLine.split(",")
    knownGeo = firstLineSplit[geoIndex]
    file_name = re.sub("[*:,./<>?\|]", "_", knownGeo)
    fw = open(input_path + '/' + file_name + '.csv', 'w')
    fw.write(header + firstLine)

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
            if IS_RUN_ON_BATCH:
                incrementSplit(str(os.environ["MODEL_ID"]))
            fw.write(fileS)
            fw.close()
            break
        else:
            if(not no_of_lines): no_of_lines = count # Set this one time
            if(count != no_of_lines): raise Exception("100: Error in Split - " + knownGeo + " current store / 1st store row count - " + str(count) + "/" + str(no_of_lines))
            if IS_RUN_ON_BATCH:
                incrementSplit(str(os.environ["MODEL_ID"]))
            fw.write(fileS)
            count = 0
            fw.close()

            nextLineSplit = nextLine.split(",")
            knownGeo = nextLineSplit[geoIndex]
            file_name = re.sub("[*:,./<>?\|]", "_", knownGeo)
            fw = open(input_path + '/' + file_name + '.csv', 'w')
            knownGeo = knownGeo + ','
            fileS = header + nextLine

    mainFile.close()

def ads_file_splitter(ads_file_fullpath, input_path, geo):
    file_splitter(ads_file_fullpath, input_path, geo)
    if IS_RUN_ON_BATCH:
        store_count = len(os.listdir(input_path))
        updateJob(str(os.environ["MODEL_ID"]))
        updateJobStore(str(os.environ["MODEL_ID"]), store_count)


def path_creator(output_path, input_path, granular_result, granular_stats, granular_support, create_output, rand_mapping_dict = {}):
    split = True
    if create_output:
        try:
            os.mkdir(output_path)
        except:
            delete_files(output_path) # Delete all files.
            pass
    try:
        os.mkdir(input_path)
        files = [f for f in os.listdir(input_path) if isfile(join(input_path, f))]
        split = True if len(files) == 0 else False
    except:
        files = [f for f in os.listdir(input_path) if isfile(join(input_path, f))]
        split = True if len(files) == 0 else False
        pass
    if granular_stats:
        try:
            os.mkdir(output_path + '/stats')
            if len(rand_mapping_dict) > 0:
                try:
                    os.mkdir(output_path + '/randomization_stats')
                    for key in rand_mapping_dict.keys():
                        os.mkdir(output_path + '/randomization_stats/' + key + '_stats')
                except:
                    pass
        except:
            pass
    if granular_result:
        try:
            os.mkdir(output_path + '/granular_result')
            if len(rand_mapping_dict) > 0:
                try:
                    os.mkdir(output_path + '/randomization_result')
                    for key in rand_mapping_dict.keys():
                        os.mkdir(output_path + '/randomization_result/' + key + '_result')
                except:
                    pass  
        except:
            pass
    if granular_support:
        try:
            os.mkdir(output_path + '/granular_support')
            os.mkdir(output_path + '/granular_support_no_postmul')
        except:
            pass
    try:
        os.mkdir(os.path.join(output_path,'agg'))
    except:
        pass
    return split

def output_path_creator(output_path, input_path, granular_result, granular_stats, granular_support, create_output, rand_mapping_dict = {}):
    return path_creator(output_path, input_path, granular_result, granular_stats, granular_support, create_output, rand_mapping_dict)

def communicator(info, status_path):
    print(info)
    fw = open(status_path, 'w')
    fw.write(info)
    fw.close()

def get_all_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def copy_dir_to_s3_s3fs(source, dest):
    print("Copying...")
    print("Source, dest: ", source, dest)
    s3_file = s3fs.S3FileSystem()
    local_path = source#"some_dir_path/some_dir_path/"
    s3_path = dest#"bucket_name/dir_path"
    s3_file.put(local_path, s3_path, recursive=True) 
    print("copied")


def copy_dir_to_s3(s3bucket, input_dir, s3_path):
    startTime = time.time()
    print("s3bucket, input_dir, s3_path: ", s3bucket, input_dir, s3_path)
    print("Copying...")
    for path, subdirs, files in os.walk(input_dir):
        for file in files:
            dest_path = path.replace(input_dir, "").replace(os.sep, '/')
            s3_file = f'{s3_path}/{dest_path}/{file}'.replace('//', '/')
            local_file = os.path.join(path, file)
            s3_client.upload_file(local_file, s3bucket, s3_file)
    print(f"Successfully uploaded {input_dir} to S3 {s3_path}")
    print(f"Copy took {str(time.time()-startTime)} secs to complete")

def upload_file(S3_BUCKET, input_path, local_file, s3_path):
    dest_path = os.path.dirname(local_file).replace(input_path, "").replace(os.sep, '/')
    s3_file = f'{s3_path}/{dest_path}/{os.path.basename(local_file)}'.replace('//', '/')
    with open(local_file, 'rb') as f:
        s3_client.upload_fileobj(f, S3_BUCKET, s3_file)
    # print(f"{os.path.basename(local_file)} Successfully uploaded {input_path} to S3 {s3_path}")


def copy_dir_to_s3_thrd_worker(S3_BUCKET, input_dir, s3_path):
    print("Copying in thrd...")
    with concurrent.futures.ThreadPoolExecutor(max_workers= 15) as executor:
        for path, subdirs, files in os.walk(input_dir):
            for file in files:
                local_file = os.path.join(path, file)
                executor.submit(upload_file, S3_BUCKET, input_dir,  local_file, s3_path)

def copy_dir_to_s3_thrd(S3_BUCKET, input_dir, s3_path):
    start_time = time.time()
    copy_dir_to_s3_thrd_worker(S3_BUCKET, input_dir, s3_path)
    print(f"Copy took {str(time.time()-start_time)} secs")
