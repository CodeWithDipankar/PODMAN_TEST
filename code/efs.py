import time
import os
import traceback
from constant import ADS,SPEC,SPLIT,ADS_PATH,SPEC_PATH,EFS_MOUNT_DIR,EFS_WORKING_DIR,EFS_ADS_DIR,EFS_SPEC_DIR,EFS_SPLIT_DIR, UNIQUE_ID, GEOGRAPHY_IDENTIFIER, EFS_OUTPUT_DIR, OUTPUT, ADS_FILE_NAME
import shutil
import re
# from db import updateADSDB, incrementSplit, insertEntry, updateJob, updateModelDB, updateJobStore
from auth import create_S3_Client_Image_Prod

s3_client = create_S3_Client_Image_Prod()

IS_From_Batch = True 
s3_bucket = "immap-app"

def file_splitter_count(ads_file_fullpath, geo):
    mainFile = open(ads_file_fullpath, 'r')
    header = mainFile.readline()
    headerList = header.split(",")
    geoIndex = headerList.index(geo)

    firstLine = mainFile.readline()
    firstLineSplit = firstLine.split(",")
    knownGeo = firstLineSplit[geoIndex]
    # file_name = re.sub("[*:,./<>?\|]", "_", knownGeo)
    # fw = open(input_path + '\\' + file_name + '.csv', 'w')
    # fw.write(header + firstLine)

    knownGeo = knownGeo + ','
    fileS = ""
    count = 0
    no_of_lines = False
    geo_count = 0
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
            geo_count+=1
            break
        else:
            if(not no_of_lines): no_of_lines = count # Set this one time
            if(count != no_of_lines): raise Exception("100: Error in Split - " + knownGeo + " current store / 1st store row count - " + str(count) + "/" + str(no_of_lines))
            # fw.write(fileS)
            geo_count+=1
            count = 0
            # fw.close()

            nextLineSplit = nextLine.split(",")
            knownGeo = nextLineSplit[geoIndex]
            # file_name = re.sub("[*:,./<>?\|]", "_", knownGeo)
            # fw = open(input_path + '\\' + file_name + '.csv', 'w')
            knownGeo = knownGeo + ','
            fileS = header + nextLine

    mainFile.close()
    return geo_count

def file_splitter(ads_file_fullpath, input_path, geo, model_id):
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
            incrementSplit(model_id)
            fw.write(fileS)
            fw.close()
            break
        else:
            if(not no_of_lines): no_of_lines = count # Set this one time
            if(count != no_of_lines): raise Exception("100: Error in Split - " + knownGeo + " current store / 1st store row count - " + str(count) + "/" + str(no_of_lines))
            incrementSplit(model_id)
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

def create_folder_efs(folder):
    try:
        print('Creating folder ' + folder)
        os.makedirs(folder)
    except Exception as e:
        print('failed to create directory')
        print(e)

def prepare_working_dirs(config, is_model = False):
    success= False
    try:
        # VERY IMPORTANT. IF YOU DON'T HAVE A UNIQUE IDENTIFIER HERE, IT WILL BREAK WHEN ANOTHER JOB ACCESS THE EFS
        # here you will pass your batch jobs unique id and create a subdirect under the efs mount dir
        # create_folder_efs(config["efs_mount_dir"])
        config[EFS_WORKING_DIR] = os.path.join(config[EFS_MOUNT_DIR], config[UNIQUE_ID]) 
        config[EFS_ADS_DIR] = os.path.join(config[EFS_WORKING_DIR], ADS)
        # config[EFS_SPLIT_DIR] = os.path.join(config[ADS_FILE_NAME], SPLIT + "_" + config[ADS_FILE_NAME])
        config[EFS_SPLIT_DIR] = os.path.join(config[EFS_WORKING_DIR], SPLIT)
        if is_model:
            config[EFS_SPEC_DIR] = os.path.join(config[EFS_WORKING_DIR], SPEC)
            config[EFS_OUTPUT_DIR] = os.path.join(config[EFS_WORKING_DIR], OUTPUT)

        print('updated config')
        print(config)
        create_folder_efs(config[EFS_WORKING_DIR])
        create_folder_efs(config[EFS_ADS_DIR])
        create_folder_efs(config[EFS_SPLIT_DIR])
        if is_model:
            create_folder_efs(config[EFS_SPEC_DIR])
            create_folder_efs(config[EFS_OUTPUT_DIR])

        success = True
    except Exception as e:
        print(e)
    return success 

def get_s3_file_parts(s3_path):
    path_parts = s3_path.replace("s3://","").replace("S3://","").split("/")
    # bucket = path_parts.pop(0) # .net will send only the key 
    key = "/".join(path_parts)
    filename = path_parts[-1]
    return key, filename

def download_s3_file(s3_source_path, efs_destination):
    start = time.perf_counter()
    bucket = s3_bucket
    key, filename = get_s3_file_parts(s3_source_path)
    status = False
    final_filename = os.path.join(efs_destination, filename)
    
    try:
        print(bucket,key,filename)
        s3_client.download_file(bucket, key, final_filename)
        status = True
    except Exception as e:
        print("Download failed")
        print(e)
    end = time.perf_counter() - start
    print('downloading file {} took {} seconds'.format(final_filename,round(end,2)))
    return {"success": status, "fq_filename": final_filename}

def log_efs_path(working_dir):
    print('display start')
    dirs = []
    start = time.perf_counter()
    try:
        for path, subdirs, files in os.walk(working_dir,topdown=True):
            for name in files:
                # print(os.path.join(path,name))
                dirs.append(os.path.join(path,name))
    except Exception:
         print(traceback.format_exc())
    finally:
        print("files in EFS")
        print(len(dirs))
        print(f"Display took {round(time.perf_counter() - start,2)} seconds")
    print('display end') 

def cleanup(directory):
    print('clean up ' + directory)
    start = time.perf_counter()
    try:
        shutil.rmtree(directory,ignore_errors=True)
    except Exception as e:
        print(traceback.format_exc())
    finally:
        print(f" cleanup took {round(time.perf_counter() - start,2)} seconds")

def startEFSProcess(config):
    try:
        config = {
            ADS_PATH: config["ads_path"],#"s3://immap-usr/client/ahold/core-as/test_ads_upload/v1.csv",
            EFS_MOUNT_DIR: "/mount/efs",
            UNIQUE_ID: config["unique_id"],
            # STORE_COUNT: config['store_count'],
            GEOGRAPHY_IDENTIFIER: config['geo_col_name']
        }
        startTime = time.time()
        updateModelDB("splitStartTIme", config[UNIQUE_ID])
        #DYNAMO ENTRY
        
        insertEntry(config[UNIQUE_ID], 0)

        if(not prepare_working_dirs(config)):
            raise Exception("Failed to create efs directories, exiting.")


        efs_ads = download_s3_file(config[ADS_PATH], config[EFS_ADS_DIR])

        file_splitter(efs_ads["fq_filename"], config[EFS_SPLIT_DIR], config[GEOGRAPHY_IDENTIFIER], config[UNIQUE_ID])
        runtime = str(time.time() - startTime)
        store_count = len(os.listdir(config[EFS_SPLIT_DIR]))
        print("split count: ", store_count)
        print("split done, runtime: ", runtime, config[UNIQUE_ID])
        updateModelDB("splitRunTIme", config[UNIQUE_ID], runtime)
        updateJob(config[UNIQUE_ID])
        updateJobStore(config[UNIQUE_ID], store_count)
        print("Dynamo status updated to RUNNING...") 
        return store_count
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        updateJob(config[UNIQUE_ID], "FAILED")
        updateModelDB("splitRunTIme", config[UNIQUE_ID], -1)
        exit(2)
    finally:
        print("I am in finally block ads split (model run )and not cleaning the splits")
        # cleanup(config[EFS_WORKING_DIR])
        # log_efs_path(config[EFS_WORKING_DIR])

def adsSplit(config):
    try:
        config = {
            ADS_PATH: config["ads_path"],#"s3://immap-usr/client/ahold/core-as/test_ads_upload/v1.csv",
            EFS_MOUNT_DIR: "/mount/efs",
            UNIQUE_ID: config["unique_id"],
            GEOGRAPHY_IDENTIFIER: config['geo_col_name']
        }
        if(not prepare_working_dirs(config)):
            raise Exception("Failed to create efs directories, exiting.")


        efs_ads = download_s3_file(config[ADS_PATH], config[EFS_ADS_DIR])

        store_count = file_splitter_count(efs_ads["fq_filename"], config[GEOGRAPHY_IDENTIFIER])
        # store_count = len(os.listdir(config[EFS_SPLIT_DIR]))
        print("split count: ", store_count)
        updateADSDB(config[UNIQUE_ID], store_count)
        print("Done")
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        updateADSDB(config[UNIQUE_ID], -1)
        #set some fail flag, to dynamo
    finally:
        print("I am in ADS SPLIT finally block and cleaning up the splits")
        cleanup(config[EFS_WORKING_DIR])
        log_efs_path(config[EFS_WORKING_DIR])

def cleanEFS(config):
    config[EFS_MOUNT_DIR] = "/mount/efs"
    efs_root_dir_count = len(os.listdir(config[EFS_MOUNT_DIR]))
    print("EFS root DIR file count before cleaning: ", efs_root_dir_count)
    try:
        if efs_root_dir_count > 0:
            print("Cleaning up...")
            cleanup(config[EFS_MOUNT_DIR])
            efs_root_dir_count = len(os.listdir(config[EFS_MOUNT_DIR]))
            print("EFS root DIR file count after cleaning: ", efs_root_dir_count)
    except Exception as e:
        print(e)
        print(traceback.format_exc())    
    finally:
        efs_root_dir_count = len(os.listdir(config[EFS_MOUNT_DIR]))
        print("EFS root DIR file count finally: ", efs_root_dir_count)
