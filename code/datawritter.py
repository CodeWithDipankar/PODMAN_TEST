from aws import write_csv_to_s3, write_file_stream_s3
import pickle

IS_AWS_BATCH = False

def write_dataframe_as_csv(dataframe, file):
    if not IS_AWS_BATCH:
        dataframe.to_csv(file, index=False)
    else:
        write_csv_to_s3(file, dataframe)


def write_obj_as_pickle(path, file):
    with open(path, "wb" ) as f:
            pickle.dump(file, f)

def read_obj_from_pickle(path):
    with open(path, "rb" ) as f:
	    return pickle.load(f)
