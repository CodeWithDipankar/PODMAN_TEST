import os
import shutil
from corelib import combine_array, gen_stochastic_sea, add_sea_and_intercept, rapid_refresh, gen_dummy, split_variables, spec_xlsx_to_json, read_spec_json, overall_stats_gen, granular_writter, prep_optimization, bounds_calc, run_transformations, run_correlation, run_optimization, run_optimization_fast, stats_calc, calc_r2_dw_mape, files_to_core_split, gen_seasonality, periodFinder
import pandas as pd
import numpy as np
import multiprocessing as mp
from os import path
import traceback
import time
from scipy import stats
import copy
import datetime
from dataread import read_as_dataFrame, read_csv_as_dataFrame_float
from utility import get_paths, output_path_creator, communicator, get_all_files
from datawritter import write_dataframe_as_csv, write_obj_as_pickle, read_obj_from_pickle
from utility import ads_file_splitter
from db import incrementStore, updateStatusModelDB, updateModelDBRunTime, updateJob
IS_RUN_ON_BATCH = bool(os.environ["aws"]) if "aws" in os.environ else False
IS_MAIN = False
#pip install openpyxl
# def get_paths(ads_path, output_path, spec_path):
#     ads_folder = path.split(ads_path)[0]
#     ads_filename = path.split(ads_path)[1].split(".")[0]
#     split_ads_path = '\\split_' + ads_filename
#     input_path = ads_folder + split_ads_path
#     status_path = output_path + '\\status.txt'
#     spec_folder = path.split(spec_path)[0]
#     spec_filename = path.split(spec_path)[1].split(".")[0]
#     spec_jsonpath = spec_folder + "\\" + spec_filename + ".json"
#     geolevel_spec = spec_folder + '\\' + spec_filename + "_" + 'geolevel_spec.xlsx'
#     return input_path, status_path, geolevel_spec, spec_jsonpath

def split_gen_template(ads_path, spec_path, output_path, gen_template=True):
    # Derive Paths
    input_path, status_path, geolevel_spec, spec_jsonpath = get_paths(ads_path, output_path, spec_path)
    # Spec Reader
    spec = read_spec_json(spec_jsonpath, status_path)
    
    rand_map_df = pd.DataFrame()
    if spec["config"]["isRandomiziedFilter"] == 1:
        randomization_mapping_path = ads_path.split('.csv')[0] + '_randomize_mapping.csv'
        if not path.exists(randomization_mapping_path):
            df = pd.read_csv(ads_path, usecols=spec["randomized_layers"], low_memory=False, memory_map=True, engine='c')[spec["randomized_layers"]]
            df = df.drop_duplicates(subset = spec["randomized_layers"], ignore_index=True)
            rand_map_df = df

    # Determine if split is required.
    split = True
    try:
        os.mkdir(input_path)
        files = [f for f in os.listdir(input_path) if path.isfile(path.join(input_path, f))]
        split = True if len(files) == 0 else False
    except:
        files = [f for f in os.listdir(input_path) if path.isfile(path.join(input_path, f))]
        split = True if len(files) == 0 else False
        pass
    # Splitter
    if split:
        communicator("SPLIT|", status_path)
        ads_file_splitter(ads_path, input_path, spec["config"]["geo_colname"])
    if gen_template:
        files = [f.split(".csv")[0] for f in os.listdir(input_path) if path.isfile(path.join(input_path, f))]
        geo_map_contri = pd.DataFrame({"GEOGRAPHY": files})
        geo_map_contri["MAP"] = "GLOBAL"
        geo_map_contri["EX_INTERCEPT"] = np.nan
        geo_map_contri["EX_EXDEP"] = np.nan
        # Data from 1st file to get some info and add to spec
        period_s = pd.read_csv(input_path + "/" + files[0] + ".csv", usecols=[spec["config"]["period_colname"]], squeeze=True)
        period_s_dt = pd.to_datetime(period_s)
        ex_sea_df = pd.DataFrame({"All_Periods": period_s_dt.dt.date})
        geo_map_contri["EX_SEA"] =  np.nan
        sample_df = pd.DataFrame(spec["contribution_range"]).T
        sample_df.reset_index(level=0, inplace=True)
        sample_df = sample_df.rename(columns={"index": "Variable", "contri_period": "Contribution", "min": "Min", "max": "Max", "sign": "Sign"}, errors="raise")
        override_df = pd.DataFrame([], columns=['Geography', "Variable", 'Contribution', 'Min', 'Max', 'Sign'])
        map_rand_user_level = pd.DataFrame([], columns=['Level', "Name", 'Map',"EX_INTERCEPT", "EX_EXDEP", "EX_SEA"])
        with pd.ExcelWriter(geolevel_spec) as writer:
            if spec["config"]["isRandomiziedFilter"] != 1:
                geo_map_contri.to_excel(writer, sheet_name='MAP', index=False)
            else:
                geo_map_contri.to_excel(writer, sheet_name='MAP', index=False)
                map_rand_user_level.to_excel(writer, sheet_name='MAP_USER', index=False)
                rand_map_df.to_excel(writer, sheet_name='REFERENCE_MAP', index = False)
            sample_df.to_excel(writer, sheet_name='GLOBAL', index=False)
            ex_sea_df.to_excel(writer, sheet_name='EX_SEA', index=False)
            override_df.to_excel(writer, sheet_name='OVERRIDE', index=False)
            writer.save()
    return True

def select_transformarion(result_q, output_path, input_path, files, spec, process):
    try:
        # Read Spec
        transformed_cols = spec['transformed_cols']
        dependent_name = spec['dependent_name']
        # aggregate variables per process
        transformstion_forgeo = {}
        for _file in files:
            # Read from file and convert to dict of numpy as pandas is slow
            file_name = _file.split(".")[0]
            geo_df = pd.read_csv(input_path + "/" + _file, usecols=spec["all_cols_read"], dtype=np.float64, low_memory=False, memory_map=True, engine='c')[spec["all_cols_read"]].to_numpy().T
            geo_df = { spec["all_cols_read"][i]: geo_df[i] for i in range(len(spec["all_cols_read"]))} # takes care of column arrangement.
            dependent_col = geo_df[dependent_name]
            # Transformation + Correlation
            all_transformations, scale_factor = run_transformations(geo_df, transformed_cols, spec["initial_pval"], spec["config"]["pval_config"])
            selected_transformations, corr = run_correlation(all_transformations, dependent_col, spec['same_check'], spec['transformation_sep'])
            transformstion_forgeo[file_name] = list(selected_transformations.values())
        result_q.put({
            "status": True,
            "transformstion_forgeo": transformstion_forgeo,
        })
    except Exception:
        result_q.put({
            "status": False,
            "message": file_name + " | " + traceback.format_exc(),
        })

def main_pipeline(result_q, output_path, input_path, files, spec, process, dynamic_opt_itr):
    try:
        # Resume and popular transformation Handle
        if spec['is_resume']:
            selected_transformations_allstores = spec['selected_transformations_allstores']
        if spec["config"]['single_transformation']:
            spec['transformed_cols'] = selected_transformations_allstores["POPULAR"]
        # Select optimization method
        optimizer = run_optimization_fast if spec['fast_optimize'] else run_optimization
        # Read Spec
        transformed_cols = spec['transformed_cols']
        dependent_name = spec['dependent_name']
        stats_dict = {}
        # aggregate variables per process
        sum_support_optimized = np.zeros((spec['week_count'],spec['var_count']))
        sum_support_dev_sum = np.zeros(spec['var_count'] - 1 if spec["config"]["include_intercept"] else spec['var_count'])
        sum_residual_SS = 0
        sum_dependent = np.zeros(spec['week_count'])
        sum_coeff = np.zeros(spec['var_count'])
        sum_std_error_col_ss = np.zeros(spec['var_count'])
        sum_transformed_support_mask = np.zeros(spec['var_count'])
        level_stats = {}
        optimizer_report = {}
        transformstion_forgeo = {}
        system_unbound_var = {}
        vol_agg_forgeo = {}

        #randomization vars
        randomization_sum_support_optimized = {}
        randomization_sum_dependent = {}
        randomization_sum_coeff = {}
        randomization_sum_support_dev_sum = {}
        for _file in files:
            if IS_RUN_ON_BATCH:
                incrementStore(str(os.environ["MODEL_ID"]))
            # Read from file and convert to dict of numpy.
            file_name = _file.split(".")[0]
            geo_df = read_csv_as_dataFrame_float(input_path + "/" + _file, usecols=spec["all_cols_read"], low_memory=False, memory_map=True, engine='c')[spec["all_cols_read"]].to_numpy().T
            # pd.read_csv(input_path + "/" + _file, usecols=spec["all_cols_read"], dtype=np.float64, low_memory=False, memory_map=True, engine='c')[spec["all_cols_read"]].to_numpy().T
            geo_df = { spec["all_cols_read"][i]: geo_df[i] for i in range(len(spec["all_cols_read"]))} # takes care of column arrangement.
            dependent_col = geo_df[dependent_name]
            # Split Variables
            geo_df = split_variables(geo_df, spec)
            # Dummies
            for k,v in spec["dummy_data"].items():
                geo_df[k] = v
            # Transformation
            if spec['is_resume'] and not spec["config"]['single_transformation']: # other cases are handled above
                transformed_cols = selected_transformations_allstores[file_name]
            all_transformations, scale_factor = run_transformations(geo_df, transformed_cols, spec["initial_pval"], spec["config"]["pval_config"])
            # Coorelation
            if spec['is_resume']: # Skip It
                selected_transformations = {k.split("__")[0]:k for k in  all_transformations.keys()}
                corr_data = False
            else:
                selected_transformations, corr_data = run_correlation(all_transformations, dependent_col, spec['same_check'], spec['transformation_sep'], debug=spec['granular_correlation'])
            # Optimization
            # Check Empty dependent -> Skip
            if dependent_col.sum() == 0:
                level_stats[file_name + " (Skipped)"] = [-1, -1, -1, 0]
                continue
            transformed_support, trans_support_no_postmul, dependent_col, all_cols, bounds, scale_vector, system_unbound, dependent_agg = prep_optimization(file_name, geo_df, spec, dependent_col, selected_transformations, all_transformations)
            system_unbound_var[file_name] = system_unbound
            coeff, opt_status, message, method = optimizer(transformed_support, dependent_col, all_cols, bounds, scale_vector, dynamic_opt_itr)
            if not opt_status:
                optimizer_report[file_name] = [message, method]
            # Basic Stats Calc
            support_optimized = coeff * transformed_support
            predicted = support_optimized.sum(axis=1)
            # STOCHASTIC SEASONALITY 
            if spec["config"]["stochastic_sea"] != 0:
                stocastic_col = gen_stochastic_sea(dependent_col, predicted, spec, spec["config"]["stochastic_sea"])
                transformed_support = np.c_[transformed_support, stocastic_col]
                all_cols.append("STO_SEA")
                scale_vector = np.append(scale_vector, 0)
                bounds.append([-np.inf, np.inf])
                coeff, opt_status, message, method = optimizer(transformed_support, dependent_col, all_cols, bounds, scale_vector)
                support_optimized = coeff * transformed_support
                predicted = support_optimized.sum(axis=1)
            r2, dw, mape, sumResidualSq, dep_agg = calc_r2_dw_mape(dependent_col, predicted)
            # Aggregate Per Process + Granular stats
            sum_dependent = sum_dependent + dependent_col
            sum_support_optimized = sum_support_optimized + support_optimized

            #randomaization calc
            # sum_support_optimized_Level = np.zeros((spec['week_count'],spec['var_count']))
            # sum_dependent_level = np.zeros(spec['week_count'])
            if spec["config"]['isRandomiziedFilter'] == 1:
                for (key, value) in spec['rand_mapping_dict'].items():
                    geo_name = _file.split('.')[0]
                    if geo_name in value.keys():
                        # sum_support_optimized_Level = sum_support_optimized_Level + support_optimized
                        # sum_dependent_level = sum_dependent_level + dependent_col
                        if key+'|'+value[geo_name] not in randomization_sum_support_optimized:
                            randomization_sum_support_optimized[key+'|'+value[geo_name]] = support_optimized
                            randomization_sum_dependent[key+'|'+value[geo_name]] = dependent_col
                        else:
                            randomization_sum_support_optimized[key+'|'+value[geo_name]] = randomization_sum_support_optimized[key+'|'+value[geo_name]] + support_optimized
                            randomization_sum_dependent[key+'|'+value[geo_name]] = randomization_sum_dependent[key+'|'+value[geo_name]] + dependent_col
            # -------------------    

            level_stats[file_name] = [r2, dw, mape, dep_agg]
            if not spec['is_resume']:
                transformstion_forgeo[file_name] = list(selected_transformations.values())
            if spec['granular_vol_agg']:
                vol_agg = []
                shift = spec['modelling_start_index']
                for p_key, splitter in spec['reporting_period_index'].items():
                    vol_agg.append(support_optimized[splitter[0] - shift:splitter[1] + 1 - shift].sum(axis=0))
                vol_agg = np.array(vol_agg).T.flatten() 
                vol_agg = np.concatenate((vol_agg, np.array(dependent_agg)))
                vol_agg_forgeo[file_name] = vol_agg
            if spec['granular_stats']:
                stats_dict, residual_SS, support_dev_sum, std_error_col_ss, transformed_support_mask = stats_calc(dependent_col, predicted, transformed_support, coeff, sumResidualSq, spec["config"]["include_intercept"])
                sum_coeff = sum_coeff + coeff
                sum_support_dev_sum = sum_support_dev_sum + support_dev_sum
                sum_residual_SS = sum_residual_SS + residual_SS
                sum_std_error_col_ss = sum_std_error_col_ss + std_error_col_ss
                sum_transformed_support_mask = sum_transformed_support_mask + transformed_support_mask

                #randomaization calc
                if spec["config"]['isRandomiziedFilter'] == 1:
                    for (key, value) in spec['rand_mapping_dict'].items():
                        geo_name = _file.split('.')[0]
                        if geo_name in value.keys():
                            if key+'|'+value[geo_name] not in randomization_sum_coeff:
                                randomization_sum_coeff[key+'|'+value[geo_name]]  =  coeff
                                randomization_sum_support_dev_sum[key+'|'+value[geo_name]]  =  support_dev_sum
                            else:
                                randomization_sum_coeff[key+'|'+value[geo_name]] = randomization_sum_coeff[key+'|'+value[geo_name]] + coeff     
                                randomization_sum_support_dev_sum[key+'|'+value[geo_name]]  =  randomization_sum_support_dev_sum[key+'|'+value[geo_name]] + support_dev_sum
                #===================
            granular_writter(output_path, file_name, spec, support_optimized, predicted, dependent_col, stats_dict, all_cols, transformed_support, trans_support_no_postmul, corr_data, bounds, scale_factor)
        # result_q.put({
        #     "status": True,
        #     "level_stats": level_stats,
        #     "support_optimized": sum_support_optimized,
        #     "dependent": sum_dependent,
        #     "residual_SS": sum_residual_SS,
        #     "support_dev_sum": sum_support_dev_sum,
        #     "coeff": sum_coeff,
        #     "std_error_col_ss": sum_std_error_col_ss,
        #     "sum_transformed_support_mask": sum_transformed_support_mask,
        #     "transformstion_forgeo": transformstion_forgeo,
        #     "optimization_report": optimizer_report,
        #     "system_unbound_var": system_unbound_var,
        #     "vol_agg_forgeo": vol_agg_forgeo,
        #     "randomization": {
        #             "randomization_sum_support_optimized": randomization_sum_support_optimized,
        #             "randomization_sum_dependent": randomization_sum_dependent,
        #             "randomization_sum_coeff": randomization_sum_coeff,
        #             "randomization_sum_support_dev_sum": randomization_sum_support_dev_sum
        #     }
            
        # })
        obj_info = {
            "status": True,
            "level_stats": level_stats,
            "support_optimized": sum_support_optimized,
            "dependent": sum_dependent,
            "residual_SS": sum_residual_SS,
            "support_dev_sum": sum_support_dev_sum,
            "coeff": sum_coeff,
            "std_error_col_ss": sum_std_error_col_ss,
            "sum_transformed_support_mask": sum_transformed_support_mask,
            "transformstion_forgeo": transformstion_forgeo,
            "optimization_report": optimizer_report,
            "system_unbound_var": system_unbound_var,
            "vol_agg_forgeo": vol_agg_forgeo,
            "randomization": {
                    "randomization_sum_support_optimized": randomization_sum_support_optimized,
                    "randomization_sum_dependent": randomization_sum_dependent,
                    "randomization_sum_coeff": randomization_sum_coeff,
                    "randomization_sum_support_dev_sum": randomization_sum_support_dev_sum
            }
            
        }
        write_obj_as_pickle(os.path.join(output_path, 'agg','agg_'+str(process)+'.p'), obj_info)
        print(f"{process} mp completed")
    except Exception as e:
        e = str(e)
        if "'NoneType' object has no attribute 'x'" in e:
            e_vars = []
            for k,v in geo_df.items():
                if(np.isnan(np.min(v))):
                    e_vars.append(k)
            if(np.isnan(np.min(dependent_col))):
                e_vars.append(dependent_name)
            if(len(e_vars) > 0):
                e = "201| Check for blank cells in the geography for variables - " + str(e_vars)
            elif(np.isnan(np.min(bounds))):
                e = "202| Check contribution applied for the geography"
        if "could not convert string to float" in e:
            e = "203| " + e + ". Please look for this value in the geography and convert it to numeric."
        
        # result_q.put({
        #     "status": False,
        #     "message": file_name + " | " + e + "\n\nDev Details\n===========\n" + traceback.format_exc(),
        # })
        error = {
            "status": False,
            "message": file_name + " | MP-" + str(process) + " | "+ e + "\n\nDev Details\n===========\n" + traceback.format_exc(),
        } 
        print(f"{process} mp failed")
        # raise Exception(error['message'])
        write_obj_as_pickle(os.path.join(output_path, 'agg','agg_'+str(process)+'.p'), error)

def run_model(input_path, spec, output_path, files, cpu_count, sample, dynamic_opt_itr):
    startTime = time.time()
    start_time_to_log = datetime.datetime.utcnow().strftime('%H:%M:%S')
    communicator(str(start_time_to_log), output_path + '/start_time.txt')
    status_path = output_path + '/status.txt' # Django communicates with this file to fetch status.
    # Handle Sample and CPU
    if sample != 0:
        files = files[0:sample]
    # if len(files) < 100:
    #     cpu_count = 1
    # Transmit the data
    communicator("STORE|" + str(len(files)), status_path)
    
    #Calculating Dep_Sum to calculate whether to use User defined bounds or engine calculated
    is_custom_dep_agg = True if spec["config"]["is_Custom_Dept_Agg"] == 1 else False# Will come from UI
    dep_threshold = spec["config"]["custom_Dept_Agg_Val"] #50 # Will come from UI
    print("is_custom_dep_agg: ", is_custom_dep_agg)
    print("dep_threshold: ", dep_threshold)
    if is_custom_dep_agg and dep_threshold > 0.0:
        print("I am in ")
        total_dep_sum = 0
        store_wise_dep_sum = {}
        for _file in files:
            file_name = _file.split(".")[0]
            store_dep_sum = pd.read_csv(input_path + "/" + _file, usecols=spec["all_cols_read"], dtype=np.float64, low_memory=False, memory_map=True, engine='c')[spec["dependent_name"]].to_numpy().T                
            store_wise_dep_sum[file_name] = store_dep_sum.sum()
            total_dep_sum += store_dep_sum.sum()

        store_wise_dep_sum_sorted_desc = dict(reversed(sorted(store_wise_dep_sum.items(), key=lambda item: item[1]))) 
        stores_need_bound = {}
        tmp_sum_to_threshold = 0
        for key, value in store_wise_dep_sum_sorted_desc.items():
            weight = (value/total_dep_sum)*100
            tmp_sum_to_threshold += weight
            stores_need_bound[key] = False if tmp_sum_to_threshold < dep_threshold else True
        
        spec['stores_need_bound'] = stores_need_bound
    spec['is_custom_dep_agg'] = is_custom_dep_agg
    #-------------------------------


    split_files = files_to_core_split(files, cpu_count)
    # Data from 1st file to get some info and add to spec
    period_s = pd.read_csv(input_path + "/" + files[0], usecols=[spec["config"]["period_colname"]], squeeze=True)
    period_s_dt = pd.to_datetime(period_s)
    # Transformation Period Check with ADS Start
    transformation_period_pos = periodFinder(period_s_dt, pd.to_datetime(spec["config"]["transformation_period"]))
    if transformation_period_pos != 0:
        raise Exception("104| Transformation start period must match with ADS start period")
    # Add Reporting Index
    try:
        reporting_period = spec["reporting_period"]
        reporting_period_index = {}
        for p_key in reporting_period.index:
            start = pd.to_datetime(reporting_period[reporting_period.columns[0]][p_key])
            end = pd.to_datetime(reporting_period[reporting_period.columns[1]][p_key])
            reporting_period_index[p_key] = [periodFinder(period_s_dt, start), periodFinder(period_s_dt, end)]
        spec['reporting_period_index'] = reporting_period_index
    except Exception:
        raise Exception("105| Reporting Period Issue - Make sure they are present in the ADS")

    # Modelling Period
    spec['modelling_start_index'] = periodFinder(period_s_dt, pd.to_datetime(spec["config"]["modelling_period"]["start_date"]))
    spec['modelling_end_index'] = periodFinder(period_s_dt, pd.to_datetime(spec["config"]["modelling_period"]["end_date"]))
    spec['modelling_end_trim'] = spec['modelling_end_index'] - (len(period_s) - 1)
    sx = spec["config"]['seasonality_exclude']
    if (sx and len(sx) > 0):
        sx = np.array(sx) - spec['modelling_start_index'] # align the index
        n, m = sx.min(), sx.max()  #  Check out of bounds
        spec["config"]['seasonality_exclude'] = list(sx)
        if(n < 0 or m > spec['modelling_end_index']):
            raise Exception("106| Sesonality Excluded is out of range - Excluded periods should be inside modelling period")

    if spec['modelling_end_trim'] == 0 : spec['modelling_end_trim'] = None
    period_s = period_s[spec['modelling_start_index']:spec['modelling_end_trim']].reset_index(drop=True)
    spec["period_s"] = period_s
    spec["week_count"] = len(period_s)
    spec["data_week_count"] = len(period_s_dt)
    # Check Degree Of Freedom
    # if(spec["granular_stats"] and spec["week_count"] + 1 < spec["var_count"]):
    #     spec["granular_stats"] = False
    #     # raise Exception("Degree of freedom error: DOF =" + str(spec["week_count"] - spec["var_count"] - 1))
    
    # Add Non Variable Related Dummies
    spec["dummy_data"] = gen_dummy(spec)
    # Transmit the data
    communicator("RUNNING|", status_path)

    # Processing starts
    result_q = mp.Queue() # Result Queue

    # If Popular transformation - Transformation Selector.
    n = 1 # Process Number
    procs = []
    if not spec['is_resume'] and spec["config"]['single_transformation']:
        for _files in split_files:
            proc = mp.Process(target=select_transformarion, args=(result_q, output_path, input_path, _files, spec, n))
            procs.append(proc)
            proc.start()
            n = n + 1

        transformstion_forgeo = {}
        for proc in procs:
            obj = result_q.get()
            if not obj["status"]:
                for proc in procs:
                    proc.terminate()
                fw = open(output_path + '/log.txt', 'w')
                fw.write(obj["message"])
                fw.close()
                #Transmit the data
                communicator("ERROR|" + obj["message"], status_path)
                for proc in procs:
                    proc.terminate()
                return
            transformstion_forgeo.update(obj['transformstion_forgeo'])
        # Popular Transformation
        transformstion_forgeo = pd.DataFrame(transformstion_forgeo)
        transformstion_forgeo = transformstion_forgeo.T
        popular = transformstion_forgeo.mode().iloc[[0]].squeeze() # There might be multiple mode, select 1st row
        popular.name = "POPULAR"
        transformstion_forgeo = transformstion_forgeo.append(popular)
        transformstion_forgeo.to_csv(output_path + '/transformstions.csv',  index=True, header=False)
        spec['is_resume'] = True
        spec['selected_transformations_allstores'] = transformstion_forgeo.T.to_dict(orient="list")
    ################
    # Main Pipe    #
    ################
    if not IS_MAIN:
        n = 1 # Process Number
        procs = []
        for _files in split_files:
            proc = mp.Process(target=main_pipeline, args=(result_q, output_path, input_path, _files, spec, n, dynamic_opt_itr))
            procs.append(proc)
            proc.start()
            n = n + 1
        for proc in procs:
            proc.join()
        print("MP completed")    
    # else: 
        # Agg Result From Queue - The main thread waits and listens to Queue till Queue returns results = no of process
        level_stats = {}
        sum_support_optimized = np.zeros((spec['week_count'],spec['var_count']))
        sum_dependent = np.zeros(spec['week_count'])
        sum_support_dev_sum = np.zeros(spec['var_count'] - 1 if spec["config"]["include_intercept"] else spec['var_count'])
        sum_std_error_col_ss = np.zeros(spec['var_count'])
        sum_transformed_support_mask = np.zeros(spec['var_count'])
        sum_residual_SS = 0
        sum_coeff = np.zeros(spec['var_count'])

        transformstion_forgeo = {}
        optimization_report = {}
        system_unbound_var = {}
        vol_agg_forgeo = {}

        # sum_support_optimized_level = np.zeros((spec['week_count'],spec['var_count']))
        # sum_dependent_level = np.zeros(spec['week_count'])
        randomization_sum_support_optimized = {}
        randomization_sum_dependent = {}
        randomization_sum_coeff = {}
        randomization_sum_support_dev_sum = {}
        randomization_level_stats = {}
        agg_files = os.listdir(os.path.join(output_path,'agg'))
        for proc in agg_files:#procs:
            obj = read_obj_from_pickle(os.path.join(output_path, 'agg', proc))#result_q.get()
            if not obj["status"]:
                raise Exception(obj["message"])
                # for proc in procs:
                #     proc.terminate()
                # fw = open(output_path + '/log.txt', 'w')
                # fw.write(obj["message"])
                # fw.close()
                # #Transmit the data
                # communicator("ERROR|" + obj["message"], status_path)
                # return
            sum_support_optimized = sum_support_optimized + obj['support_optimized']
            sum_dependent = sum_dependent + obj['dependent']
            level_stats.update(obj['level_stats'])
            transformstion_forgeo.update(obj['transformstion_forgeo'])
            optimization_report.update(obj['optimization_report'])
            system_unbound_var.update(obj['system_unbound_var'])
            vol_agg_forgeo.update(obj['vol_agg_forgeo'])

            if spec['granular_stats']:
                sum_support_dev_sum = sum_support_dev_sum + obj['support_dev_sum']
                sum_residual_SS = sum_residual_SS + obj['residual_SS']
                sum_coeff = sum_coeff + obj['coeff']
                sum_std_error_col_ss = sum_std_error_col_ss +  obj['std_error_col_ss']
                sum_transformed_support_mask = sum_transformed_support_mask + obj['sum_transformed_support_mask']
                for (key, value) in obj["randomization"]["randomization_sum_coeff"].items():
                    # sum_support_optimized_level = sum_support_optimized_level + value
                    if key not in randomization_sum_coeff:
                        randomization_sum_coeff[key] = value
                    else:
                        randomization_sum_coeff[key] = randomization_sum_coeff[key] + value
                for (key, value) in obj["randomization"]["randomization_sum_support_dev_sum"].items():
                    # sum_support_optimized_level = sum_support_optimized_level + value
                    if key not in randomization_sum_support_dev_sum:
                        randomization_sum_support_dev_sum[key] = value
                    else:
                        randomization_sum_support_dev_sum[key] = randomization_sum_support_dev_sum[key] + value        

            for (key, value) in obj["randomization"]["randomization_sum_support_optimized"].items():
                # sum_support_optimized_level = sum_support_optimized_level + value
                if key not in randomization_sum_support_optimized:
                    randomization_sum_support_optimized[key] = value
                else:
                    randomization_sum_support_optimized[key] = randomization_sum_support_optimized[key] + value
            for (key, value) in obj["randomization"]["randomization_sum_dependent"].items():
                # sum_dependent_level = sum_dependent_level + value
                if key not in randomization_sum_dependent:
                    randomization_sum_dependent[key] = value
                else:
                    randomization_sum_dependent[key] = randomization_sum_dependent[key] + value

        # End the process
        # for proc in procs:
        #     proc.join()
        # for (key, sum_support_optimized) in randomization_sum_support_optimized.items():
        #         print(key)
        #         print(value)
        #         predicted = sum_support_optimized.sum(axis=1)
        #         final_file = np.c_[sum_support_optimized, predicted, randomization_sum_dependent[key]]
                
        
        
        
        # for i in len(randomization_sum_support_optimized):
            
        #     final_file = np.c_[value, predicted, dependent_col]
        #     all_cols.append('Predicted')
        #     all_cols.append(spec['dependent_name'])
        #     df = pd.DataFrame(final_file, columns=all_cols)
        #     df.insert(0, period_s.name, period_s)
        #     df.to_csv(path + '/granular_result/' + file_name + '__result.csv', index=False)
        # # exit()
        # Level Stats
        # print(stats.rankdata)
        level_stats = pd.DataFrame(level_stats, index=['R - Square', 'DW', 'MAPE',"Aggregate"]).T
        no_of_stores = len(level_stats[level_stats["DW"] != -1])
        if no_of_stores > 1:
            level_stats["Rank"] = (level_stats["Aggregate"].rank(method="min") - 1) * (100 / no_of_stores) # percentile 0 to 100
        else:
            level_stats["Rank"] = 100
        level_stats["Ratio"] = level_stats["Aggregate"] / level_stats["Aggregate"].sum() * 100
        level_stats.insert(0, "Geography", level_stats.index)
        level_stats_out = os.path.join(output_path, "level_stats.csv")
        write_dataframe_as_csv(level_stats, level_stats_out)
        # level_stats.to_csv(output_path + '/level_stats.csv',  index=False)
        # Popular Transformation
        if spec['is_resume']:
            popular = spec['selected_transformations_allstores']["POPULAR"]
        else:
            transformstion_forgeo = pd.DataFrame(transformstion_forgeo)
            transformstion_forgeo = transformstion_forgeo.T
            popular = pd.Series(transformstion_forgeo.mode().iloc[[0]].squeeze()) # There might be multiple mode, select 1st row
            popular.name = "POPULAR"
            transformstion_forgeo = transformstion_forgeo.append(popular)
            transformstion_forgeo.to_csv(output_path + '/transformstions.csv',  index=True, header=False)
            popular = list(popular.values)
        # Optimization Report
        if optimization_report:
            optimization_report = pd.DataFrame(optimization_report, index=["Message", "Method"])
            optimization_report = optimization_report.T
            optimization_report.to_csv(output_path + '/optimization_report.csv',  index=True)
        # system_unbound_var 
        unbound_by_variable = {}
        for store in system_unbound_var.keys():
            for varItem in system_unbound_var[store]:
                if varItem["c"] in unbound_by_variable.keys():
                    unbound_by_variable[varItem["c"]].append(store + " = " + varItem["r"])
                else:
                    unbound_by_variable[varItem["c"]] = [store + " = " + varItem["r"]]
        for k in unbound_by_variable.keys():
            unbound_by_variable[k] = pd.Series(unbound_by_variable[k])
        unbound_by_variable = pd.DataFrame(unbound_by_variable)
        # unbound_by_variable.to_csv(output_path + '/contraints_rejected.csv',  index=False)
        unbound_by_variable_out = os.path.join(output_path, "contraints_rejected.csv")
        write_dataframe_as_csv(unbound_by_variable, unbound_by_variable_out)

        # Overall Calc
        predicted = sum_support_optimized.sum(axis=1)
        r2, dw, mape, sumResidualSq, dep_agg = calc_r2_dw_mape(sum_dependent, predicted)
        r2_dw_mape = pd.DataFrame([[r2,dw,mape]], columns=["R-Square", "DW", "MAPE"])
        r2_dw_mape_out = os.path.join(output_path, "RDM.csv")
        write_dataframe_as_csv(r2_dw_mape, r2_dw_mape_out)
        
        # r2_dw_mape.to_csv(output_path + '/RDM.csv',  index=False)
        print(r2_dw_mape)

        # Final File Gen
        additional_cols = []
        if spec["config"]["include_seasonality"]:
            additional_cols.append("Seasonality")
        if spec["config"]["include_intercept"]:
            additional_cols.append("Intercept")
        if spec["config"]["stochastic_sea"]:
            additional_cols.append("STO_SEA")

        all_var_cols = popular + spec['nontransformation_cols'] + additional_cols

        # Volume Agg 
        if spec['granular_vol_agg']:
            indexes = pd.MultiIndex.from_product([all_var_cols + [spec['dependent_name']], spec['reporting_period_index'].keys()],
                                    names=['Variable', 'Period'])
            vol_agg_df = pd.DataFrame(vol_agg_forgeo, index = indexes)
            vol_agg_out = os.path.join(output_path, "volume_agg.csv")
            write_dataframe_as_csv(vol_agg_df.T, vol_agg_out)
            # vol_agg_df.T.to_csv(output_path + "\\" + "volume_agg.csv")

        # Overall Stats
        if spec['granular_stats']:
            stats_dict, total_transformed_support = overall_stats_gen(sum_coeff, sum_support_dev_sum, sum_std_error_col_ss, sum_residual_SS, sum_transformed_support_mask, no_of_stores, spec['week_count'], spec["config"]["include_intercept"], sum_support_optimized, sumResidualSq)
            stats_dict['Pval'] = stats.t.sf(np.abs(stats_dict['T Stat']), len(sum_support_dev_sum) - 1) * 2
            stats_df = pd.DataFrame(stats_dict)
            stats_df.insert(0, "Variable", all_var_cols)
            stats_df_out = os.path.join(output_path, "result_stats.csv")
            write_dataframe_as_csv(stats_df, stats_df_out)
            total_transformed_support_df = pd.DataFrame(total_transformed_support, columns = all_var_cols)
            total_transformed_support_df_corr = total_transformed_support_df.corr()
            total_transformed_support_df_corr.insert(0, "Variable", all_var_cols)
            total_transformed_support_out = os.path.join(output_path, "multicollinearity_matrix.csv")
            write_dataframe_as_csv(total_transformed_support_df_corr, total_transformed_support_out)
            # stats_df.to_csv(output_path + '/result_stats.csv', index=False)

            #Randomization stats
            for (key, sum_coeff) in randomization_sum_coeff.items():
                r_sum_support_optimized = randomization_sum_support_optimized[key]
                no_of_stores = len(dict(filter(lambda item: key.split('|')[1] in item[1], spec['rand_mapping_dict'][key.split('|')[0]].items())))
                coeff = sum_coeff / no_of_stores 
                r_predicted = r_sum_support_optimized.sum(axis=1)
                r2, dw, mape, sumResidualSq, dep_agg = calc_r2_dw_mape(randomization_sum_dependent[key], r_predicted)
                transformed_support = np.divide(r_sum_support_optimized, coeff, out=np.zeros_like(r_sum_support_optimized), where=coeff!=0) #r_sum_support_optimized/coeff
                stats_dict, residual_SS, support_dev_sum, std_error_col_ss, transformed_support_mask = stats_calc(0, 0, transformed_support, coeff, sumResidualSq, spec["config"]["include_intercept"], no_of_stores)
                #print
                stats_dict['Pval'] = stats.t.sf(np.abs(stats_dict['T Stat']), len(randomization_sum_support_dev_sum) - 1) * 2
                df = pd.DataFrame(stats_dict)
                df.insert(0, "Variable", all_var_cols)
                rand_out = os.path.join(output_path, "randomization_stats", key.split('|')[0]+'_stats', str(key.split('|')[1]) + '__stats.csv')
                write_dataframe_as_csv(df, rand_out)
                # df.to_csv(output_path + '\\randomization_stats\\'+key.split('|')[0]+'_stats\\' + str(key.split('|')[1]) + '__stats.csv', index=False)
        
        
        
        all_cols = all_var_cols + ['Predicted', spec['dependent_name']]

        final_file = np.c_[sum_support_optimized, predicted, sum_dependent]
        final_file = pd.DataFrame(final_file, columns=all_cols)
        final_file.insert(0, period_s.name, period_s)
        final_file.insert(1, spec["config"]["geo_colname"], ["Total"] * len(final_file))
        final_out = os.path.join(output_path, "result.csv")
        write_dataframe_as_csv(final_file, final_out)
        # final_file.to_csv(output_path + '/result.csv',  index=False)

        #randomization

        for (key, sum_support_optimized) in randomization_sum_support_optimized.items():

            predicted = sum_support_optimized.sum(axis=1)
            
            r2, dw, mape, sumResidualSq, dep_agg = calc_r2_dw_mape(randomization_sum_dependent[key], predicted)
            level_stats = {}
            level_stats[key.split('|')[1]] = [r2, dw, mape, dep_agg]
            randomization_level_stats.update(level_stats)
            
            final_file = np.c_[sum_support_optimized, predicted, randomization_sum_dependent[key]]
            df = pd.DataFrame(final_file, columns=all_cols)
            df.insert(0, period_s.name, period_s)
            rand_res_out = os.path.join(output_path, "randomization_result", key.split('|')[0]+'_result', str(key.split('|')[1]) + '__result.csv')
            write_dataframe_as_csv(df, rand_res_out)
            # df.to_csv(output_path + '\\randomization_result\\'+key.split('|')[0]+'_result\\' + str(key.split('|')[1]) + '__result.csv', index=False)
        
        if bool(spec["config"]["isRandomiziedFilter"]):
            level_stats = pd.DataFrame(randomization_level_stats, index=['R - Square', 'DW', 'MAPE',"Aggregate"]).T
            no_of_stores = len(level_stats[level_stats["DW"] != -1])
            if no_of_stores > 1:
                level_stats["Rank"] = (level_stats["Aggregate"].rank(method="min") - 1) * (100 / no_of_stores) # percentile 0 to 100
            else:
                level_stats["Rank"] = 100
            level_stats["Ratio"] = level_stats["Aggregate"] / level_stats["Aggregate"].sum() * 100
            level_stats.insert(0, "Level", level_stats.index)
            rand_level_stat_out = os.path.join(output_path, "randomized_level_stats.csv")
            write_dataframe_as_csv(level_stats, rand_level_stat_out)
            # level_stats.to_csv(output_path + '/randomized_level_stats.csv',  index=False)
        
        #New granular_transformed_support no postmul aggregation
        trans_no_postmul_path = os.path.join(output_path,"granular_support_no_postmul")
        transS_no_postmul_df = None

        for files in os.listdir(trans_no_postmul_path):
            file_path = os.path.join(trans_no_postmul_path, files)
            temp_df = pd.read_csv(file_path)
            
            if transS_no_postmul_df is None:
                period = temp_df["Period"]
                column_names = [eachCol.split("__")[0] for eachCol in temp_df.columns]
                temp_df.columns = column_names
                transS_no_postmul_df = temp_df.iloc[:,1:]
            else:
                temp_df.columns = column_names            
                transS_no_postmul_df += temp_df.iloc[:,1:]
        transS_no_postmul_df.insert(0,"Period",period)
        transS_no_postmul_df = transS_no_postmul_df.set_index("Period").T
        transS_no_postmul_df.index.name = "Variable"
        transS_no_postmul_df.to_csv(os.path.join(output_path, "transformed_support_agg_no_postmul.csv"))
        shutil.rmtree(trans_no_postmul_path)
        
        done_time = str(time.time() - startTime)
        communicator("DONE|" + done_time, status_path)
        if IS_RUN_ON_BATCH:
            updateJob(str(os.environ["MODEL_ID"]), "DONE", True)
            updateStatusModelDB(str(os.environ["MODEL_ID"]), 0)
            updateModelDBRunTime(str(os.environ["MODEL_ID"]), start_time_to_log, done_time)
        return "SUCCESS"

def gen_dt_index(data, period_s):
    data = data[~data.isnull()].apply(lambda x: periodFinder(period_s,x))
    return data.to_list()

def run_model_initiator(ads_path, efs_split_dir, spec_path, output_path, cpu_count = 4, sample=False, use_store=False, fast_optimize=False, resume=False, granular_result=False, granular_stats=False, granular_support=False, granular_correlation=False, granular_vol_agg=False, geolevel_contraint=True, dynamic_opt_itr = 200):    
    # Derive Paths
    input_path, status_path, geolevel_spec, spec_jsonpath = get_paths(ads_path, output_path, spec_path)
    print("input_path, status_path, geolevel_spec, spec_jsonpath: ", input_path, status_path, geolevel_spec, spec_jsonpath)
    input_path = input_path if efs_split_dir=='' else efs_split_dir #splitted in a separated job
    print("input_path: ", input_path)
    try: 
        # Spec Reader
        spec = read_spec_json(spec_jsonpath, status_path)
        #Special case where user will give exact desired volume as a constrints inplace of contribution
        if spec["is_volumetric_constraints"]:
            df = pd.read_csv(ads_path, usecols=[spec["config"]["geo_colname"], spec["dependent_name"]])
            grouped = df.groupby(spec["config"]["geo_colname"])[spec["dependent_name"]].sum().reset_index()
            sum_dependent = sum(grouped[spec["dependent_name"]])
            dependent_distribution = {}
            for idx, i in enumerate(grouped[spec["dependent_name"]]):
                dependent_distribution[grouped[spec["config"]["geo_colname"]][idx]] = (i/sum_dependent)
            spec["dependent_distribution"] = dependent_distribution
        #============================================================================================== 
        rand_mapping_dict = {}
        if spec["config"]["isRandomiziedFilter"] ==1:
            randomization_mapping_path = ads_path.split('.csv')[0] + '_randomize_mapping.csv'
            # if not path.exists(randomization_mapping_path):
            df = read_as_dataFrame(ads_path, spec["randomized_layers"])[spec["randomized_layers"]]
            df = df.drop_duplicates(subset = spec["randomized_layers"], ignore_index=True)
            write_dataframe_as_csv(df, randomization_mapping_path)#df.to_csv(randomization_mapping_path , index=False)     
            rand_mapping_dict = read_randomization_mapping(randomization_mapping_path, spec["randomized_layers"])
        spec['rand_mapping_dict'] = rand_mapping_dict
        
        # Add Options
        spec['fast_optimize'] = fast_optimize or spec["config"]["fast_opt"]
        spec['is_resume'] = resume
        if spec['is_resume']:
            spec['selected_transformations_allstores'] = pd.read_csv(output_path + "/" + "transformstions.csv", index_col=0, header=None).T.to_dict(orient="list")
        spec['granular_stats'] = granular_stats
        spec['granular_result'] = granular_result
        spec['granular_support'] = granular_support
        spec['granular_correlation'] = granular_correlation
        spec['granular_vol_agg'] = granular_vol_agg
        
        # Create Path if does not exit, also determine weather split is required.
        split = output_path_creator(output_path, input_path, granular_result, granular_stats, granular_support, True, rand_mapping_dict)
        # Splitter
        if split:
            communicator("SPLIT|", status_path)
            ads_file_splitter(ads_path, input_path, spec["config"]["geo_colname"])
        # Rapid Refresh
        contri_map = {}
        # if spec['old_model_path']:
        #     spec, old_model_diff_files = add_sea_and_intercept(input_path, spec, output_path)
        #     for _file in old_model_diff_files: # Allow Optimizer to include sea & inter for new files
        #         name = _file.split(".")[0]
        #         contri_map[name] = spec["contribution_range"]
        #         if "Seasonality" in contri_map[name].keys(): contri_map[name]["Seasonality"]['contri_period'] = None
        #         if "Intercept" in contri_map[name].keys(): contri_map[name]["Intercept"]['contri_period'] = None
        # Files for processes
        if use_store:
            files_to_run = pd.read_csv(use_store, low_memory=False, memory_map=True, engine='c', squeeze=True)
            files_to_run = files_to_run + ".csv"
            files = files_to_run.to_list()
        else:
            files = get_all_files(input_path)
        # Handle Granular Spec & Multiple Contribution
        contri_map["GLOBAL"] = spec["contribution_range"]
        geo_map_contri = {}
        geo_ex_intercept = {}
        geo_map_sea = {}
        sea_map = {}
        geo_ex_exdep = {}
        override_forgeo = {}
        if isinstance(geolevel_contraint, str) and geolevel_contraint != "": # For console a string will be passed.
            geolevel_spec = geolevel_contraint
        if path.isfile(geolevel_spec) and geolevel_contraint:
            # Use any keyword to map apart from the existing GLOBAL, MAP, EX-SEA. Changes in GLOBAL will not reflect.
            geo_map = pd.read_excel(geolevel_spec, sheet_name='MAP')
            # Exclude Intercept
            if("EX_INTERCEPT" in geo_map):
                geo_ex_intercept = geo_map[geo_map["EX_INTERCEPT"] == 1]
                geo_ex_intercept["EX_INTERCEPT"] = True
                geo_ex_intercept.set_index('GEOGRAPHY', inplace=True)
                geo_ex_intercept = geo_ex_intercept['EX_INTERCEPT'].to_dict()
            # Exclude Of Exclude Dependent Zero
            if("EX_EXDEP" in geo_map):
                geo_ex_exdep = geo_map[geo_map["EX_EXDEP"] == 1]
                geo_ex_exdep["EX_EXDEP"] = True
                geo_ex_exdep.set_index('GEOGRAPHY', inplace=True)
                geo_ex_exdep = geo_ex_exdep['EX_EXDEP'].to_dict()
            # Exclude Seasonality
            if("EX_SEA" in geo_map):
                geo_map_sea = geo_map[~geo_map["EX_SEA"].isnull()]
                ex_sea_tab = pd.read_excel(geolevel_spec, sheet_name='EX_SEA')
                # Data from 1st file to get some info and add to spec
                period_s = pd.read_csv(input_path + "/" + files[0], usecols=[spec["config"]["period_colname"]], squeeze=True)
                period_s_dt = pd.to_datetime(period_s)
                for group in geo_map_sea['EX_SEA'].unique():
                    sea_map[group] = gen_dt_index(ex_sea_tab[group], period_s_dt)
            # Granular Contribution
            geo_map_contri = geo_map[geo_map["MAP"] != "GLOBAL"]
            unique_contri = geo_map_contri['MAP'].unique()
            for tab in unique_contri:
                tab_df = pd.read_excel(geolevel_spec, sheet_name=tab).rename(columns={"Geography": "Variable","Contribution": "contri_period", "Min": "min", "Max": "max", "Sign": "sign"}, errors="ignone").set_index("Variable").replace({np.nan: None})
                tab_df['contri_period'] = tab_df['contri_period'].apply(lambda v: str(v) if v else None).tolist() # Handle Numbers -> String
                f_contri = copy.deepcopy(spec["contribution_range"])
                tab_dict = tab_df.T.to_dict()
                for k,v in tab_dict.items():
                    f_contri[k] = v
                contri_map[tab] = f_contri

            geo_map_contri = geo_map_contri.set_index("GEOGRAPHY")['MAP'].to_dict() if isinstance(geo_map_contri, pd.DataFrame) else {}
            # Granular Contribution - Override
            
            try:
                override_df = pd.read_excel(geolevel_spec, sheet_name='OVERRIDE').rename(columns={"Geography": "geo", "Variable": "variable", "Contribution": "contri_period", "Min": "min", "Max": "max", "Sign": "sign"}, errors="igone").replace({np.nan: None})
                override_df['contri_period'] = override_df['contri_period'].apply(lambda v: str(v) if v else None).tolist() # Handle Numbers -> String
                override_df = list(override_df.T.to_dict().values())
                for item in override_df:
                    if(item['geo'] in override_forgeo.keys()):
                        override_forgeo[item['geo']].append(item)
                        item.pop("geo")
                    else:
                        override_forgeo[item['geo']] = [item]
                        item.pop("geo")
            except: 
                pass
        spec['geo_map_contri'] = geo_map_contri
        spec['contri_map'] = contri_map
        spec['contri_override'] = override_forgeo
        spec['geo_map_sea'] = geo_map_sea.set_index("GEOGRAPHY")['EX_SEA'].to_dict() if isinstance(geo_map_contri, pd.DataFrame) else {}
        spec['sea_map'] = sea_map
        spec['geo_ex_intercept'] = geo_ex_intercept
        spec['geo_ex_exdep'] = geo_ex_exdep
        # Let's Roll
        return run_model(input_path, spec, output_path, files, cpu_count=cpu_count, sample=sample, dynamic_opt_itr=dynamic_opt_itr)
    except Exception as e:
        msg = "PREMODEL-ERROR|" + str(e) + "\n\nDev Details\n===========\n" + traceback.format_exc()
        communicator(msg, status_path)
        if IS_RUN_ON_BATCH:
            updateJob(str(os.environ["MODEL_ID"]), "FAILED", False)
        updateStatusModelDB(str(os.environ["MODEL_ID"]),-1)
        raise Exception(str(e))    

def gen_report(ads_path, spec_path, output_path, filename):
    # Derive Paths
    input_path, status_path, geolevel_spec, spec_jsonpath = get_paths(ads_path, output_path, spec_path)
    # Spec Reader
    spec = read_spec_json(spec_jsonpath, status_path)
    # Create Path if does not exit, also determine weather split is required.
    split = output_path_creator(output_path, input_path, False, False, False, False)
    # Splitter
    if split:
        communicator("SPLIT|", status_path)
        ads_file_splitter(ads_path, input_path, spec["config"]["geo_colname"])

    # Files for processes
    files = [f for f in os.listdir(input_path) if path.isfile(path.join(input_path, f))]
    # Data from 1st file to get some info and add to spec
    period_s = pd.read_csv(input_path + "/" + files[0], usecols=[spec["config"]["period_colname"]], squeeze=True)
    period_s_dt = pd.to_datetime(period_s)
    
    spec['modelling_start_index'] = periodFinder(period_s_dt, pd.to_datetime(spec["config"]["modelling_period"]["start_date"]))
    spec['modelling_end_index'] = periodFinder(period_s_dt, pd.to_datetime(spec["config"]["modelling_period"]["end_date"]))
    spec['modelling_end_trim'] = spec['modelling_end_index'] - (len(period_s) - 1)

    if spec['modelling_end_trim'] == 0 : spec['modelling_end_trim'] = None
    period_s = period_s[spec['modelling_start_index']:spec['modelling_end_trim']].reset_index(drop=True)
    
    spec["period_s"] = period_s
    spec["week_count"] = len(period_s)
    spec["data_week_count"] = len(period_s_dt)
    # Add Reporting Index
    try:
        reporting_period = spec["reporting_period"]
        reporting_period_index = {}
        for p_key in reporting_period.index:
            start = pd.to_datetime(reporting_period[reporting_period.columns[0]][p_key])
            end = pd.to_datetime(reporting_period[reporting_period.columns[1]][p_key])
            reporting_period_index[p_key] = [periodFinder(period_s_dt, start), periodFinder(period_s_dt, end)]
        spec['reporting_period_index'] = reporting_period_index
    except Exception:
        communicator("ERROR| Reporting Period Issue - Make sure they are present in the ADS", status_path)
        raise Exception("ERROR| Transformation start period must match with ADS start period",)

    
    spec["dummy_data"] = gen_dummy(spec)
    nontransformation_colnames = spec['nontransformation_cols']
    transformation_colnames = spec['transformation_cols']
    dependent_name = spec['dependent_name']
    report = {}
    # Each file operaration
    for _file in files:
        # Read from file and convert to dict of numpy.
        file_name = _file.split(".")[0]
        geo_df = pd.read_csv(input_path + "/" + _file, usecols=spec["all_cols_read"], dtype=np.float64, low_memory=False, memory_map=True, engine='c')[spec["all_cols_read"]].to_numpy().T
        geo_df = { spec["all_cols_read"][i]: geo_df[i] for i in range(len(spec["all_cols_read"]))} # takes care of column arrangement.
        dependent_col = geo_df[dependent_name]
        # Split Variables
        geo_df = split_variables(geo_df, spec)
        # Dummies
        for k,v in spec["dummy_data"].items():
            geo_df[k] = v
        # 3 Data  - transformed and 2 support, transformed support
        support_transformation_data = np.array([geo_df[col] for col in transformation_colnames]).T
        support_nontransformation_data = np.array([geo_df[col] for col in nontransformation_colnames]).T
        support = combine_array(support_transformation_data, support_nontransformation_data)
        support_agg = []
        dependent_agg = []
        i = 0
        for p_key, splitter in reporting_period_index.items():
            dependent_agg.append(dependent_col[splitter[0]:splitter[1] + 1].sum())
            support_agg.append(support[splitter[0]:splitter[1] + 1].sum(axis=0)) 
        support_agg = np.array(support_agg).T.flatten() 
        support_agg = np.concatenate((support_agg, np.array(dependent_agg)))
        report[file_name] = support_agg
    indexes = pd.MultiIndex.from_product([transformation_colnames + nontransformation_colnames + [dependent_name], reporting_period_index.keys()],
                            names=['Variable', 'Period'])
    # indexes = list(itertools.product(transformation_colnames + nontransformation_colnames + [dependent_name], reporting_period_index.keys()))
    report_df = pd.DataFrame(report, index = indexes)
    report_df.T.to_csv(output_path + "\\" + filename + ".csv")
    return "Success"

def rapid_refresh_new(old_path, new_path, output_path, spec_jsonpath, is_refresh = None, is_final_opt_check = None):
    try:
        # var_input = {
        #     "MAS_SPD": {
        #         "start": "06/11/2016",
        #         "end": "12/31/2016"
        #     },
        #     "TAS_SPD": {
        #         "start": "06/11/2016",
        #         "end": "12/31/2016"
        #     }
        # }
        # new_specs = {
            # "DEP_ALL": {
            #     "min":0.04019999999999999,
            #     "max": 0.0661,
            #     "contri_period": "FY 2016",
            #     "sign": "pos",
            # }
        # }
        use_old_intercept = False
        stitch_var_path = output_path+"\\stitch_var_input.csv"
        refresh = False if is_refresh and is_final_opt_check else True if is_refresh else False
        is_Final_Opt = True if is_final_opt_check else False
        
        print("refresh; ", refresh)
        print("is_Final_Opt: ", is_Final_Opt)
        
        var_input = {}
        
        if refresh and os.path.exists(stitch_var_path):
            selected_vars = pd.read_csv(stitch_var_path)
            for i in selected_vars.to_numpy():
                var_input[i[0]] = {'start': i[1], 'end': i[2] }
        # print("var_input: ", var_input)            
        new_specs = {}
        custom_intercept = ''
        if is_Final_Opt and os.path.exists(output_path+"\\final_opt_constraints.csv"):
            final_opt_specs = pd.read_csv(output_path+"\\final_opt_constraints.csv")
            for i in final_opt_specs.to_numpy():
                new_specs[i[0]] = {'contri_period': i[1], 'min': i[2], 'max': i[3], 'sign': i[4] }
                if int(i[2]) == 0 and int(i[3]) == 0:
                    custom_intercept = i[0]
        # print("new_specs: ", new_specs) 
        #===================================================Input ends here=========================================
        startTime = time.time()
        
        
        #====================================Rudra's==============================================================

        status_path = output_path + "\\status.txt"
        communicator("RUNNING|", status_path)
        # Spec read
        
        spec = read_spec_json(spec_jsonpath, status_path)
        
        g_old = old_path + "\\granular_result"
        g_new = new_path + "\\granular_result"
        g_old_files = [f for f in os.listdir(g_old) if os.path.isfile(os.path.join(g_old, f))]
        g_new_files = [f for f in os.listdir(g_new) if os.path.isfile(os.path.join(g_new, f))]
        # 1st file operations
        df_old = pd.read_csv(g_old + "\\" + g_old_files[0], low_memory=False, memory_map=True, engine='c')
        df_new = pd.read_csv(g_new + "\\" + g_new_files[0], low_memory=False, memory_map=True, engine='c')

        # Data from 1st file to get some info and add to spec
        period_s = pd.read_csv(g_new + "\\" + g_new_files[0], usecols=[spec["config"]["period_colname"]], squeeze=True)
        period_s_dt = pd.to_datetime(period_s)


        col_old = df_old.columns.tolist()
        col_new = df_new.columns.tolist()
        
        col_new_vars = [i.split("__")[0] for i in col_new]

        if not custom_intercept and "INT_CNT" in col_new_vars:#should change
            custom_intercept = "INT_CNT"
        # custom_intercept_var = "INT_CNT" #should come from UI
        intercept_var = custom_intercept if custom_intercept else "Intercept"
        if intercept_var in col_new and intercept_var in col_old:
            use_old_intercept = True 

        print("intercept_var: ", intercept_var)
        print("custom_intercept: ", custom_intercept)
            # -- Based on condition, find location of seasonality & intercept - "standard names apply"
        # remove period
        period_name = col_old.pop(0)
        col_new.pop(0)
        # remove dependent
        dep_name = col_old.pop()
        col_new.pop()
        # remove predicted
        col_old.pop()
        col_new.pop()
        # Common Cols in Old (removing transformations)
        common_cols = list(set([i.split("__")[0] for i in col_old]) & set([i.split("__")[0] for i in col_new]))

        sum_support_optimized = np.zeros((len(period_s),len(col_new))) #np.zeros((spec['week_count'],spec['var_count'])), need spec here
        sum_dependent =  np.zeros(len(period_s)) #np.zeros(spec['week_count'])
        level_stats = {}
        
        
        old_to_new_col_map = {}
        for i, v in enumerate(col_new):
            for j, v2 in enumerate(common_cols):
                if v.split("__")[0] == v2:
                    old_to_new_col_map[j] = i
        # Note: Old will be read using common, new we will read fully. We need to remove period in new for index match. 

        # print("OLD: ",col_old)
        # print( "NEW: ", col_new)
        # print( "common_cols: ", common_cols)
        # print("old_to_new_col_map: ",old_to_new_col_map)

        # Common Period
        period_old = pd.to_datetime(df_old.iloc[:, 0]) #old period column
        period_new = pd.to_datetime(df_new.iloc[:, 0]) #new period column    
        new_start_index_at_old = periodFinder(period_old,period_new[0])
        old_end_index_at_new = periodFinder(period_new,period_old[len(period_old) -1])
        
        old_chunk = [new_start_index_at_old, len(period_old)]
        new_chunk = [0, old_end_index_at_new + 1]
        # This is the common section
        
        # Add to locations
        for i in var_input:
            var_input[i]["colLoc"] = col_new_vars.index(i)
            var_input[i]["startLoc"] = periodFinder(pd.to_datetime(period_new), pd.to_datetime(var_input[i]["start"]))
            var_input[i]["endLoc"] = periodFinder(pd.to_datetime(period_new), pd.to_datetime(var_input[i]["end"]))+1
        
        # Lets start with actual files
        for _file in g_new_files:
            new_df = pd.read_csv(g_new + "\\" + _file, low_memory=False, memory_map=True, engine='c')
            dep_col = new_df[dep_name].to_numpy().T
            period_s = new_df[period_name]
            new_df = new_df.drop([period_name, "Predicted", dep_name], axis=1)
            
            new_numpy = new_df.to_numpy().T
            backup_new = np.copy(new_numpy)
            if not is_Final_Opt:
                if _file in g_old_files:
                    # print("stitching")
                    old_df = pd.read_csv(g_old + "\\" + _file,usecols= lambda x: x.split("__")[0] in common_cols, low_memory=False, memory_map=True, engine='c')
                    old_df.columns = [x.split("__")[0] for x in old_df.columns]
                    old_df = old_df[common_cols] # arrangement
                    old_numpy = old_df.to_numpy().T
                    # -- Handle Intercept & Seasonality - either generate new, use old, use new. Refer to old rapid refresh for using existing, refer main sea gen for generating new. 
                    # Stiching begins
                    # Block Handle - replace new with old
                    for oldI, newI in old_to_new_col_map.items():
                        new_numpy[newI][new_chunk[0]: new_chunk[1]] = old_numpy[oldI][old_chunk[0]: old_chunk[1]]
                    # Individual input - retain the new inputs.
                    for k,v in var_input.items():
                        new_numpy[v["colLoc"]-1][v["startLoc"]:v["endLoc"]] = backup_new[v["colLoc"]-1][v["startLoc"]:v["endLoc"]]
                    if use_old_intercept:
                        old_intercept_val = old_df[intercept_var].values[0]
                        new_intercept_col = col_new_vars.index(intercept_var)
                        result = []
                        for index, item, in enumerate(new_numpy[new_intercept_col-1], start=0):
                            result.append(old_intercept_val)
                        new_numpy[new_intercept_col-1] = result
                    # -- Nothing to do here
            transformed_support = new_numpy.T # 159 of engine.py
            #=====================================Rudra's End============================================================
            dependent_col = dep_col
            all_cols = col_new
            #======================================If Final opt=====================================================================
            # Add Reporting Index
            if is_Final_Opt and len(new_specs) > 0:
                # print("Optimizing")
                try:
                    reporting_period = spec["reporting_period"]
                    reporting_period_index = {}
                    for p_key in reporting_period.index:
                        start = pd.to_datetime(reporting_period[reporting_period.columns[0]][p_key])
                        end = pd.to_datetime(reporting_period[reporting_period.columns[1]][p_key])
                        reporting_period_index[p_key] = [periodFinder(period_s_dt, start), periodFinder(period_s_dt, end)]
                    spec['reporting_period_index'] = reporting_period_index
                except Exception:
                    raise Exception("105| Reporting Period Issue - Make sure they are present in the ADS")
                
                # 7 Period Aggregations - dependent, transformed and 2 support
                transformed_support_agg = []
                support_agg = []
                dependent_agg = []
                for p_key, splitter in reporting_period_index.items():
                    transformed_support_agg.append(transformed_support[splitter[0]:splitter[1] + 1].sum(axis=0))
                    dependent_agg.append(dependent_col[splitter[0]:splitter[1] + 1].sum())
                    # support_agg.append(transformed_support_agg[splitter[0]:splitter[1] + 1].sum(axis=0))

                contribution_range = {}
                allvar_support_colnames = [i.split("__")[0] for i in col_new]
                for i in allvar_support_colnames:
                    if i in new_specs.keys():
                        contribution_range[i] = {'contri_period': new_specs[i]["contri_period"], 'min': new_specs[i]["min"], 'max': new_specs[i]["max"], 'sign': new_specs[i]["sign"]}   
                    else:
                        contribution_range[i] = {'contri_period': 'FIXED', 'min': np.nan, 'max': np.nan, 'sign': np.nan}
                # print("contribution_range: ", contribution_range)
                bounds, scale_vector, system_unbound = bounds_calc_for_refresh(contribution_range, transformed_support_agg, dependent_agg, allvar_support_colnames, list(reporting_period_index.keys()))
                if "Intercept" in col_new_vars and "Seasonality" in col_new_vars:
                    bounds[-1] = [-np.inf,np.inf]
                    bounds[-2] = [-np.inf,np.inf]
                elif "Intercept" in col_new_vars:
                    bounds[-1] = [-np.inf,np.inf]
                elif "Seasonality" in col_new_vars:
                    bounds[-1] = [-np.inf,np.inf]
                if custom_intercept:
                    custom_intercept_Col = col_new_vars.index(custom_intercept)
                    bounds[custom_intercept_Col-1] = [-np.inf,np.inf]  
                # print("bounds: ", bounds)      
                coeff, opt_status, message, method = run_optimization(transformed_support, dependent_col, all_cols, bounds, scale_vector)
                # print("coeff: ", coeff)
                transformed_support = transformed_support*coeff
                # exit()
            #===============================================================================================================
            # exit()
            # if is_Final_Opt and len(new_specs) > 0:
            #     allvar_support_colnames = [i.split("__")[0] for i in col_new]
            #     print("allvar_support_colnames: ", allvar_support_colnames)
            #     exit()
            #     for col_index in range(len(allvar_support_colnames)):
            #         exit()
            # print("transformed_support: ", len(transformed_support))
            # --Bounds Formation - [1,1] for fixed, [-inf, inf] for intercept, seasonality (optional)
            #--------- refer to main_pipeline
            # --Optimization
            # --R2 Main Stats Calculation 
            # --Aggregate per process
            # --Granular File Generation
            #===============================================================================================================

            
            support_optimized = transformed_support #coeff * transformed_support # we do not need co-eff here
            predicted = support_optimized.sum(axis=1)
            r2, dw, mape, sumResidualSq, dep_agg = calc_r2_dw_mape(dependent_col, predicted)
            
            # Aggregate Per Process + Granular stats
            sum_dependent = sum_dependent + dependent_col
            
            sum_support_optimized = sum_support_optimized + support_optimized
            
            
            level_stats[_file.split('__result.csv')[0]] = [r2, dw, mape, dep_agg]
            #gran res
            final_file = np.c_[support_optimized, predicted, dependent_col]
            all_cols.append('Predicted')
            all_cols.append(dep_name)
            df = pd.DataFrame(final_file, columns=all_cols)
            all_cols.pop()
            all_cols.pop()
            df.insert(0, period_s.name, period_s)
            df.to_csv(output_path + '\\granular_result\\' + _file , index=False)#+ '__result.csv'
            # print("Done: ", _file)
        #--Overall Results  - DRM, result, level stats (389 of engine)
        # Level Stats
        # print(stats.rankdata)
        level_stats = pd.DataFrame(level_stats, index=['R - Square', 'DW', 'MAPE',"Aggregate"]).T
        no_of_stores = len(level_stats[level_stats["DW"] != -1])
        if no_of_stores > 1:
            level_stats["Rank"] = (level_stats["Aggregate"].rank(method="min") - 1) * (100 / no_of_stores) # percentile 0 to 100
        else:
            level_stats["Rank"] = 100
        level_stats["Ratio"] = level_stats["Aggregate"] / level_stats["Aggregate"].sum() * 100
        level_stats.insert(0, "Geography", level_stats.index)
        level_stats.to_csv(output_path + '\\level_stats.csv',  index=False)
        
        # Overall Calc
        predicted = sum_support_optimized.sum(axis=1)
        r2, dw, mape, sumResidualSq, dep_agg = calc_r2_dw_mape(sum_dependent, predicted)
        r2_dw_mape = pd.DataFrame([[r2,dw,mape]], columns=["R-Square", "DW", "MAPE"])
        r2_dw_mape.to_csv(output_path + '/RDM.csv',  index=False)
        print(r2_dw_mape)
        
        #Main Result
        all_cols = col_new + ['Predicted', dep_name]
        final_file = np.c_[sum_support_optimized, predicted, sum_dependent]
        final_file = pd.DataFrame(final_file, columns=all_cols)
        final_file.insert(0, period_s.name, period_s)
        final_file.insert(1, "Geography", ["Total"] * len(final_file))#final_file.insert(1, spec["config"]["geo_colname"], ["Total"] * len(final_file))#we need spec here
        final_file.to_csv(output_path + '\\result.csv',  index=False)
        communicator("DONE|" + str(time.time() - startTime), status_path)
        return "SUCCESS"
    except Exception as e:   
        msg = "MergeL-ERROR|" + str(e) + "\n\nDev Details\n===========\n" + traceback.format_exc()
        communicator(msg, status_path)
        # return JsonResponse({"message": str(e)}, status=400) 

def bounds_calc_for_refresh(contribution_range, transformed_support_agg, dependent_agg, allvar_support_colnames, periods):
    reverse_period = {}
    system_unbound = []
    for i in range(len(periods)):
        reverse_period[periods[i]] = i
    transformed_support = np.array(transformed_support_agg)

    # if actual for a period is 0, look for a period where dependent and support is not zero, else it should be sign contraint else with desired volume
    for col_index in range(len(allvar_support_colnames)):
        
        col = allvar_support_colnames[col_index]
        selected_period = contribution_range[col]["contri_period"]
        if not selected_period or selected_period == "FIXED":
            continue # all zeros

        pos = reverse_period[selected_period]

        #We need support here
        # if dependent_agg[pos] == 0: # look for a period where dependent and support is not zero .. starting from highest support, if this fails this condition remails -> dependent_agg[pos] == 0
        #     supports = support_abs[:,col_index]
        #     support_map = supports.tolist()
        #     for s in np.sort(supports)[::-1]: # Decending sort
        #         if s >= agg_constraint_cutoff:
        #             check_pos = support_map.index(s)
        #             if dependent_agg[check_pos] != 0:
        #                 pos = check_pos
        #                 break
        #         else:
        #             break

        dep_agg_period_value = dependent_agg[pos]
        value_transformed = transformed_support[pos][col_index]
        if dep_agg_period_value == 0 or value_transformed == 0:
            system_unbound.append({"c": col, "r": "T2|DEP/SUP0"})
            l = contribution_range[col]["min"]
            u = contribution_range[col]["max"]
            if (l*u >= 0):
                if l + u > 0:
                    contribution_range[col]["sign"] = "POS"
                else :
                    contribution_range[col]["sign"] = "NEG"
            # contribution_range[col]["contri_period"] = False
        else:            
            contribution_range[col]["min"] *= (dep_agg_period_value / value_transformed)
            contribution_range[col]["max"] *= (dep_agg_period_value / value_transformed)
    bounds = []
    scale_vector = np.ones(len(allvar_support_colnames))
    for col_index in range(len(allvar_support_colnames)):
        col = allvar_support_colnames[col_index]
        selected_period = contribution_range[col]["contri_period"]
        # if(col == "OOH_SPD"):
        #     print(contribution_range[col]["sign"])
        #     print(contribution_range[col]["contri_period"])
        #     print(contribution_range[col]["min"])
        #     print(contribution_range[col]["max"])
        #     print(sum_support_agg)
        # exit()
        if not selected_period:
            sign = contribution_range[col]["sign"]
            sum_support = sum_support_agg[col_index]
            scale_vector[col_index] = 0
            if sign == "POS":
                if sum_support >= 0:
                    bounds.append([0, np.inf])
                else:
                    bounds.append([-np.inf, 0])
            elif sign == "NEG":
                if sum_support >= 0:
                    bounds.append([-np.inf, 0])
                else:
                    bounds.append([0, np.inf])
            else:
                bounds.append([-np.inf, np.inf])
        elif selected_period == "FIXED":
            bounds.append([1, 1])
        else:
            pos = reverse_period[selected_period]
            # value_transformed = transformed_support[pos][col_index]
            l = contribution_range[col]["min"]
            u = contribution_range[col]["max"]
            # print(u - l)
            
            scale_vector[col_index] = u - l
            if(l > u):
                l,u=u,l
            bounds.append([l, u])
        # if(col == "ET2_MUL_BRN"):
    # print(bounds[19])
    # print('scale_vector: ', scale_vector)
    # print('bounds: ', bounds)
    # print('scale_vector: ', scale_vector)
    # exit()
    return bounds , scale_vector, system_unbound               

def read_randomization_mapping(randomization_mapping_path, cols_needed):
    data = read_as_dataFrame(randomization_mapping_path, cols_needed)[cols_needed].to_numpy().T#pd.read_csv(randomization_mapping_path, low_memory=False, memory_map=True, engine='c')[cols_needed].to_numpy().T
    mapping_dict = {}
    for i, j in cols_needed.items():
        if j !=0:
            d = dict(zip(data[0], data[j])) # one to one mapping
            mapping_dict[i] = d
    return mapping_dict


if __name__ == "__main__":
    
    # rapid_refresh("D:\\MODEL\\old_test_new", "D:\\MODEL\\old_test_newer", "D:\\MODEL\\old_test_merge")
    cpu = 6
    runList = [{
        "ads": r"C:\Users\Samujit.Das\Desktop\S3_test\Core AWS MP\v1.csv",
        "spec": r"C:\Users\Samujit.Das\Desktop\S3_test\Core AWS MP\v1.xlsx",
        "output": r"C:\Users\Samujit.Das\Desktop\S3_test\Core AWS MP\new"
    },
    ]

    for model in runList[0:1]:
        # Convert XLSX to JSON
        # try:
        json_spec_path = spec_xlsx_to_json(model["spec"])
        # except Exception:
        #     print(traceback.format_exc()),
        #     print("WORKS!")
        #     exit()
        # split_gen_template(model["ads"], model["spec"], model["output"])
        # gen_report(model["ads"], model["spec"], model["output"], "TEST")
        run_model_initiator(model["ads"], model["spec"], model["output"], cpu_count = cpu, sample=0, use_store=False, fast_optimize=False, resume=False, granular_result=True, granular_stats=True, granular_support=True, granular_correlation=False, granular_vol_agg=True, geolevel_contraint=True)

# if __name__ == "__main__":
#     old_path = 'D:\\MODEL\\BumbleB Workspace\\OLD'
#     new_path = 'D:\\MODEL\\BumbleB Workspace\\ALL'
#     output_path = 'D:\\MODEL\\BumbleB Workspace\\NEW'
#     rapid_refresh(old_path, new_path, output_path)
