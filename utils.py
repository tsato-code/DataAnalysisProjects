import feather
import logging
import os
import platform
import sys
import numpy as np
import pandas as pd
from contextlib import contextmanager
from datetime import datetime
from glob import glob
from time import time, sleep


def start(fname):
    global st_time
    st_time = time()
    print("="*80)
    print("DATE: {}".format(datetime.now()))
    print("FILE: {}".format(fname))
    print("PID: {}".format(os.getpid()))
    print("HOST: {}".format(platform.node()))
    print("ENV: {}".format(platform.platform()))
    print("="*80)


def end():
    print("="*80)
    print("ELAPSED: {:.2f} (sec)".format(elapsed_time()))
    print("="*80)


def elapsed_time():
    return (time() - st_time)


def reset_time():
    global st_time
    st_time = time()


def to_feather(df, path="./"):
    df.reset_index(inplace=True, drop=True)
    os.makedirs(path, exist_ok=True)
    print(f"write to {path}")
    for c in df.columns:
        path_file = os.path.join(path, f"{c}.f")
        if not glob(path_file):
            df[[c]].to_feather(path_file)
        else:
            print(f"WARNIG: {path_file} is exists!")
            sys.exit()

            
def read_feather(path="./", col=None):
    if col is None:
        path_file = os.path.join(path, "*.f")
        print(f"read {path_file}")
        df = pd.concat([ feather.read_dataframe(f) for f in sorted(glob(path_file)) ], axis=1)
    else:
        path_file = os.path.join(path, col)
        if not glob(path_file):
            print(f"read {path_file}")
            df = feather.read_dataframe(path_file)
        else:
            print(f"WARNIG: {path_file} is exists!")
            sys.exit()
    return df


def to_pickles(df, path="./"):
    df.reset_index(inplace=True, drop=True)
    os.makedirs(path, exist_ok=True)
    print(f"write to {path}")
    for c in df.columns:
        path_file = os.path.join(path, f"{c}.f")
        if not glob(path_file):
            df[[c]].to_pickle(path_file)
        else:
            print(f"WARNIG: {path_file} is exists!")
            sys.exit()


def read_pickles(path="./", col=None):
    if col is None:
        path_file = os.path.join(path, "*.f")
        print(f"read {path_file}")
        df = pd.concat([ pd.read_pickle(f) for f in sorted(glob(path_file)) ], axis=1)
    else:
        path_file = os.path.join(path, col)
        if not glob(path_file):
            print(f"read {path_file}")
            df = pd.read_pickle(path_file)
        else:
            print(f"WARNIG: {path_file} is exists!")
            sys.exit()
    return df


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


@contextmanager
def timer(name, logger=None, level=logging.DEBUG):
    print_ = print if logger is None else lambda msg: logger.log(level, msg)
    t0 = time()
    print_(f'[{name}] start')
    yield
    print_(f'[{name}] done in {time() - t0:.0f} s')


def get_dummies(df):
    col = df.select_dtypes('O').columns.tolist()
    nunique = df[col].nunique()
    col_binary = nunique[nunique==2].index.tolist()
    [col.remove(c) for c in col_binary]
    df = pd.get_dummies(df, columns=col)
    df = pd.get_dummies(df, columns=col_binary, drop_first=True)
    return df


if __name__ == "__main__":
	start("utils.py")
	with timer("create"):
		df = pd.DataFrame({"a":[1,2,3,4,5], "b":[6,7,8,9,10], "c":["a","b","c","d","e"]})
		print(df)
	# with timer("to_feather"):
	#	to_feather(df, "new_folder00")
	#	df00 = read_feather("new_folder00")
	#	print(f00)
	#with timer("to_pickle"):
	#	to_pickles(df, "new_folder01")
	#	df01 = read_pickles("new_folder01")
	#	print(df01)
	with timer("reduce_mem_usage"):
	    df = reduce_mem_usage(df)
	with timer("dummy"):
		df = get_dummies(df)
		print(df)
	end()
