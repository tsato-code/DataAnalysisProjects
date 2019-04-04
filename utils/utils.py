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


def _target_mean_encoding(df, src_col, tgt_col, prfx, conn):
    """
    target mean encoding
    target が数値のときに利用できる
    """
    target_mean = df.groupby(src_col)[tgt_col].mean().astype("float16")
    df = df[src_col].map(target_mean).copy()
    df.name = prfx+str(src_col)+conn+str(tgt_col)+"_target_mean"
    return df


def target_mean_encoding(df, tgt_col, prfx="@", conn="_"):
    """
    target mean encoding
    すべてのカテゴリ変数に target mean encoding
    """
    src_col = df.select_dtypes("O").columns.tolist()
    if tgt_col in src_col: src_col.remove(tgt_col)
    df = pd.concat([ _target_mean_encoding(df, c, tgt_col, prfx, conn) for c in src_col ], axis=1)
    return df


def _bin_counting(df, src_col, tgt_col, prfx, conn):
    """
    bin counting
    target がカテゴリのときに利用できる
    """
    a = df[src_col]
    b = df[tgt_col]
    cross_df         = pd.crosstab(a, b, rownames=["a"], colnames=["b"], normalize="index")
    labels           = [prfx+str(src_col)+conn+str(tgt_col)+conn+str(col)+"_bin_counting" for col in cross_df.columns]
    cross_df.columns = labels
    df  = pd.merge(df, cross_df, left_on=src_col, right_index=True, how="left")
    return df[labels]


def bin_counting(df, tgt_col, prfx="@", conn="_"):
    """
    bin counting
    すべてのカテゴリ変数と tgt_col に bin counting
    """
    src_col = df.select_dtypes('O').columns.tolist()
    if tgt_col in src_col: src_col.remove(tgt_col)
    df = pd.concat([ _bin_counting(df, c, tgt_col, prfx, conn) for c in src_col ], axis=1)
    return df


# バックオフ：少数派カテゴリ値を抽出
def _backoff_cand(df, cutoff=0.01):
    col = df.select_dtypes('O').columns.tolist()
    cand = {c: [idx for idx, val in df[c].value_counts(normalize=True, dropna=False).iteritems() if val < cutoff] for c in col}
    cand = {key: val for key, val in cand.items() if val!=[]} # 空リストを削除
    return cand


# バックオフ：少数派カテゴリ値をダミーで置き換え
def _backoff_replace(df, cand, repl, prfx, rep):
    _df = pd.DataFrame({})
    col = []
    for key, val in cand.items():
        tmp = df[key]
        col_name = prfx+str(key)
        for idx in val:
            tmp = tmp.replace(idx, rep)
        _df[col_name] = tmp
        col.append(col_name)
    return _df


def backoff(df, prfx="@", rep="@backoff", repl=False):
    cand = _backoff_cand(df)
    _df = _backoff_replace(df, cand, repl, prfx, rep)
    if repl:
        for key, val in cand.items():
            df[key] = _df[prfx+str(key)]
        return df
    else:
        return pd.concat([df, _df], axis=1)


def send_line_notification(message, png=None):
    line_notify_token = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    if png is None:
        requests.post(line_notify_api, data=payload, headers=headers)
    else:
        files = {"imageFile": open(png, "rb")}
        requests.post(line_notify_api, data=payload, headers=headers, files=files)
    print(message)


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
