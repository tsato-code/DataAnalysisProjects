import linecache


file_name = "data/amzn-anon-access-samples/amzn-anon-access-samples-2.0.csv"


# ファイルの5行目だけを読み込む
a = 20000
target_line = linecache.getline(file_name, int(a))
print(target_line)


# 読み込んだデータを削除
linecache.clearcache() 