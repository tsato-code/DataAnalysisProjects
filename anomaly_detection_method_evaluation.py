# ------------------
# モジュールインポート
# ------------------


# ------------------
# 定数パラメータ設定
# ------------------
IN_FILE = "../data/SkillCraft1_Dataset.csv"
OUT_DIR = "../out"


# ------------------
# 関数の定義
# ------------------
def break_even(a, y_test):
    """ 分岐点精度の計算 """
    # 正常標本精度と異常標本精度の計算
    y_test.reset_index(drop=True, inplace=True) # インデックスリセット
    idx = a.argsort()[::-1] # 降順のインデックス計算
    n_total = len(y_test)
    n_anom = sum(y_test)
    n_norm = n_total - n_anom
    coverage = np.zeros(n_total) # 異常標本精度
    detection = np.zeros(n_total) # 正常標本精度
    for i in range(n_total):
        n_detected_anom = sum(y_test[idx][:i])
        n_detected_norm = n_total - i - sum(y_test[idx][i:])
        coverage[i] = n_detected_anom / n_anom
        detection[i] = n_detected_norm / n_norm

    # 分岐点精度の計算
    thresh = 0
    for i, (c_score, d_score) in enumerate(zip(coverage, detection)):
        if c_score >= d_score:
            thresh = i
            break
    break_even_point = a[idx][thresh]
    print(break_even_point, c_score, d_score)
    return (break_even_point, c_score), (coverage, detection)


def hold_out(X, y):
	# ホールドアウト法による精度評価
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=47)

# ------------------
#  メイン処理
# ------------------
def main():
	# ファイル読み込み
	df = pd.read_csv(IN_FILE, header=0, index_col=None, sep=',')
	
	# 前処理
	df = df[(df=='?').sum(axis=1)==0] # '?'を含む行を削除
	df = df.sample(frac=1, random_state=0) # 行シャッフル
	df = df.reset_index(drop=True) # インデックスの更新

	# データセットの作成
	target_col = 'LeagueIndex'
	target = 1
	del_col = [target_col, 'GameID']
	X = df.drop(del_col, axis=1)
	y = (df[target_col]==target).astype(np.int32)
	print(X.shape)










if __name__ == '__main__':
	main()