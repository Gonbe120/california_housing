#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def mkdf(california):

# 辞書型のcaliforniaの'data'というキーを用いてDataFrameを作成し、californiaの'feature_names'キーの要素を列名に代入
    df = pd.DataFrame(california.data,columns = california['feature_names'])

# dfに新しく'Price'キーを追加し、californiaの'target'キーの要素を代入
    df['Price'] = california['target']

# データの分布を確認、dfをヒストグラムにし、４８つの棒で表現し、図の大きさを横１４，縦１０を表示
    
# axes = df.hist(bins = 48,figsize = (14,10))

# dfの'Price'の最大値未満のデータのみ取得
    df = df[df['Price']<df['Price'].max()]
    return df

def sdz():
# 標準化
    sds = StandardScaler()
    x_training = sds.fit_transform(x_training)
    x_test = sds.transform(x_test)
    
    y_training = sds.fit_transform(y_training)
    y_test = sds.transform(y_test)

def learn_model(model):
# modelをx_training,y_trainingに適合させる
    model.fit(x_training,y_training)

def display_fig():
#図の表示
    
# 平均二乗偏差を計算
    rmse = mean_squared_error(y_test,model.predict(x_test),squared = False)
# 図の大きさを横５，縦５として表示
    plt.figure(figsize = (5,5))
# x軸の表示範囲を設定
    plt.xlim(-3,4)
# y軸の表示範囲を設定
    plt.ylim(-3,4)
# 二次元に(xのテストデータから得たyの予測値,yのtestデータ)をplotの種類を丸としてグラフ作成
    plt.plot(model.predict(x_test),y_test,'o')
# タイトルの追加
    plt.title('RMSE:{:.3f}'.format(rmse))
# x軸のラベル名
    plt.xlabel('Predict')
# Y軸のラベル名
    plt.ylabel('Actual')
# グラフの背景に格子を描く
    plt.grid()
# 図の表示
    plt.show()
    dirname = './notebook/data/output/'
    filename = dirname + 'img.png'
    plt.savefig(filename)
    print('end')

if __name__=='__main__':
# californiaにfetch_california_housingを代入し、データフレームを作成
    california = fetch_california_housing()
    df = mkdf(california)
# dfからx,yを定義
    x = df.drop(['Price'],axis = 1)
    y = df['Price'] 
# randaom_state = 42で訓練データとテストデータを作成
    x_training,x_test,y_training,y_test = train_test_split(x,y,test_size=0.1,random_state = 42)
# 目的変数を標準化するためにy_training,y_testを二次元配列に変換
    y_training = y_training.values.reshape(-1,1)
    y_test = y_test.values.reshape(-1,1)
# 標準化
    sds = StandardScaler()
    x_training = sds.fit_transform(x_training)
    x_test = sds.transform(x_test)
    
    y_training = sds.fit_transform(y_training)
    y_test = sds.transform(y_test)
# modelにLinearRegressionを代入
    model = LinearRegression(fit_intercept = False)
# 学習 
    learn_model(model)
# 図の表示
    display_fig()




