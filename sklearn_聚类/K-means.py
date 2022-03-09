import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 让jupyternotebook直接在单元格内显示生成的图形
# %matplotlib inline
# 解决matplotlib在windows电脑上中文乱码问题
plt.rcParams['font.sans-serif'] = 'SimHei'
# 解决matplotlib负号无法显示的问题
plt.rcParams['axes.unicode_minus'] = False

# 让图形变成矢量形式，显示更清晰
# %config InlineBackend.figure_format='svg'

def data_pre(path):
    df = pd.read_csv(path)
    print(df.columns)
    data = df[['得分', '命中率', '三分命中率', '罚球命中率', '场次', '上场时间']]
    data['出手数量'] = df['命中-出手'].apply(lambda x: x.split('-')[1])
    data['罚球出手数量'] = df['命中-罚球'].apply(lambda x: x.split('-')[1])
    data = data.astype(np.float)
    print(data)
    return data


def Silhouette_Coefficient(data):
    for i in range(2, 10):
        model = KMeans(n_clusters=i)
        s = model.fit(data)
        y_pre = s.labels_
        ans = silhouette_score(data, y_pre)
        print(i, ans, sep='⑧⑧⑧⑧⑧')


def k_SSE(data):
    TSSE = []
    K = range(2, 10)
    for i in range(2, 10):
        SSE = []
        model = KMeans(n_clusters=i)
        s = model.fit(data)
        # 返回簇标签
        labels = s.labels_
        # 返回簇中心
        centers = s.cluster_centers_
        # 计算各簇样本的离差平方和，并保存到列表中
        for label in set(labels):
            SSE.append(np.sum((data.loc[labels == label,] - centers[label, :]) ** 2))
        # 计算总的簇内离差平方和
        TSSE.append(np.sum(SSE))
    # 设置绘图风格
    plt.style.use('ggplot')
    # 绘制K的个数与GSSE的关系
    plt.plot(K, TSSE, 'b*-')
    plt.xlabel('簇的个数')
    plt.ylabel('簇内离差平方和之和')
    # 显示图形
    plt.show()

def visualize(data):
    fig = plt.figure(figsize=(12, 12))
    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
    ax.scatter(data[data['cluster'] == 0]['上场时间'], data[data['cluster'] == 0]['出手数量'],
                data[data['cluster'] == 0]['得分'])
    ax.scatter(data[data['cluster'] == 1]['上场时间'], data[data['cluster'] == 1]['出手数量'],
                data[data['cluster'] == 1]['得分'])
    ax.scatter(data[data['cluster'] == 2]['上场时间'], data[data['cluster'] == 2]['出手数量'],
                data[data['cluster'] == 2]['得分'])
    ax.set_xlabel('上场时间')
    ax.set_ylabel('出手数量')
    ax.set_zlabel('得分')
    plt.show()


if __name__ == '__main__':
    path = 'players.csv'
    df = pd.read_csv(path)
    data = data_pre(path=path)

    Silhouette_Coefficient(data)
    k_SSE(data)

    k = 3
    model = KMeans(n_clusters=k)
    model.fit(data)
    # for index, label in enumerate(model.labels_, 1):
    #     print("index:{}⑧⑧⑧⑧⑧label:{}".format(index, label))

    df['cluster'] = model.labels_
    data['cluster'] = model.labels_
    print(data)
    df.to_csv('NBA_players.csv', index=False, encoding='utf_8_sig')

    visualize(data)

