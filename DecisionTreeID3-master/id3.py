import numpy as np
import pandas as pd
import treePlotter

class TreeNode(object):
    def __init__(self, ids=None, children=[], entropy=0, depth=0):
        self.ids = ids  # 此节点中的数据索引
        self.entropy = entropy  # 熵，稍后填充
        self.depth = depth  # 到根节点的距离
        self.split_attribute = None  # 选择哪个属性，它是非叶子的
        self.children = children  # 其子节点列表
        self.order = None  # 孩子中 split_attribute 的值顺序
        self.label = None  # 如果是叶子节点的标签

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label


def entropy(freq):
    # remove prob 0
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0 / float(freq_0.sum())
    return -np.sum(prob_0 * np.log(prob_0))


class DecisionTreeID3(object):
    def __init__(self, max_depth=10, min_samples_split=2, min_gain=1e-4):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.Ntrain = 0
        self.min_gain = min_gain

    def fit(self, data, target):
        self.Ntrain = data.count()[0]
        self.data = data
        self.attributes = list(data)
        self.target = target
        self.labels = target.unique()

        ids = range(self.Ntrain)
        self.root = TreeNode(ids=ids, entropy=self._entropy(ids), depth=0)
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.depth < self.max_depth or node.entropy < self.min_gain:
                node.children = self._split(node)
                if not node.children:  # leaf node
                    self._set_label(node)
                queue += node.children
            else:
                self._set_label(node)

    def _entropy(self, ids):
        # 计算具有索引 id 的节点的熵
        # print('ncaa', len(ids))
        if len(ids) == 0:
            return 0
        ids = [i + 1 for i in ids]  # 熊猫系列索引从1开始
        # print('ids', ids)
        freq = np.array(self.target[ids].value_counts())
        # print('ncaa', self.target[ids].value_counts())
        return entropy(freq)

    def _set_label(self, node):
        # find label for a node if it is a leaf
        # simply chose by major voting
        target_ids = [i + 1 for i in node.ids]  # target is a series variable
        node.set_label(self.target[target_ids].mode()[0])  # most frequent label

    def _split(self, node):
        ids = node.ids
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :]
        for i, att in enumerate(self.attributes):
            values = self.data.iloc[ids, i].unique().tolist()
            if len(values) == 1:
                continue  # entropy = 0
            splits = []
            for val in values:
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                splits.append([sub_id - 1 for sub_id in sub_ids])
            # don't split if a node has too small number of points
            if min(map(len, splits)) < self.min_samples_split:
                continue
            # information gain
            HxS = 0
            for split in splits:
                HxS += len(split) * self._entropy(split) / len(ids)
            gain = node.entropy - HxS
            print(att, '   ⑧⑧⑧⑧⑧⑧⑧⑧   ' , gain,sep='   ')
            if gain < self.min_gain:
                continue  # stop if small gain
            if gain > best_gain:
                best_gain = gain
                best_splits = splits
                best_attribute = att
                order = values
        if best_attribute != None:
            print('信息增益最大的属性为', best_attribute)
        node.set_properties(best_attribute, order)
        child_nodes = [TreeNode(ids=split,
                                entropy=self._entropy(split), depth=node.depth + 1) for split in best_splits]
        return child_nodes

    def predict(self, new_data):
        """
        param new_data: 一个新的数据框，每一行都是一个数据点
        return: 每行的预测标签
        """
        npoints = new_data.count()[0]
        labels = [None] * npoints
        for n in range(npoints):
            x = new_data.iloc[n, :]  # one point
            # 如果没有遇到叶子，则从根开始并递归旅行
            node = self.root
            while node.children:
                node = node.children[node.order.index(x[node.split_attribute])]
            labels[n] = node.label

        return labels


def show_tree(tnode: TreeNode):
    global decs_tree
    flag = True
    if not tnode.children:
        return
    if tnode.split_attribute != None:
        decs_tree += "'{}':{{".format(tnode.split_attribute)
        print(tnode.split_attribute, ':{', sep='', end='')
        ans = tnode.order
        for i in range(len(ans)):
            temp = tnode.children[ans.index(ans[i])]
            decs_tree += "'{}':".format(ans[i])
            print(ans[i], ':', sep=' ', end='')
            if temp.label != None:
                decs_tree += "'" + temp.label + "'"
                print(temp.label,end='')
            else:
                flag = not flag
                decs_tree += '{'
                print('{',end='')
            show_tree(temp)
            if not flag:
                decs_tree += '}'
                print('}',end='')
                flag = True
            if i != len(ans) - 1:
                decs_tree += ','
                print(',',end='')
        decs_tree += '}'
        print('}',end='')
    # if tnode.label != None:


if __name__ == "__main__":
    global decs_tree
    decs_tree = ''
    df = pd.read_csv('weather.csv', index_col=0, parse_dates=True)
    print(df)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    tree = DecisionTreeID3(max_depth=3, min_samples_split=2)
    tree.fit(X, y)
    print(tree.predict(X))

    node = tree.root
    show_tree(node)
    print()
    decs_tree = '{' + decs_tree + '}'
    print(eval(decs_tree))
    treePlotter.ID3_Tree(eval(decs_tree))
