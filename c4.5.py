import pandas as pd
import math
from typing import Any, Dict, List
from graphviz import Digraph


# 定义计算熵、条件熵、信息增益、增益率以及选择最优划分属性的函数

# 计算数据集D的熵
def calc_entropy(D: pd.DataFrame) -> float:
    """
    计算数据集D的熵
    :param D: 数据集
    :return: 数据集D的熵
    """
    labels = D.iloc[:, -1]
    label_counts = labels.value_counts()
    entropy = -sum((count / len(D)) * math.log2(count / len(D)) for count in label_counts)
    return entropy


# 计算属性A的分裂信息（IV）
def calc_conditional_entropy(D: pd.DataFrame, A: str) -> float:
    """
    计算属性A的条件熵
    :param D: 数据集
    :param A: 属性
    :return: 属性A的分裂信息
    """
    A_values = D[A].unique()
    conditional_entropy = 0.0
    for v in A_values:
        sub_D = D[D[A] == v]
        conditional_entropy += (len(sub_D) / len(D)) * calc_entropy(sub_D)
    return conditional_entropy


# 计算连续属性在给定划分点下的信息增益
def calc_information_gain_continuous(D: pd.DataFrame, attribute: str, split_point: float) -> float:
    """
    计算连续属性在给定划分点下的信息增益
    :param D: 数据集
    :param attribute: 属性名称
    :param split_point: 划分点
    :return: 信息增益
    """
    # 按照划分点分割数据集
    D1 = D[D[attribute] <= split_point]
    D2 = D[D[attribute] > split_point]

    # 计算原数据集的熵
    entropy_before = calc_entropy(D)

    # 计算划分后的加权熵
    entropy_after = (len(D1) / len(D)) * calc_entropy(D1) + \
                    (len(D2) / len(D)) * calc_entropy(D2)

    # 计算信息增益
    information_gain = entropy_before - entropy_after
    return information_gain


# 计算信息增益，自动处理连续和分类属性
def calc_information_gain(D: pd.DataFrame, A: str, split_point: float = None) -> float:
    """
    计算信息增益，自动处理连续和分类属性。连续属性需要划分点。
    :param D: 数据集
    :param A: 属性
    :param split_point: 连续属性的划分点，对于分类属性，这个参数不使用
    :return: 属性A的信息增益
    """
    # 检查属性A是连续还是分类的
    if pd.api.types.is_numeric_dtype(D[A]) and split_point is not None:
        # 对于连续属性，使用划分点来计算信息增益
        return calc_information_gain_continuous(D, A, split_point)
    else:
        # 对于分类属性，使用原始的信息增益计算方法
        return calc_entropy(D) - calc_conditional_entropy(D, A)


# 计算属性A的增益率
def calc_gain_ratio(D: pd.DataFrame, A: str) -> float:
    """
    计算属性A的增益率
    :param D: 数据集
    :param A: 属性
    :return: 属性A的增益率
    """
    information_gain: float = calc_information_gain(D, A)
    A_values: list = D[A].unique()
    iv: float = -sum((len(D[D[A] == v]) / len(D)) * math.log2(len(D[D[A] == v]) / len(D)) for v in A_values)
    return information_gain / iv if iv != 0 else 0.0


# 选择最优划分属性
def choose_best_feature(D: pd.DataFrame) -> str:
    """
    选择最优划分属性
    :param D: 数据集
    :return: 最优划分属性
    """
    features = D.columns[:-1]
    gain_ratios = {feature: calc_gain_ratio(D, feature) for feature in features}
    return max(gain_ratios, key=gain_ratios.get)


# 寻找连续属性的最优划分点
def find_best_split_point_for_continuous_attribute(D: pd.DataFrame, attribute: str) -> float:
    """
    寻找连续属性的最优划分点
    :param D: 数据集
    :param attribute: 属性名称
    :return: 最优划分点
    """
    sorted_values = D[attribute].sort_values().unique()
    split_points = [(sorted_values[i] + sorted_values[i + 1]) / 2 for i in range(len(sorted_values) - 1)]
    max_gain = -float('inf')
    best_split = None
    for split_point in split_points:
        # 计算每个可能划分点的信息增益
        gain = calc_information_gain_continuous(D, attribute, split_point)
        if gain > max_gain:
            max_gain = gain
            best_split = split_point
    return best_split


# 递归构建决策树（不包含剪枝）
def create_decision_tree(D: pd.DataFrame, features: List[str], continuous_attributes: List[str]) -> dict:
    """
    递归构建决策树，适应连续变量处理
    :param D: 数据集
    :param features: 特征列表
    :param continuous_attributes: 连续属性列表
    :return: 决策树
    """
    # 基本终止条件
    class_counts = D.iloc[:, -1].value_counts()
    if len(class_counts) == 1:
        return class_counts.index[0]
    if not features:
        return class_counts.idxmax()

    # 选择最优属性及其划分点（如果是连续的）
    best_gain = -float('inf')
    best_feature = None
    split_point = None
    for feature in features:
        if feature in continuous_attributes:
            # 对于连续属性，找到最优划分点
            point = find_best_split_point_for_continuous_attribute(D, feature)
            gain = calc_information_gain(D, feature, point)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                split_point = point
        else:
            # 对于离散属性，正常处理
            gain = calc_information_gain(D, feature)  # 离散属性的信息增益计算
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

    # 根据选择的最优属性分割数据集
    tree = {best_feature: {}}
    if best_feature in continuous_attributes:
        # 处理连续属性的分割
        left_D = D[D[best_feature] <= split_point]
        right_D = D[D[best_feature] > split_point]
        tree[best_feature]['≤' + str(split_point)] = create_decision_tree(left_D,
                                                                          [f for f in features if f != best_feature],
                                                                          continuous_attributes)
        tree[best_feature]['>' + str(split_point)] = create_decision_tree(right_D,
                                                                          [f for f in features if f != best_feature],
                                                                          continuous_attributes)
    else:
        # 处理离散属性的分割
        for value in D[best_feature].unique():
            sub_D = D[D[best_feature] == value]
            tree[best_feature][value] = create_decision_tree(sub_D, [f for f in features if f != best_feature],
                                                             continuous_attributes)

    return tree


# 后剪枝函数
def post_pruning(tree: Dict[str, Any], D: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    """
    对决策树进行后剪枝
    :param tree: 当前决策树
    :param D: 数据集
    :param features: 特征集
    :return: 剪枝后的决策树
    """
    # 检查树是否是叶节点
    if not isinstance(tree, dict):
        return tree

    # 遍历树中的每个节点
    for feature, branches in tree.items():
        for value, subtree in branches.items():
            # 递归剪枝子树
            subtree = post_pruning(subtree, D[D[feature] == value], [f for f in features if f != feature])
            tree[feature][value] = subtree

    # 尝试剪枝当前节点
    if all(not isinstance(subtree, dict) for subtree in tree[feature].values()):
        # 计算剪枝前后的准确性
        accuracy_before_pruning = calc_accuracy(tree, D)
        # 将当前节点替换为最常见的类
        most_common_class = D.iloc[:, -1].mode()[0]
        pruned_tree = most_common_class
        accuracy_after_pruning = calc_accuracy(pruned_tree, D)
        # 如果剪枝后准确性不降低，则进行剪枝
        if accuracy_after_pruning >= accuracy_before_pruning:
            return pruned_tree

    return tree


# 对单个实例进行预测
def predict(tree: Dict[str, Any], instance: Dict[str, Any]) -> Any:
    """
    对单个实例进行预测
    :param tree: 决策树
    :param instance: 单个数据实例
    :return: 预测结果
    """
    if not isinstance(tree, dict):
        return tree
    root = next(iter(tree))
    subtree = tree[root]
    value = instance[root]
    if value in subtree:
        return predict(subtree[value], instance)
    else:
        return None


# 计算决策树在数据集D上的准确性
def calc_accuracy(tree: Dict[str, Any], D: pd.DataFrame) -> float:
    """
    计算决策树在数据集D上的准确性
    :param tree: 决策树
    :param D: 数据集
    :return: 准确性
    """
    correct_predictions = 0
    for _, row in D.iterrows():
        if predict(tree, row) == row.iloc[-1]:
            correct_predictions += 1
    return correct_predictions / len(D)


# 绘制决策树
def plot_decision_tree(tree, parent_name=None, edge=None, graph=None):
    if graph is None:
        graph = Digraph(comment='Decision Tree', format='png')

    if not isinstance(tree, dict):
        node_name = f"Leaf_{tree}"
        graph.node(node_name, label=str(tree), shape='ellipse')
        if parent_name is not None:
            graph.edge(parent_name, node_name, label=str(edge))
    else:
        for idx, (feature, branches) in enumerate(tree.items()):
            node_name = f"Node_{feature}_{idx}"
            if parent_name is None:
                graph.node(node_name, label=str(feature))
            else:
                graph.edge(parent_name, node_name, label=str(edge))
                graph.node(node_name, label=str(feature))

            for value, subtree in branches.items():
                plot_decision_tree(subtree, node_name, value, graph)

    return graph


# 载入数据
data = dict(
    色泽=['青绿', '乌黑', '乌黑', '青绿', '浅白', '青绿', '乌黑', '乌黑', '青绿', '浅白', '浅白', '青绿', '浅白',
        '乌黑', '浅白', '青绿'],
    根蒂=['蜷缩', '蜷缩', '蜷缩', '蜷缩', '蜷缩', '稍蜷', '稍蜷', '稍蜷', '硬挺', '硬挺', '蜷缩', '稍蜷', '稍蜷',
        '稍蜷', '蜷缩', '蜷缩'],
    敲声=['浊响', '沉闷', '浊响', '沉闷', '浊响', '浊响', '浊响', '浊响', '清脆', '清脆', '浊响', '浊响', '沉闷',
        '浊响', '浊响', '沉闷'],
    纹理=['清晰', '清晰', '清晰', '清晰', '清晰', '清晰', '稍糊', '清晰', '清晰', '模糊', '模糊', '稍糊', '稍糊',
        '清晰', '模糊', '稍糊'],
    脐部=['凹陷', '凹陷', '凹陷', '凹陷', '凹陷', '稍凹', '稍凹', '稍凹', '平坦', '平坦', '平坦', '凹陷', '凹陷',
        '稍凹', '平坦', '稍凹'],
    触感=['硬滑', '硬滑', '硬滑', '硬滑', '硬滑', '软粘', '软粘', '硬滑', '软粘', '硬滑', '软粘', '硬滑', '硬滑',
        '软粘', '硬滑', '硬滑'],
    含糖率=[0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370, 0.042, 0.103],
    好瓜=['是', '是', '是', '是', '是', '是', '是', '是', '否', '否', '否', '否', '否', '否', '否', '否']
)

# 创建数据集
df = pd.DataFrame(data)

# 特征列表
features = list(df.columns[:-1])

# 连续属性列表
continuous_attributes = ['含糖率']

# 创建决策树
decision_tree = create_decision_tree(df, features, continuous_attributes)

# 输出剪枝前决策树
graph_decision_tree = plot_decision_tree(decision_tree)

# 对决策树进行后剪枝
pruned_tree = post_pruning(decision_tree, df, features)

# 输出剪枝后决策树
graph_pruned_tree = plot_decision_tree(pruned_tree)

# 保存并显示图像
graph_pruned_tree.render(filename='pruned_tree', directory='.', view=True)  # 将文件保存在当前工作目录
graph_decision_tree.render(filename='decision_tree', directory='.', view=True)  # 将文件保存在当前工作目录
