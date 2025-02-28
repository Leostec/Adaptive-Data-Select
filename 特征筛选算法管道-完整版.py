import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
import shap
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, kendalltau
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import community
import random
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import community
from sklearn.metrics import mean_squared_error, r2_score
from deap import base, creator, tools, algorithms
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import csv

def select_data(X_train, y_train, X_test, y_test, num, model, task_type):
    feature_importances = []

    # 模型初始化
    if model == 'XGB':
        model_class = xgb.XGBRegressor if task_type == 'regression' else xgb.XGBClassifier
        model = model_class()
        model.fit(X_train, y_train)

    # 工具函数：计算并排序特征重要性
    def get_feature_importance(method, importance):
        importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        return importance_df['Feature'].tolist()

    # 1. XGBoost内置特征重要性
    feature_importances.append(['XGBoost'] + get_feature_importance('XGBoost', model.feature_importances_))

    # 2. 排列重要性 PermutationImportance
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=1,
                                    scoring='neg_mean_squared_error' if task_type == 'regression' else 'accuracy')
    feature_importances.append(['PermutationImportance'] + get_feature_importance('PermutationImportance', -result.importances_mean))

    # 3. 相关性分析
    correlations = X_train.corrwith(y_train).abs()
    feature_importances.append(['Correlation'] + get_feature_importance('Correlation', correlations))

    # 4. 递归特征消除 Recursive Feature Elimination
    rfe = RFE(model, n_features_to_select=num)
    rfe.fit(X_train, y_train)
    feature_importances.append(['RFE'] + get_feature_importance('RFE', rfe.ranking_))

    # 5. 随机森林内置特征重要性
    rf_class = RandomForestRegressor if task_type == 'regression' else RandomForestClassifier
    rf = rf_class(n_estimators=100, random_state=1)
    rf.fit(X_train, y_train)
    feature_importances.append(['RandomForest'] + get_feature_importance('RandomForest', rf.feature_importances_))

    # 6. 主成分分析
    pca = PCA()
    pca.fit(X_train)
    feature_importances.append(['PCA'] + get_feature_importance('PCA', pca.explained_variance_ratio_))

    # 7. 互信息
    if task_type == 'regression':
        mi = mutual_info_regression(X_train, y_train)
    else:
        mi = mutual_info_classif(X_train, y_train)
    feature_importances.append(['Mutual Information'] + get_feature_importance('Mutual Information', mi))

    # 8. 决策树内置特征重要性
    dt_class = DecisionTreeRegressor if task_type == 'regression' else DecisionTreeClassifier
    dt = dt_class(random_state=42)
    dt.fit(X_train, y_train)
    feature_importances.append(['Decision Tree'] + get_feature_importance('Decision Tree', dt.feature_importances_))

    # 9. 方差分析 ANOVA
    fval = f_classif(X_train, y_train)
    feature_importances.append(['ANOVA'] + get_feature_importance('ANOVA', fval[0]))

    # 10. Shapley 值
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_train)
    mean_shap_values = shap_values.mean(axis=0)
    feature_importances.append(['Shapley'] + get_feature_importance('Shapley', mean_shap_values))

    # 11. 信息增益
    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities))

    def conditional_entropy(feature, labels):
        unique_values = np.unique(feature)
        entropy_total = 0
        for value in unique_values:
            subset_labels = labels[feature == value]
            entropy_total += (len(subset_labels) / len(labels)) * entropy(subset_labels)
        return entropy_total

    def information_gain(feature, labels):
        return entropy(labels) - conditional_entropy(feature, labels)

    feature_importance_info = []
    for feature_name in X_train.columns:
        ig = information_gain(X_train[feature_name], y_train)
        feature_importance_info.append({'Feature': feature_name, 'Importance': ig})

    feature_importance_ig_df = pd.DataFrame(feature_importance_info)
    feature_importance_ig_df = feature_importance_ig_df.sort_values(by='Importance', ascending=False)
    feature_importances.append(['Information Gain'] + get_feature_importance('Information Gain', feature_importance_ig_df['Importance']))

    # 12. LDA线性判别分析
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    feature_importances.append(['LDA'] + get_feature_importance('LDA', lda.coef_[0]))

    # 13. 基于遗传算法的特征选择 (Feature Selection via Genetic Algorithm)


    log_reg = LogisticRegression(solver='liblinear')
    log_reg.fit(X_train, y_train)
    selector = SelectFromModel(log_reg, threshold="mean", max_features=num)
    selector.fit(X_train, y_train)
    feature_importances.append(['Genetic Algorithm'] + get_feature_importance('Genetic Algorithm', selector.get_support()))

    # 14. 集成学习模型特征选择
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    gb_class = GradientBoostingRegressor if task_type == 'regression' else GradientBoostingClassifier
    gb = gb_class(n_estimators=100, random_state=1)
    gb.fit(X_train, y_train)
    feature_importances.append(['GradientBoosting'] + get_feature_importance('GradientBoosting', gb.feature_importances_))
    with open('./sorted_feature_importances.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(feature_importances)
    return feature_importances

def spearman_simli():
    # 读取Excel文件，假设方法名称在每行的第一列
    df = pd.read_csv('./sorted_feature_importances.csv', header=None, index_col=0)

    # 获取方法名称列表
    methods = df.index.tolist()

    # 获取标签信息
    labels = df.columns.tolist()

    # 获取特征重要性排序数据
    features_rankings = df.values

    # 初始化相似度矩阵，并用方法名称作为行和列标签
    similarity_matrix = pd.DataFrame(np.zeros((len(methods), len(methods))), index=methods, columns=methods)

    # 计算相似度
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1_ranking = features_rankings[i]
            method2_ranking = features_rankings[j]
            correlation, _ = spearmanr(method1_ranking, method2_ranking)
            similarity_matrix.iloc[i, j] = correlation
            similarity_matrix.iloc[j, i] = correlation

    # 将相似度矩阵标准化为0到1之间
    scaler = MinMaxScaler()
    spearman_similarities_normalized = scaler.fit_transform(similarity_matrix)

    # 创建相似度矩阵DataFrame
    thresholded_matrix = pd.DataFrame(spearman_similarities_normalized, index=methods, columns=methods)

    return thresholded_matrix


def kendall_simli():
    # 读取Excel文件，假设方法名称在每行的第一列
    df = pd.read_csv('./sorted_feature_importances.csv', header=None, index_col=0)

    # 获取方法名称列表
    methods = df.index.tolist()

    # 获取标签信息
    labels = df.columns.tolist()

    # 获取特征重要性排序数据
    features_rankings = df.values

    # 初始化相似度矩阵，并用方法名称作为行和列标签
    similarity_matrix = pd.DataFrame(np.zeros((len(methods), len(methods))), index=methods, columns=methods)

    # 计算相似度
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1_ranking = features_rankings[i]
            method2_ranking = features_rankings[j]
            correlation, _ = kendalltau(method1_ranking, method2_ranking)
            similarity_matrix.iloc[i, j] = correlation
            similarity_matrix.iloc[j, i] = correlation

    # 将相似度矩阵标准化为0到1之间
    scaler = MinMaxScaler()
    kendalltau_similarities_normalized = scaler.fit_transform(similarity_matrix)

    # 创建相似度矩阵DataFrame
    thresholded_matrix = pd.DataFrame(kendalltau_similarities_normalized, index=methods, columns=methods)

    return thresholded_matrix


def cosine_similarity_threshold():
    # 读取Excel文件，假设方法名称在每行的第一列
    df = pd.read_csv('./sorted_feature_importances.csv', header=None, index_col=0)
    # 获取方法名称列表
    methods = df.index.tolist()
    # 获取标签信息
    labels = df.columns.tolist()
    # 获取特征重要性排序数据
    features_rankings = df.values
    # 创建特征名到数字编码的映射字典
    feature_to_code = {feature: i for i, feature in enumerate(features_rankings[0])}
    # 将特征重要性排序数据转换为向量表示
    features_vectors = np.zeros((len(methods), len(labels)))
    for i in range(len(methods)):
        for j in range(len(labels)):
            features_vectors[i, j] = feature_to_code[features_rankings[i, j]]
    # 计算余弦相似度
    cosine_similarities = cosine_similarity(features_vectors)
    # 将相似度矩阵标准化为0到1之间
    scaler = MinMaxScaler()
    cosine_similarities_normalized = scaler.fit_transform(cosine_similarities)
    # 创建相似度矩阵DataFrame
    similarity_matrix = pd.DataFrame(cosine_similarities_normalized, index=methods, columns=methods)
    return similarity_matrix


def custom_similarity():
    # 读取Excel文件，假设方法名称在每行的第一列
    df = pd.read_csv('./sorted_feature_importances.csv', index_col= 0,header=None)
    # 获取方法名称列表
    methods = df.index.tolist()
    # 获取特征重要性排序数据
    features_rankings = df.values

    # 初始化相似度矩阵，并用方法名称作为行和列标签
    similarity_matrix = pd.DataFrame(np.zeros((len(methods), len(methods))), index=methods, columns=methods)
    # 计算相似度
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1_ranking = features_rankings[i]
            method2_ranking = features_rankings[j]
            intersection = sum(1 for idx, (f1, f2) in enumerate(zip(method1_ranking, method2_ranking))
                               if f1 == f2 or (idx > 0 and method1_ranking[idx - 1] == method2_ranking[idx - 1])
                               or (idx < len(method1_ranking) - 1 and method1_ranking[idx + 1] == method2_ranking[
                idx + 1]))
            similarity_matrix.iloc[i, j] = intersection
            similarity_matrix.iloc[j, i] = intersection

    # 将相似度矩阵标准化为0到1之间
    scaler = MinMaxScaler()
    similarity_matrix_normalized = scaler.fit_transform(similarity_matrix)

    # 创建相似度矩阵DataFrame
    thresholded_matrix = pd.DataFrame(similarity_matrix_normalized, index=methods, columns=methods)

    return thresholded_matrix


def borda_count(rankings):
    n = len(rankings)
    score = {}
    for i, feature in enumerate(rankings):
        score[feature] = score.get(feature, 0) + (n - i - 1)
    return score





def construct_adjacency_matrix(lambda_matrix_1, lambda_matrix_2, lambda_matrix_3, lambda_matrix_4, threshold, a, task_type):
    # 获取标准化的相似度矩阵
    custom_similarit = custom_similarity()
    cosine_similarity_threshol = cosine_similarity_threshold()  # 你需要定义或调用这个函数
    kendall_siml = kendall_simli()  # 你需要定义或调用这个函数
    spearman_siml = spearman_simli()  # 你需要定义或调用这个函数

    # 使用lambda矩阵和相似度计算最终相似度矩阵
    final_simli = lambda_matrix_1 * custom_similarit + lambda_matrix_2 * cosine_similarity_threshol + lambda_matrix_3 * kendall_siml + lambda_matrix_4 * spearman_siml

    # 应用阈值，生成二值矩阵
    final_simli = np.where(final_simli > threshold, 1, 0)

    # 转换为DataFrame格式
    final_simli = pd.DataFrame(final_simli)

    # 读取特征排序文件
    df = pd.read_csv('./sorted_feature_importances.csv', index_col=0)
    methods = df.index.tolist()

    # 创建无向图
    G = nx.Graph()

    # 添加节点到图中
    G.add_nodes_from(methods)

    # 根据最终相似度矩阵添加边
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            if final_simli.iloc[i, j] == 1:
                G.add_edge(methods[i], methods[j])

    # 社团划分
    partition = community.best_partition(G)

    # 获取社团划分结果
    communities = set(partition.values())  # 获取所有社团编号
    all_rankings = []

    # 对每个社团进行处理
    for community_id in communities:
        # 获取当前社团中的节点
        selected_methods = [method for method, community in partition.items() if community == community_id]

        if selected_methods:
            # 从 df 中筛选出对应节点的数据
            selected_data = df[df.index.isin(selected_methods)].values

            # 将社团ID与每个节点的排序数据拼接
            for row in selected_data:
                all_rankings.append(np.concatenate(([community_id], row)))

    # 如果只有一个社团，则直接对所有排序进行波达计数
    if len(communities) == 1:
        all_rankings_without_id = [row[1:] for row in all_rankings]  # 排除掉社团ID列
        final_ranking = borda_count(np.concatenate(all_rankings_without_id))  # 你需要定义或调用这个函数
        final_ranking = sorted(final_ranking.items(), key=lambda x: x[1], reverse=True)  # 排序
        # 转换为 DataFrame
        final_global_ranking_df = pd.DataFrame(final_ranking)
        # 删除排名得分列，只保留社团ID和节点ID
        final_ranking = final_global_ranking_df.drop(final_global_ranking_df.columns[1], axis=1)
    else:
        final_ranking = {}  # 用于存储最终汇总的排名
        # 对每个社团分别进行波达计数处理
        for community_id in communities:
            # 获取当前社团的所有排名数据
            community_rankings = [row[1:] for row in all_rankings if row[0] == community_id]  # 排除掉社团ID列
            # 计算该社团内的波达计数
            community_score = borda_count(np.concatenate(community_rankings))  # 你需要定义或调用这个函数
            # 对社团内的波达计数结果进行排序
            community_score = sorted(community_score.items(), key=lambda x: x[1], reverse=True)
            # 将社团的排序结果存入字典
            final_ranking[community_id] = community_score
            # 汇总所有社团的排名并将每个社团的排名数据合并
            all_community_rankings = []
            for community_id, rankings in final_ranking.items():
                # 只保存每个社团的最终排序
                final_community_ranking = [rank for rank, _ in rankings]
                all_community_rankings.append(final_community_ranking)
        # 将所有社团的排名结果进行一次全局波达计数
        final_global_ranking = borda_count(np.concatenate(all_community_rankings))  # 你需要定义或调用这个函数
        # 对全局波达计数结果进行排序
        final_global_ranking = sorted(final_global_ranking.items(), key=lambda x: x[1], reverse=True)
        # 转换为 DataFrame
        final_global_ranking_df = pd.DataFrame(final_global_ranking)
        # 删除排名得分列，只保留社团ID和节点ID
        final_ranking = final_global_ranking_df.drop(final_global_ranking_df.columns[1], axis=1)

    # 加载数据
    X_train, X_test, y_train, y_test = data_load()

    # 选取排名前 a 个特征
    top_features = final_ranking.head(round(a)).iloc[:, 0].tolist()

    # 使用排名前 a 个特征重新构建 X
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]

    if task_type == 'regression':
        # 初始化 XGBRegressor 模型
        model_top = xgb.XGBRegressor()
        model_top.fit(X_train_top, y_train)

        # 使用训练好的模型进行预测
        y_pred_top = model_top.predict(X_test_top)

        # 计算均方误差 (MSE)
        mse_top = mean_squared_error(y_test, y_pred_top)

        # 计算 R²（决定系数）
        r2_top = r2_score(y_test, y_pred_top)

        return mse_top, r2_top
    elif task_type == 'classification':
        # 初始化 XGBClassifier 模型
        model_top = xgb.XGBClassifier()
        model_top.fit(X_train_top, y_train)

        # 使用训练好的模型进行预测
        y_pred_top = model_top.predict(X_test_top)
        y_prob_top = model_top.predict_proba(X_test_top)[:, 1]  # 预测概率，用于AUC计算

        # 计算准确率 (accuracy)
        acc_top = accuracy_score(y_test, y_pred_top)

        # 计算AUC（曲线下面积）
        auc_top = roc_auc_score(y_test, y_prob_top)

        return acc_top, auc_top


# Initialize population
def initialize_population(population_size, num_a):
    population = []
    for _ in range(population_size):
        individual = {
            'lambda_matrix_1': random.uniform(0, 1),
            'lambda_matrix_2': random.uniform(0, 1),
            'lambda_matrix_3': random.uniform(0, 1),
            'lambda_matrix_4': random.uniform(0, 1),
            'threshold': random.uniform(0, 1),
            'a': random.randint(10, num_a)  # Assuming 'a' is an integer in the range [1, 10]
        }
        population.append(individual)
    return population

# Evaluate population
def evaluate_population(population, task_type):
    scores = []
    for individual in population:
        lambda_matrix_1 = individual['lambda_matrix_1']
        lambda_matrix_2 = individual['lambda_matrix_2']
        lambda_matrix_3 = individual['lambda_matrix_3']
        lambda_matrix_4 = individual['lambda_matrix_4']
        threshold = individual['threshold']
        a = individual['a']

        if task_type == 'classification':
            acc, auc = construct_adjacency_matrix(lambda_matrix_1, lambda_matrix_2, lambda_matrix_3,
                                                                 lambda_matrix_4, threshold, a,task_type)
            scores.append((individual, acc, auc))
        elif task_type == 'regression':
            mse, r2 = construct_adjacency_matrix(lambda_matrix_1, lambda_matrix_2, lambda_matrix_3, lambda_matrix_4,
                                                 threshold, a,task_type)
            scores.append((individual, mse, r2))
        else:
            raise ValueError("Invalid task type. Choose either 'classification' or 'regression'.")
    return scores

# Select best individuals
def select_best_individuals(scores, num_best, task_type):
    if task_type == 'classification':
        # 分类任务：最大化 accuracy + AUC
        sorted_population = sorted(scores, key=lambda x: x[1] + x[2], reverse=True)  # 最大化 accuracy + AUC
    elif task_type == 'regression':
        # 回归任务：最小化 MSE
        sorted_population = sorted(scores, key=lambda x: x[1])  # 假设 x[1] 是 MSE
    else:
        raise ValueError("Invalid task type. Choose either 'classification' or 'regression'.")

    best_individuals = [individual for individual, _, _ in sorted_population[:num_best]]
    return best_individuals

# Crossover
def crossover(parent1, parent2):
    child = {
        'lambda_matrix_1': (parent1['lambda_matrix_1'] + parent2['lambda_matrix_1']) / 2,
        'lambda_matrix_2': (parent1['lambda_matrix_2'] + parent2['lambda_matrix_2']) / 2,
        'lambda_matrix_3': (parent1['lambda_matrix_3'] + parent2['lambda_matrix_3']) / 2,
        'lambda_matrix_4': (parent1['lambda_matrix_4'] + parent2['lambda_matrix_4']) / 2,
        'threshold': (parent1['threshold'] + parent2['threshold']) / 2,
        'a': (parent1['a'] + parent2['a']) // 2  # Assuming 'a' is an integer
    }
    return child

# Mutation
def mutation(individual,mutation_rate):

    if random.random() < mutation_rate:
        individual['lambda_matrix_1'] = random.uniform(0, 1)
    if random.random() < mutation_rate:
        individual['lambda_matrix_2'] = random.uniform(0, 1)
    if random.random() < mutation_rate:
        individual['lambda_matrix_3'] = random.uniform(0, 1)
    if random.random() < mutation_rate:
        individual['lambda_matrix_4'] = random.uniform(0, 1)
    if random.random() < mutation_rate:
        individual['threshold'] = random.uniform(0, 1)
    if random.random() < mutation_rate:
        individual['a'] = random.randint(10, 57)
    return individual


def get_final_ranking(lambda_matrix_1, lambda_matrix_2, lambda_matrix_3, lambda_matrix_4, threshold):
    # 获取标准化的相似度矩阵
    custom_similarit = custom_similarity()
    cosine_similarity_threshol = cosine_similarity_threshold()  # 你需要定义或调用这个函数
    kendall_siml = kendall_simli()  # 你需要定义或调用这个函数
    spearman_siml = spearman_simli()  # 你需要定义或调用这个函数

    # 使用lambda矩阵和相似度计算最终相似度矩阵
    final_simli = lambda_matrix_1 * custom_similarit + lambda_matrix_2 * cosine_similarity_threshol + lambda_matrix_3 * kendall_siml + lambda_matrix_4 * spearman_siml

    # 应用阈值，生成二值矩阵
    final_simli = np.where(final_simli > threshold, 1, 0)

    # 转换为DataFrame格式
    final_simli = pd.DataFrame(final_simli)

    # 读取特征排序文件
    df = pd.read_csv('./sorted_feature_importances.csv', index_col=0)
    methods = df.index.tolist()

    # 创建无向图
    G = nx.Graph()

    # 添加节点到图中
    G.add_nodes_from(methods)

    # 根据最终相似度矩阵添加边
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            if final_simli.iloc[i, j] == 1:
                G.add_edge(methods[i], methods[j])

    # 社团划分
    partition = community.best_partition(G)

    # 获取社团划分结果
    communities = set(partition.values())  # 获取所有社团编号
    all_rankings = []

    # 对每个社团进行处理
    for community_id in communities:
        # 获取当前社团中的节点
        selected_methods = [method for method, community in partition.items() if community == community_id]

        if selected_methods:
            # 从 df 中筛选出对应节点的数据
            selected_data = df[df.index.isin(selected_methods)].values

            # 将社团ID与每个节点的排序数据拼接
            for row in selected_data:
                all_rankings.append(np.concatenate(([community_id], row)))

    # 如果只有一个社团，则直接对所有排序进行波达计数
    if len(communities) == 1:
        all_rankings_without_id = [row[1:] for row in all_rankings]  # 排除掉社团ID列
        final_ranking = borda_count(np.concatenate(all_rankings_without_id))  # 你需要定义或调用这个函数
        final_ranking = sorted(final_ranking.items(), key=lambda x: x[1], reverse=True)  # 排序
        # 转换为 DataFrame
        final_global_ranking_df = pd.DataFrame(final_ranking)
        # 删除排名得分列，只保留社团ID和节点ID
        final_ranking = final_global_ranking_df.drop(final_global_ranking_df.columns[1], axis=1)
    else:
        final_ranking = {}  # 用于存储最终汇总的排名
        # 对每个社团分别进行波达计数处理
        for community_id in communities:
            # 获取当前社团的所有排名数据
            community_rankings = [row[1:] for row in all_rankings if row[0] == community_id]  # 排除掉社团ID列
            # 计算该社团内的波达计数
            community_score = borda_count(np.concatenate(community_rankings))  # 你需要定义或调用这个函数
            # 对社团内的波达计数结果进行排序
            community_score = sorted(community_score.items(), key=lambda x: x[1], reverse=True)
            # 将社团的排序结果存入字典
            final_ranking[community_id] = community_score
            # 汇总所有社团的排名并将每个社团的排名数据合并
            all_community_rankings = []
            for community_id, rankings in final_ranking.items():
                # 只保存每个社团的最终排序
                final_community_ranking = [rank for rank, _ in rankings]
                all_community_rankings.append(final_community_ranking)
        # 将所有社团的排名结果进行一次全局波达计数
        final_global_ranking = borda_count(np.concatenate(all_community_rankings))  # 你需要定义或调用这个函数
        # 对全局波达计数结果进行排序
        final_global_ranking = sorted(final_global_ranking.items(), key=lambda x: x[1], reverse=True)
        # 转换为 DataFrame
        final_global_ranking_df = pd.DataFrame(final_global_ranking)
        # 删除排名得分列，只保留社团ID和节点ID
        final_ranking = final_global_ranking_df.drop(final_global_ranking_df.columns[1], axis=1)
        return final_ranking
# Genetic algorithm with optimal individual tracking, ensuring improvement every generation
def genetic_algorithm(population_size, generations, num_best, crossover_rate, mutation_rate, num_a,task_type):
    population = initialize_population(population_size, num_a)
    best_individual_overall = None
    if task_type == 'classification':
        best_score_overall = float('-inf')
    elif task_type == 'regression':
        best_score_overall = float('inf')  # Initialize with a large number for minimization (for regression)

    # Track the overall best individual
    for generation in range(generations):
        print(f"Generation {generation + 1}")
        scores = evaluate_population(population, task_type)  # Pass task_type to evaluation function

        # Get the best individual in this generation
        if task_type == 'classification':
            best_individual_in_generation = max(scores, key=lambda x: x[2])  # Maximizing AUC (3rd element in scores)
            best_individual, acc, auc = best_individual_in_generation
        elif task_type == 'regression':
            best_individual_in_generation = min(scores, key=lambda x: x[1])  # Minimizing MSE (2nd element in scores)
            best_individual, mse, r2 = best_individual_in_generation

        # If this generation's best individual is better than the overall best, update the overall best
        if task_type == 'classification':
            score = auc+acc
            if score > best_score_overall:
                best_score_overall = score
                best_individual_overall = best_individual
                print(f"Updated best individual at generation {generation + 1}: {best_individual}")
                print(f"Accuracy: {acc}, AUC: {auc}")
        elif task_type == 'regression':
            if mse < best_score_overall:
                best_score_overall = mse
                best_individual_overall = best_individual
                print(f"Updated best individual at generation {generation + 1}: {best_individual}")
                print(f"Score (MSE): {mse}, R2: {r2}")

        # Ensure that the population evolves towards the best solution
        best_individuals = select_best_individuals(scores, num_best, task_type)
        new_population = best_individuals[:]

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(best_individuals, 2)
            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
                new_population.append(child)
            else:
                new_population.append(random.choice(best_individuals))
        # Apply mutation to the new population
        new_population = [mutation(ind, mutation_rate) for ind in new_population]
        population = new_population

    # Return the best individual after all generations
    print("\nFinal optimal individual after all generations:")
    print(best_individual_overall)

    # Extract the parameters from the best individual
    lambda_matrix_1 = best_individual_overall["lambda_matrix_1"]
    lambda_matrix_2 = best_individual_overall["lambda_matrix_2"]
    lambda_matrix_3 = best_individual_overall["lambda_matrix_3"]
    lambda_matrix_4 = best_individual_overall["lambda_matrix_4"]
    threshold = best_individual_overall["threshold"]
    a = best_individual_overall["a"]

    # Construct the final similarity matrix and get the final ranking
    final_ranking = get_final_ranking(lambda_matrix_1, lambda_matrix_2, lambda_matrix_3, lambda_matrix_4,
                                      threshold)

    # Extract the top 'a' features from the final ranking
    top_a_features = final_ranking.head(a)  # Top 'a' features based on the final ranking

    print(f"\nTop {a} features selected by the optimal individual:")
    print(top_a_features)

    return best_individual_overall


def data_load():
    # 加载数据
    data1 = pd.read_excel('./data/augmented_data_train.xlsx', index_col=0)
    data2 = pd.read_excel('./data/augmented_data_test.xlsx', index_col=0)

    # 分离特征和目标变量
    X_train = data1.drop(['过会金额（万元）', '项目编号', '企业名称', '项目经理A', '过会期限（年）'], axis=1)
    y_train = data1['过会金额（万元）']
    X_test = data2.drop(['过会金额（万元）', '项目编号', '企业名称', '项目经理A', '过会期限（年）'], axis=1)
    y_test = data2['过会金额（万元）']
    return X_train,X_test,y_train,y_test

X_train,X_test,y_train,y_test = data_load()
task_type = 'regression'
select_data(X_train,y_train,X_test,y_test,20,'XGB',task_type)
num_a = X_train.shape[1]
# Example usage
best_individual = genetic_algorithm(population_size=100, generations=50, num_best=5, crossover_rate=0.8, mutation_rate=0.1, num_a=num_a, task_type=task_type)

print("Final optimal individual:")
print(best_individual)