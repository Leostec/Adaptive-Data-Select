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
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor
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


def select_data(data, X_train,y_train,X_test,y_test, num, model):
    # 初始化一个空的列表来保存每次计算的特征重要性排序结果
    all_feature_importances = []

    if model == 'XGB':
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)

    ## 1- XGBoost 内置特征重要性
    feature_importances_xgb = model.feature_importances_
    feature_importance_xgb_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances_xgb})
    feature_importance_xgb_df = feature_importance_xgb_df.sort_values(by='Importance', ascending=False)
    feature_importance_xgb_sorted = feature_importance_xgb_df['Feature'].tolist()
    all_feature_importances.append(['XGBoost'] + feature_importance_xgb_sorted)

    ## 2- 排列重要性 PermutationImportance
    feature_names = X_train.columns.tolist()
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=1, scoring='neg_mean_squared_error')
    importances = -result.importances_mean
    feature_importances_lea_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances_lea_df = feature_importances_lea_df.sort_values(by='Importance', ascending=False)
    feature_importance_lea_sorted = feature_importances_lea_df['Feature'].tolist()
    all_feature_importances.append(['PermutationImportance'] + feature_importance_lea_sorted)

    ## 4- 相关性分析
    correlations = X_train.corrwith(y_train).abs()
    correlations = pd.DataFrame(data=correlations, columns=['Correlation'])
    correlations['Feature'] = correlations.index
    feature_importances_cor_df = correlations.sort_values(by='Correlation', ascending=False)
    feature_importances_cor_sorted = feature_importances_cor_df['Feature'].tolist()
    all_feature_importances.append(['Correlation'] + feature_importances_cor_sorted)

    ## 5- 递归特征消除 Recursive Feature Elimination
    rfe = RFE(model, n_features_to_select=num)
    rfe.fit(X_train, y_train)
    rankings = rfe.ranking_
    feature_names = X_train.columns.tolist()
    feature_importances_RFE_df = sorted(zip(feature_names, rankings), key=lambda x: x[1])
    feature_importances_RFE_sorted = [x[0] for x in feature_importances_RFE_df]
    all_feature_importances.append(['RFE'] + feature_importances_RFE_sorted)

    ## 6- 随机森林内置重要性
    rf = RandomForestRegressor(n_estimators=100, random_state=1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    feature_importance_RF_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
    feature_importance_RF_df = feature_importance_RF_df.sort_values(by='Importance', ascending=False)
    feature_importance_RF_sorted = feature_importance_RF_df['Feature'].tolist()
    all_feature_importances.append(['RandomForest'] + feature_importance_RF_sorted)

    ## 7- 主成分分析
    pca = PCA()
    pca.fit(X_train)
    explained_variance_ratio = pca.explained_variance_ratio_
    feature_importance_PCA_df = pd.DataFrame({'Feature': X_train.columns, 'Explained Variance Ratio': explained_variance_ratio})
    feature_importance_PCA_df = feature_importance_PCA_df.sort_values(by='Explained Variance Ratio', ascending=False)
    feature_importance_PCA_sorted = feature_importance_PCA_df['Feature'].tolist()
    all_feature_importances.append(['PCA'] + feature_importance_PCA_sorted)

    ## 8- 方差分析 ANOVA
    fval = f_classif(X_train, y_train)
    fval_series = pd.Series(fval[0], index=X_train.columns)
    feature_importance_ANO_df = fval_series.sort_values(ascending=False)
    feature_importance_ANO_sorted = feature_importance_ANO_df.index.tolist()
    all_feature_importances.append(['ANOVA'] + feature_importance_ANO_sorted)

    ## 9- Shapley 值
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_train)
    mean_shap_values = shap_values.mean(axis=0)
    mean_shap_df = pd.DataFrame({'Feature': X_train.columns, 'Mean SHAP Value': mean_shap_values})
    feature_importance_shap_df = mean_shap_df.sort_values(by='Mean SHAP Value', ascending=False)
    feature_importance_shap_sorted = feature_importance_shap_df['Feature'].tolist()
    all_feature_importances.append(['Shapley'] + feature_importance_shap_sorted)

    ## 10 - 互信息
    mi = mutual_info_regression(X_train, y_train)
    feature_importance_mf_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': mi})
    feature_importance_mf_df = feature_importance_mf_df.sort_values(by='Importance', ascending=False)
    feature_importance_mf_sorted = feature_importance_mf_df['Feature'].tolist()
    all_feature_importances.append(['Mutual Information'] + feature_importance_mf_sorted)

    ## 11- 信息增益
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
    feature_importance_ig_sorted = feature_importance_ig_df['Feature'].tolist()
    all_feature_importances.append(['Information Gain'] + feature_importance_ig_sorted)

    ## 13 - 决策树
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    feature_importances = dt.feature_importances_
    feature_importance_DT_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    feature_importance_DT_df = feature_importance_DT_df.sort_values(by='Importance', ascending=False)
    feature_importance_DT_sorted = feature_importance_DT_df['Feature'].tolist()
    all_feature_importances.append(['Decision Tree'] + feature_importance_DT_sorted)

    ## 14- 线性判别分析
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    feature_importances = lda.coef_[0]
    feature_importance_LDA_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    feature_importance_LDA_df = feature_importance_LDA_df.sort_values(by='Importance', ascending=False)
    feature_importance_LDA_sorted = feature_importance_LDA_df['Feature'].tolist()
    all_feature_importances.append(['LDA'] + feature_importance_LDA_sorted)

    # 将所有方法的结果保存到 CSV 文件
    import csv
    with open('./sorted_feature_importances.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_feature_importances)


# 加载数据
data1 = pd.read_excel('/Users/leo/研究生/资产评估/汕头合作/线下交流/augmented_data_train.xlsx', index_col=0)
data2 = pd.read_excel('/Users/leo/研究生/资产评估/汕头合作/线下交流/augmented_data_test.xlsx', index_col=0)
# 分离特征和目标变量
X_train = data1.drop(['过会金额（万元）','项目编号','企业名称','项目经理A','过会期限（年）'], axis=1)
y_train = data1['过会金额（万元）']
X_test = data2.drop(['过会金额（万元）','项目编号','企业名称','项目经理A','过会期限（年）'], axis=1)
y_test = data2['过会金额（万元）']

data = data1.drop(['项目编号','企业名称','项目经理A','过会期限（年）'], axis=1)
select_data(data,X_train,y_train,X_test,y_test,20,'XGB')


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


import networkx as nx
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler


def construct_adjacency_matrix(lambda_matrix_1, lambda_matrix_2, lambda_matrix_3, lambda_matrix_4, threshold, a):
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
    data1 = pd.read_excel('/Users/leo/研究生/资产评估/汕头合作/线下交流/augmented_data_train.xlsx', index_col=0)
    data2 = pd.read_excel('/Users/leo/研究生/资产评估/汕头合作/线下交流/augmented_data_test.xlsx', index_col=0)

    # 分离特征和目标变量
    X_train = data1.drop(['过会金额（万元）', '项目编号', '企业名称', '项目经理A', '过会期限（年）'], axis=1)
    y_train = data1['过会金额（万元）']
    X_test = data2.drop(['过会金额（万元）', '项目编号', '企业名称', '项目经理A', '过会期限（年）'], axis=1)
    y_test = data2['过会金额（万元）']

    # 选取排名前 a 个特征
    top_features = final_ranking.head(round(a)).iloc[:, 0].tolist()

    # 使用排名前 a 个特征重新构建 X
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]

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


# Initialize population
def initialize_population(population_size, num_genes):
    population = []
    for _ in range(population_size):
        individual = {
            'lambda_matrix_1': random.uniform(0, 1),
            'lambda_matrix_2': random.uniform(0, 1),
            'lambda_matrix_3': random.uniform(0, 1),
            'lambda_matrix_4': random.uniform(0, 1),
            'threshold': random.uniform(0, 1),
            'a': random.randint(10, 57)  # Assuming 'a' is an integer in the range [1, 10]
        }
        population.append(individual)
    return population

# Evaluate population
def evaluate_population(population):
    scores = []
    for individual in population:
        lambda_matrix_1 = individual['lambda_matrix_1']
        lambda_matrix_2 = individual['lambda_matrix_2']
        lambda_matrix_3 = individual['lambda_matrix_3']
        lambda_matrix_4 = individual['lambda_matrix_4']
        threshold = individual['threshold']
        a = individual['a']
        mse, r2 = construct_adjacency_matrix(lambda_matrix_1, lambda_matrix_2, lambda_matrix_3, lambda_matrix_4, threshold, a)
        scores.append((individual, mse, r2))
    return scores

# Select best individuals
def select_best_individuals(population, scores, num_best):
    sorted_population = sorted(scores, key=lambda x: x[1])
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
def mutation(individual):
    mutation_rate = 0.1
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


def get_final_ranking(lambda_matrix_1, lambda_matrix_2, lambda_matrix_3, lambda_matrix_4, threshold, a):
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
def genetic_algorithm(population_size, generations, num_best, crossover_rate, mutation_rate):
    population = initialize_population(population_size, 6)
    best_individual_overall = None
    best_score_overall = float('inf')  # Initialize with a large number (for minimization problem)

    # Track the overall best individual
    for generation in range(generations):
        print(f"Generation {generation + 1}")
        scores = evaluate_population(population)

        # Get the best individual in this generation (minimizing MSE)
        best_individual_in_generation = min(scores, key=lambda x: x[1])  # Minimizing MSE
        best_individual, mse, r2 = best_individual_in_generation

        # If this generation's best individual is better than the overall best, update the overall best
        if mse < best_score_overall:
            best_score_overall = mse
            best_individual_overall = best_individual
            print(f"Updated best individual at generation {generation + 1}: {best_individual}")
            print(f"Score (MSE): {mse}, R2: {r2}")

        # Ensure that the population evolves towards the best solution
        best_individuals = select_best_individuals(population, scores, num_best)
        new_population = best_individuals[:]

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(best_individuals, 2)
            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
                new_population.append(child)
            else:
                new_population.append(random.choice(best_individuals))

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
                                               threshold, a)

    # Extract the top 'a' features from the final ranking
    top_a_features = final_ranking.head(a)  # Top 'a' features based on the final ranking

    print(f"\nTop {a} features selected by the optimal individual:")
    print(top_a_features)

    return best_individual_overall


# Example usage
best_individual = genetic_algorithm(population_size=100, generations=50, num_best=5, crossover_rate=0.8,
                                    mutation_rate=0.1)

print("Final optimal individual:")
print(best_individual)