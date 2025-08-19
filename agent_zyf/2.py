import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
import os
import matplotlib.pyplot as plt
import re
import glob
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold

def clean_filename(filename):
    """清理文件名，替换特殊字符为下划线，确保文件名一致性"""
    filename = str(filename)
    # 先清理可能存在的制表符、回车符和换行符
    filename = filename.strip()
    # 替换所有非法字符为下划线
    invalid_chars = r'[\\/*?:"<>|\ \t\n\r]'
    return re.sub(invalid_chars, '_', filename)

def perform_feature_selection(X, y, target_name, substance_name):
    """对单个目标变量进行特征选择"""
    try:
        print(f"\n处理物质 {substance_name} 的目标变量: {target_name}")
        
        # 使用均值填充缺失值
        X_valid = X.copy()
        y_valid = y.copy()
        
        if y_valid.isna().any():
            print(f"注意: {target_name} 存在缺失值,使用均值填充")
            y_valid = y_valid.fillna(y_valid.mean())
            
        #if len(y_valid) < 10:
        #    print(f"警告: {target_name} 的样本数量不足 ({len(y_valid)} < 10)")
        #    return None
            
        print(f"有效样本数: {len(y_valid)}")
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        
        # 自适应交叉验证折数，确保每个测试折至少2个样本
        n_samples = len(y_valid)
        n_splits = min(5, max(2, n_samples // 2))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # 创建RFECV对象
        selector = RFECV(
            estimator=RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1),
            step=5,  # 每次移除5个特征
            min_features_to_select=20,  # 至少保留20个特征
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=2
        )
        
        print("开始递归特征消除...")
        selector.fit(X_scaled, y_valid)
        
        # 获取RFECV结果
        print(f"最优特征数量: {selector.n_features_}")
        print(f"特征选择的R²得分: {selector.score(X_scaled, y_valid):.4f}")
        
        # 获取选中的特征
        selected_features = X.columns[selector.support_]
        X_selected = X[selected_features]
        
        # 精简输出：不再绘制和保存CV曲线图
        
        # 获取特征重要性
        # 使用选中的特征重新训练随机森林来获取特征重要性
        forest = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
        forest.fit(X_selected, y_valid)
        importances = forest.feature_importances_
        
        # 将特征和重要性配对
        feature_importances = [(feature, importance) 
                              for feature, importance in zip(selected_features, importances)]
        
        # 按重要性降序排序
        feature_importances.sort(key=lambda x: x[1], reverse=True)
        
        # 创建结果字典
        result_dict = {
            'substance_name': substance_name,
            'target_name': target_name,
            'n_features': selector.n_features_,
            'r2_score': selector.score(X_scaled, y_valid),
            'selected_features': selected_features.tolist(),
            'feature_importances': feature_importances,
            'feature_ranking': selector.ranking_.tolist()
        }
        
        # 精简输出：不再附加CV曲线明细
        
        return result_dict
    
    except Exception as e:
        print(f"特征选择过程中出错 ({substance_name} - {target_name}): {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def aggregate_and_save_per_substance(selection_results):
    """按物质聚合多个因变量实验的特征分数，输出每个物质的合并排序结果"""
    print("\n开始按物质聚合与筛选特征...")
    
    for substance_name, substance_results in selection_results.items():
        # 收集该物质在不同目标变量中的所有选中特征及其分数
        collected = []
        for target_name, result in substance_results.items():
            if result is None:
                continue
            for feature, importance in result['feature_importances']:
                collected.append((feature, importance))
        
        if not collected:
            print(f"物质 {substance_name} 无可聚合的特征")
            continue
        
        # 聚合分数：同一特征在多个实验中的重要性“取平均”
        sum_scores = {}
        count_scores = {}
        for feature, importance in collected:
            sum_scores[feature] = sum_scores.get(feature, 0.0) + float(importance)
            count_scores[feature] = count_scores.get(feature, 0) + 1
        avg_scores = {f: (sum_scores[f] / count_scores[f]) for f in sum_scores}
        
        # 计算应选择的特征数量 N = 总选中特征数 / 实验次数（向下取整）
        total_experiments = len([r for r in substance_results.values() if r is not None])
        total_selected_features = len(collected)
        features_to_select = total_selected_features // max(1, total_experiments)
        if features_to_select <= 0:
            features_to_select = len(avg_scores)
        
        # 排序并取前N个不重复特征
        sorted_items = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:features_to_select]
        
        # 保存到CSV
        clean_substance = clean_filename(substance_name)
        out_df = pd.DataFrame(top_items, columns=['feature', 'score'])
        out_df.to_csv(f'top_ranked_features_{clean_substance}.csv', encoding='utf-8-sig', index=False)
        print(f"物质 {substance_name}: 合并后前{len(out_df)}个特征已保存 -> top_ranked_features_{clean_substance}.csv")

def main():
    try:
        print("开始多物质多目标变量特征选择过程...")
        
        # 查找所有features_substanceX.csv文件
        feature_files = glob.glob('features_Substance*.csv')
        if not feature_files:
            print("错误: 未找到任何features_Substance*.csv文件")
            return
        
        print(f"找到特征文件: {feature_files}")
        
        # 读取目标变量
        if not os.path.exists('target_variables.csv'):
            print("错误: 未找到target_variables.csv文件")
            return
            
        target_variables = pd.read_csv('target_variables.csv', encoding='utf-8-sig')
        print(f"目标变量数量: {target_variables.shape[1]}")
        print(f"目标变量: {list(target_variables.columns)}")
        
        # 存储所有特征选择结果
        selection_results = {}
        
        # 对每个特征文件进行处理
        for feature_file in feature_files:
            substance_name = feature_file.replace('features_', '').replace('.csv', '')
            print(f"\n处理物质: {substance_name}")
            
            # 读取特征矩阵
            features_matrix = pd.read_csv(feature_file, encoding='utf-8-sig')
            print(f"特征矩阵大小: {features_matrix.shape}")
            
            # 存储该物质的结果
            substance_results = {}
            
            # 对每个目标变量进行特征选择
            for col in target_variables.columns:
                # 清理目标变量名称，用于文件命名
                clean_target_name = clean_filename(col)
                clean_substance_name = clean_filename(substance_name)
                
                # 提取目标变量
                y = target_variables[col]
                
                # 执行特征选择
                result = perform_feature_selection(features_matrix, y, col, substance_name)
                
                if result is not None:
                    substance_results[col] = result
                    
                    # 仅保存“筛选后特征及其分数”
                    importance_df = pd.DataFrame(result['feature_importances'], columns=['feature', 'score'])
                    importance_df.to_csv(
                        f'selected_features_with_scores_{clean_substance_name}_{clean_target_name}.csv',
                        encoding='utf-8-sig', index=False
                    )
            
            # 保存该物质的结果
            selection_results[substance_name] = substance_results
        
        # 按物质聚合并输出合并后的排序特征
        aggregate_and_save_per_substance(selection_results)
        
        # 精简：不再输出整体汇总CSV
        total_experiments = sum(len(substance_results) for substance_results in selection_results.values())
        print(f"\n多物质多目标变量特征选择完成!成功处理了 {len(selection_results)} 个物质,总实验次数: {total_experiments}")

    except Exception as e:
        print(f"主程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 