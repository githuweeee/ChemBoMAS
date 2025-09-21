# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fitting Agent Tools - 模型分析与可视化"""

import os
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from google.adk.tools import ToolContext

# BayBE和机器学习导入
try:
    from baybe import Campaign
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_score
    BAYBE_AVAILABLE = True
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: 某些依赖未安装: {e}")
    BAYBE_AVAILABLE = False
    ML_AVAILABLE = False


def analyze_campaign_performance(tool_context: ToolContext) -> str:
    """
    分析BayBE Campaign的性能和优化效果
    """
    state = tool_context.state
    session_id = state.get("session_id", "unknown")
    
    if not BAYBE_AVAILABLE:
        return "❌ BayBE未安装，无法进行性能分析。"
    
    try:
        campaign = state.get("baybe_campaign")
        current_round = state.get("optimization_round", 0)
        
        if not campaign:
            return "❌ 未找到BayBE Campaign。"
        
        if not hasattr(campaign, 'measurements') or len(campaign.measurements) < 3:
            return f"""
📊 **Campaign性能分析** (轮次 {current_round})

⚠️ **数据不足**: 当前实验数量 {len(campaign.measurements) if hasattr(campaign, 'measurements') else 0}

🔍 **建议**: 至少需要3-5轮实验数据才能进行有效的性能分析

📈 **可用分析** (当有足够数据时):
- 优化轨迹可视化
- 目标值改进趋势
- 参数重要性分析
- 代理模型性能评估
- 收敛性诊断
            """
        
        # 执行详细的性能分析
        performance_results = _detailed_performance_analysis(campaign, current_round)
        
        # 生成可视化
        viz_files = _generate_visualizations(campaign, session_id)
        
        # 更新状态
        state["performance_analysis"] = performance_results
        state["visualization_files"] = viz_files
        state["analysis_timestamp"] = datetime.now().isoformat()
        
        return _format_performance_report(performance_results, viz_files, current_round)
        
    except Exception as e:
        return f"❌ 性能分析失败: {str(e)}"


def create_interpretable_model(tool_context: ToolContext) -> str:
    """
    创建可解释的代理模型
    """
    state = tool_context.state
    
    if not (BAYBE_AVAILABLE and ML_AVAILABLE):
        return "❌ 缺少必要依赖，无法创建代理模型。"
    
    try:
        campaign = state.get("baybe_campaign")
        
        if not campaign or not hasattr(campaign, 'measurements') or len(campaign.measurements) < 5:
            return "⚠️ 实验数据不足，无法训练可靠的代理模型。建议至少进行5轮实验。"
        
        # 提取特征和目标
        measurements = campaign.measurements
        feature_columns = [col for col in measurements.columns if not col.startswith('Target_')]
        target_columns = [col for col in measurements.columns if col.startswith('Target_')]
        
        interpretable_results = {}
        
        # 为每个目标创建代理模型
        for target in target_columns:
            if target in measurements.columns:
                X = measurements[feature_columns]
                y = measurements[target]
                
                # 处理分类特征（分子参数）
                X_processed = _preprocess_features_for_ml(X, campaign)
                
                # 训练随机森林（可解释性好）
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_processed, y)
                
                # 计算性能指标
                y_pred = rf_model.predict(X_processed)
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                
                # 特征重要性
                feature_importance = dict(zip(X_processed.columns, rf_model.feature_importances_))
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                
                interpretable_results[target] = {
                    "r2_score": r2,
                    "rmse": rmse,
                    "top_features": top_features,
                    "model_quality": "good" if r2 > 0.8 else "moderate" if r2 > 0.6 else "poor"
                }
        
        # 保存结果
        state["interpretable_models"] = interpretable_results
        
        return _format_interpretable_model_report(interpretable_results)
        
    except Exception as e:
        return f"❌ 代理模型创建失败: {str(e)}"


def generate_optimization_report(report_type: str, tool_context: ToolContext) -> str:
    """
    生成综合优化报告
    
    Args:
        report_type: 报告类型 ("summary" | "detailed" | "publication")
    """
    state = tool_context.state
    session_id = state.get("session_id", "unknown")
    
    try:
        campaign = state.get("baybe_campaign")
        performance_analysis = state.get("performance_analysis", {})
        interpretable_models = state.get("interpretable_models", {})
        current_round = state.get("optimization_round", 0)
        
        if not campaign:
            return "❌ 未找到BayBE Campaign数据。"
        
        # 根据报告类型生成不同详细程度的报告
        if report_type == "summary":
            report = _generate_summary_report(campaign, current_round, performance_analysis)
        elif report_type == "detailed":
            report = _generate_detailed_report(campaign, performance_analysis, interpretable_models)
        elif report_type == "publication":
            report = _generate_publication_ready_report(campaign, performance_analysis, interpretable_models)
        else:
            report = _generate_summary_report(campaign, current_round, performance_analysis)
        
        # 保存报告
        report_file = f"optimization_report_{report_type}_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        state["latest_report"] = report_file
        
        return f"""
📄 **优化报告已生成**

📊 **报告类型**: {report_type}
📁 **文件路径**: {report_file}

{report}
        """
        
    except Exception as e:
        return f"❌ 报告生成失败: {str(e)}"


# 辅助函数
def _detailed_performance_analysis(campaign, current_round):
    """详细性能分析"""
    measurements = campaign.measurements
    targets = [t.name for t in campaign.objective.targets]
    
    analysis = {
        "total_experiments": len(measurements),
        "optimization_rounds": current_round,
        "target_analysis": {},
        "optimization_efficiency": "unknown"
    }
    
    for target in targets:
        if target in measurements.columns:
            values = measurements[target].values
            analysis["target_analysis"][target] = {
                "best_value": float(np.max(values)),
                "worst_value": float(np.min(values)),
                "mean_value": float(np.mean(values)),
                "std_value": float(np.std(values)),
                "improvement_ratio": float((np.max(values) - values[0]) / abs(values[0])) if values[0] != 0 else 0
            }
    
    return analysis


def _generate_visualizations(campaign, session_id):
    """生成优化过程可视化"""
    if not hasattr(campaign, 'measurements') or len(campaign.measurements) < 3:
        return []
    
    viz_files = []
    measurements = campaign.measurements
    targets = [t.name for t in campaign.objective.targets]
    
    try:
        # 1. 优化轨迹图
        plt.figure(figsize=(12, 8))
        
        for i, target in enumerate(targets):
            if target in measurements.columns:
                plt.subplot(2, 2, i+1)
                values = measurements[target].values
                plt.plot(values, marker='o')
                plt.title(f'{target} 优化轨迹')
                plt.xlabel('实验轮次')
                plt.ylabel(target)
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        viz_file = f"optimization_trajectory_{session_id}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files.append(viz_file)
        
        # 2. 目标值分布图
        if len(targets) >= 2:
            plt.figure(figsize=(10, 8))
            
            for i, target in enumerate(targets[:3]):  # 最多显示3个目标
                plt.subplot(2, 2, i+1)
                if target in measurements.columns:
                    plt.hist(measurements[target], bins=10, alpha=0.7)
                    plt.title(f'{target} 分布')
                    plt.xlabel(target)
                    plt.ylabel('频次')
            
            plt.tight_layout()
            viz_file = f"target_distributions_{session_id}.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            viz_files.append(viz_file)
            
    except Exception as e:
        print(f"可视化生成警告: {e}")
    
    return viz_files


def _preprocess_features_for_ml(X, campaign):
    """为机器学习预处理特征"""
    X_processed = X.copy()
    
    # 处理分类变量（分子参数）
    for param_name in campaign.searchspace.parameter_names:
        if param_name in X_processed.columns:
            if X_processed[param_name].dtype == 'object':
                # 分类编码
                X_processed[param_name] = pd.factorize(X_processed[param_name])[0]
    
    # 确保所有列都是数值类型
    X_processed = X_processed.select_dtypes(include=[np.number])
    
    return X_processed


def _format_performance_report(performance_results, viz_files, current_round):
    """格式化性能报告"""
    report = f"""
📊 **Campaign性能分析完成** (轮次 {current_round})

📈 **实验统计**:
- 总实验数: {performance_results.get('total_experiments', 0)}
- 优化轮次: {performance_results.get('optimization_rounds', 0)}

🎯 **目标分析**:
"""
    
    target_analysis = performance_results.get("target_analysis", {})
    for target, analysis in target_analysis.items():
        report += f"""
📌 **{target}**:
   - 最佳值: {analysis.get('best_value', 'N/A'):.3f}
   - 平均值: {analysis.get('mean_value', 'N/A'):.3f}
   - 改进比例: {analysis.get('improvement_ratio', 0):.1%}
"""
    
    if viz_files:
        report += f"\n📊 **可视化文件**: {len(viz_files)} 个图表已生成\n"
        for viz_file in viz_files:
            report += f"   - {viz_file}\n"
    
    report += "\n🚀 **分析完成**: 可以使用 generate_optimization_report 工具生成详细报告"
    
    return report


def _format_interpretable_model_report(interpretable_results):
    """格式化可解释模型报告"""
    report = """
🤖 **代理模型分析完成**

📊 **模型性能摘要**:
"""
    
    for target, results in interpretable_results.items():
        report += f"""
🎯 **{target}**:
   - R² 评分: {results['r2_score']:.3f}
   - RMSE: {results['rmse']:.3f}
   - 模型质量: {results['model_quality']}
   
   🔍 **重要特征** (Top 5):
"""
        for feature, importance in results['top_features']:
            report += f"      {feature}: {importance:.3f}\n"
    
    return report


def _generate_summary_report(campaign, current_round, performance_analysis):
    """生成摘要报告"""
    return f"""
🎯 ChemBoMAS 优化摘要报告

📊 基本信息:
- 优化轮次: {current_round}
- 总实验数: {len(campaign.measurements) if hasattr(campaign, 'measurements') else 0}
- 参数数量: {len(campaign.searchspace.parameter_names)}
- 目标数量: {len(campaign.objective.targets)}

📈 优化效果: 
{json.dumps(performance_analysis.get('target_analysis', {}), indent=2, ensure_ascii=False)}
    """