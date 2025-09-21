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

"""Recommender Agent Tools - 贝叶斯优化推荐和迭代管理"""

import os
import pandas as pd
import numpy as np
import json
import tempfile
from datetime import datetime
from google.adk.tools import ToolContext

# BayBE导入
try:
    from baybe import Campaign
    from baybe.utils.dataframe import add_fake_measurements
    BAYBE_AVAILABLE = True
except ImportError:
    print("Warning: BayBE not installed. Recommender Agent will not function.")
    BAYBE_AVAILABLE = False


def generate_recommendations(batch_size: str, tool_context: ToolContext) -> str:
    """
    生成实验推荐
    
    Args:
        batch_size: 推荐的实验数量
        tool_context: ADK工具上下文
        
    Returns:
        str: 实验推荐结果
    """
    state = tool_context.state
    session_id = state.get("session_id", "unknown")
    
    if not BAYBE_AVAILABLE:
        return "❌ BayBE未安装，无法生成推荐。请运行: pip install 'baybe[chem]'"
    
    try:
        # 获取准备好的Campaign
        campaign = state.get("baybe_campaign")
        
        if not campaign:
            return "❌ 未找到BayBE Campaign。请先运行SearchSpace Construction Agent。"
        
        # 验证batch_size
        try:
            batch_size = int(batch_size)
            if batch_size <= 0 or batch_size > 20:
                batch_size = 5  # 默认值
        except ValueError:
            batch_size = 5  # 默认值
        
        # 生成推荐
        recommendations = campaign.recommend(batch_size=batch_size)
        
        # 保存推荐结果
        recommendation_file = f"recommendations_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        recommendations.to_csv(recommendation_file, index=False)
        
        # 更新状态
        state["latest_recommendations"] = recommendations.to_dict('records')
        state["recommendation_file"] = recommendation_file
        state["recommendations_generated"] = True
        state["awaiting_experimental_results"] = True
        
        # 生成用户友好的推荐显示
        return _format_recommendations_output(recommendations, campaign, recommendation_file)
        
    except Exception as e:
        return f"❌ 推荐生成失败: {str(e)}"


def upload_experimental_results(results_file_path: str, tool_context: ToolContext) -> str:
    """
    处理用户上传的实验结果并更新Campaign
    
    Args:
        results_file_path: 实验结果CSV文件路径或内容
        tool_context: ADK工具上下文
        
    Returns:
        str: 结果处理状态
    """
    state = tool_context.state
    session_id = state.get("session_id", "unknown")
    
    if not BAYBE_AVAILABLE:
        return "❌ BayBE未安装，无法处理实验结果。"
    
    try:
        # 获取当前Campaign
        campaign = state.get("baybe_campaign")
        
        if not campaign:
            return "❌ 未找到BayBE Campaign。请先完成搜索空间构建。"
        
        # 处理文件路径 vs 文件内容（复用Enhanced Verification Agent的逻辑）
        if ',' in results_file_path and '\n' in results_file_path and not os.path.exists(results_file_path):
            # 是CSV内容，写入临时文件
            temp_file_path = f"temp_results_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                f.write(results_file_path)
            results_file_path = temp_file_path
            print(f"接收到CSV内容，已写入临时文件: {results_file_path}")
        
        # 验证文件存在
        if not os.path.exists(results_file_path):
            return f"❌ 实验结果文件不存在: {results_file_path}"
        
        # 读取实验结果
        results_df = pd.read_csv(results_file_path)
        
        # 简化的格式验证
        expected_targets = [target.name for target in campaign.objective.targets]
        missing_targets = [col for col in expected_targets if col not in results_df.columns]
        
        if missing_targets:
            return f"❌ 实验结果缺少目标列: {', '.join(missing_targets)}"
        
        # 数据预处理
        processed_results = _preprocess_experimental_results(results_df, campaign)
        
        if processed_results.empty:
            return "❌ 处理后的实验结果为空，请检查数据格式。"
        
        # 更新BayBE Campaign
        campaign.add_measurements(processed_results)
        
        # 更新状态
        current_round = state.get("optimization_round", 0) + 1
        state["optimization_round"] = current_round
        state["campaign_updated"] = True
        state["awaiting_experimental_results"] = False
        state["ready_for_next_recommendations"] = True
        
        return f"""
✅ **实验结果已成功添加到Campaign**

📊 **本轮实验摘要**:
- 轮次: {current_round}
- 新增实验: {len(processed_results)}
- Campaign总实验数: {len(campaign.measurements)}

🔄 **状态更新**:
- Campaign已更新 ✅
- 可以生成下一轮推荐 ✅

🚀 **下一步**: 使用 generate_recommendations 工具获取新的实验推荐
        """
        
    except Exception as e:
        return f"❌ 实验结果处理失败: {str(e)}"


def check_convergence(tool_context: ToolContext) -> str:
    """
    检查优化收敛性
    """
    state = tool_context.state
    
    if not BAYBE_AVAILABLE:
        return "❌ BayBE未安装，无法进行收敛性分析。"
    
    try:
        campaign = state.get("baybe_campaign")
        current_round = state.get("optimization_round", 0)
        
        if not campaign:
            return "❌ 未找到BayBE Campaign。"
        
        if current_round < 2:
            return f"""
📊 **优化进展分析** (轮次 {current_round})

🔄 **当前状态**: 优化初期
- 完成轮次: {current_round}
- 建议: 继续收集更多实验数据

🎯 **下一步建议**:
- 再进行 2-3 轮实验以建立有效的代理模型
- 推荐批次大小: 3-5 个实验
- 重点: 探索参数空间
            """
        
        # 简单的收敛性分析
        measurements = campaign.measurements
        
        if len(measurements) >= 5:
            # 计算最近几轮的改进
            targets = [t.name for t in campaign.objective.targets]
            recent_improvement = 0
            
            for target in targets:
                if target in measurements.columns:
                    values = measurements[target].values
                    if len(values) >= 3:
                        recent_avg = np.mean(values[-3:])
                        previous_avg = np.mean(values[-6:-3]) if len(values) >= 6 else values[0]
                        improvement = abs((recent_avg - previous_avg) / previous_avg) if previous_avg != 0 else 0
                        recent_improvement = max(recent_improvement, improvement)
            
            if recent_improvement < 0.05:
                return f"""
📊 **优化收敛性分析** (轮次 {current_round})

🎯 **收敛状态**: 接近收敛 
- 最近改进率: {recent_improvement:.3f}
- 总实验数: {len(measurements)}

🛑 **建议**: 考虑停止优化
- 改进速度已明显放缓
- 可以使用当前最优参数进行生产

📊 **最终分析**: 建议运行Fitting Agent进行详细结果分析
                """
            else:
                return f"""
📊 **优化收敛性分析** (轮次 {current_round})

▶️ **收敛状态**: 仍在改进中
- 最近改进率: {recent_improvement:.3f}
- 总实验数: {len(measurements)}

🚀 **建议**: 继续优化
- 仍有显著改进空间
- 建议再进行2-3轮实验
                """
        
        return "📊 实验数据不足，无法进行收敛性分析。建议至少进行5轮实验。"
        
    except Exception as e:
        return f"❌ 收敛性分析失败: {str(e)}"


def _preprocess_experimental_results(results_df: pd.DataFrame, campaign: Campaign) -> pd.DataFrame:
    """
    预处理实验结果数据
    """
    processed_df = results_df.copy()
    
    # 确保只包含Campaign需要的列
    required_columns = list(campaign.searchspace.parameter_names) + [t.name for t in campaign.objective.targets]
    
    # 保留需要的列
    available_columns = [col for col in required_columns if col in processed_df.columns]
    processed_df = processed_df[available_columns]
    
    # 数据类型转换
    for target in campaign.objective.targets:
        if target.name in processed_df.columns:
            processed_df[target.name] = pd.to_numeric(processed_df[target.name], errors='coerce')
    
    # 移除包含NaN的行
    processed_df = processed_df.dropna()
    
    return processed_df


def _format_recommendations_output(recommendations: pd.DataFrame, campaign: Campaign, file_path: str) -> str:
    """
    格式化推荐输出
    """
    output = f"""
🎯 **实验推荐已生成**

📊 **推荐概览**:
- 推荐实验数: {len(recommendations)}
- 参数数量: {len(campaign.searchspace.parameter_names)}
- 目标数量: {len(campaign.objective.targets)}

🧪 **推荐的实验条件**:
"""
    
    # 显示推荐的实验条件
    for idx, row in recommendations.iterrows():
        output += f"\n**实验 {idx + 1}**:\n"
        for param_name in campaign.searchspace.parameter_names:
            if param_name in row:
                value = row[param_name]
                if isinstance(value, float):
                    output += f"   - {param_name}: {value:.3f}\n"
                else:
                    output += f"   - {param_name}: {value}\n"
    
    output += f"""

📄 **文件保存**: {file_path}

🔄 **下一步**:
1. 按照上述条件进行实验
2. 测量目标变量: {', '.join([t.name for t in campaign.objective.targets])}
3. 使用 upload_experimental_results 工具上传结果

💡 **实验提示**:
- 请确保实验条件严格按照推荐值执行
- 记录任何异常情况或偏差
- 测量所有目标变量以获得最佳优化效果
    """
    
    return output