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

"""自适应实验推荐策略 - 根据优化进展动态调整推荐策略"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class OptimizationPhase:
    """优化阶段配置"""
    name: str
    acquisition_function: str
    batch_size: int
    focus: str
    description: str
    min_iterations: int
    max_iterations: int


class AdaptiveRecommendationStrategy:
    """
    自适应实验推荐策略
    
    根据优化进展自动调整：
    - 获取函数 (acquisition function)
    - 批次大小 (batch size)
    - 探索vs利用平衡
    """
    
    def __init__(self):
        """初始化策略阶段定义"""
        self.phases = {
            "exploration": OptimizationPhase(
                name="探索阶段",
                acquisition_function="qEI",  # Expected Improvement
                batch_size=5,
                focus="space_coverage",
                description="广泛探索参数空间，寻找有希望的区域",
                min_iterations=1,
                max_iterations=3
            ),
            
            "intensification": OptimizationPhase(
                name="强化阶段",
                acquisition_function="qNEI",  # Noisy Expected Improvement
                batch_size=3,
                focus="best_region_refinement",
                description="在有希望的区域进行精细搜索",
                min_iterations=4,
                max_iterations=7
            ),
            
            "exploitation": OptimizationPhase(
                name="利用阶段",
                acquisition_function="qUCB",  # Upper Confidence Bound
                batch_size=2,
                focus="optimal_point_confirmation",
                description="确认和验证最优参数",
                min_iterations=8,
                max_iterations=15
            ),
            
            "convergence": OptimizationPhase(
                name="收敛阶段",
                acquisition_function="qPI",  # Probability of Improvement
                batch_size=1,
                focus="final_optimization",
                description="最终优化和验证",
                min_iterations=16,
                max_iterations=100
            )
        }
        
        self.current_phase = "exploration"
        self.phase_history = []
    
    def select_strategy(
        self, 
        campaign, 
        iteration_number: int,
        force_phase: Optional[str] = None
    ) -> Dict:
        """
        基于Campaign状态和迭代轮次选择优化策略
        
        Args:
            campaign: BayBE Campaign对象
            iteration_number: 当前迭代轮次
            force_phase: 强制指定阶段（用于测试或用户覆盖）
            
        Returns:
            Dict: 策略配置字典
        """
        if force_phase and force_phase in self.phases:
            phase = self.phases[force_phase]
            self.current_phase = force_phase
        else:
            # 分析优化进展
            progress_analysis = self._analyze_optimization_progress(campaign)
            
            # 基于进展选择阶段
            phase_name = self._determine_phase(
                progress_analysis, 
                iteration_number
            )
            
            phase = self.phases[phase_name]
            self.current_phase = phase_name
        
        # 记录阶段历史
        self.phase_history.append({
            "iteration": iteration_number,
            "phase": self.current_phase,
            "timestamp": pd.Timestamp.now()
        })
        
        # 生成策略配置
        strategy_config = {
            "phase_name": phase.name,
            "acquisition_function": phase.acquisition_function,
            "batch_size": phase.batch_size,
            "focus": phase.focus,
            "description": phase.description,
            "iteration_number": iteration_number,
            "progress_analysis": progress_analysis if not force_phase else None
        }
        
        return strategy_config
    
    def _analyze_optimization_progress(self, campaign) -> Dict:
        """
        分析优化进展
        
        返回包含多个指标的分析结果
        """
        if not hasattr(campaign, 'measurements') or len(campaign.measurements) < 2:
            return {
                "status": "初始阶段",
                "improvement_rate": 1.0,
                "convergence_trend": "unknown",
                "iterations_completed": 0,
                "recommendation": "开始探索"
            }
        
        measurements = campaign.measurements
        targets = [t.name for t in campaign.objective.targets]
        
        analysis = {
            "iterations_completed": len(measurements),
            "improvement_metrics": {},
            "convergence_indicators": {},
            "recommendation": ""
        }
        
        # 为每个目标计算改进指标
        for target in targets:
            if target in measurements.columns:
                values = measurements[target].values
                
                # 计算改进率
                improvement_metrics = self._calculate_improvement_metrics(values)
                analysis["improvement_metrics"][target] = improvement_metrics
        
        # 综合改进率（取所有目标的最大值）
        all_improvement_rates = [
            metrics["recent_improvement_rate"] 
            for metrics in analysis["improvement_metrics"].values()
        ]
        
        if all_improvement_rates:
            analysis["improvement_rate"] = max(all_improvement_rates)
            analysis["avg_improvement_rate"] = np.mean(all_improvement_rates)
        else:
            analysis["improvement_rate"] = 0.0
            analysis["avg_improvement_rate"] = 0.0
        
        # 收敛性分析
        analysis["convergence_indicators"] = self._assess_convergence(measurements, targets)
        
        # 判断收敛趋势
        if analysis["improvement_rate"] < 0.02:
            analysis["convergence_trend"] = "已收敛"
        elif analysis["improvement_rate"] < 0.05:
            analysis["convergence_trend"] = "接近收敛"
        elif analysis["improvement_rate"] < 0.15:
            analysis["convergence_trend"] = "缓慢改进"
        else:
            analysis["convergence_trend"] = "快速改进"
        
        # 生成建议
        analysis["recommendation"] = self._generate_progress_recommendation(analysis)
        
        return analysis
    
    def _calculate_improvement_metrics(self, values: np.ndarray) -> Dict:
        """
        计算单个目标的改进指标
        """
        if len(values) < 2:
            return {
                "total_improvement": 0.0,
                "recent_improvement_rate": 1.0,
                "best_value": float(values[0]) if len(values) > 0 else 0.0,
                "worst_value": float(values[0]) if len(values) > 0 else 0.0
            }
        
        # 总体改进
        best_value = float(np.max(values))
        worst_value = float(np.min(values))
        initial_value = float(values[0])
        
        if initial_value != 0:
            total_improvement = (best_value - initial_value) / abs(initial_value)
        else:
            total_improvement = 0.0
        
        # 最近的改进率（最近3个 vs 之前3个）
        if len(values) >= 6:
            recent_values = values[-3:]
            previous_values = values[-6:-3]
            
            recent_avg = np.mean(recent_values)
            previous_avg = np.mean(previous_values)
            
            if previous_avg != 0:
                recent_improvement_rate = abs((recent_avg - previous_avg) / previous_avg)
            else:
                recent_improvement_rate = 0.0
        elif len(values) >= 3:
            recent_avg = np.mean(values[-2:])
            previous_avg = float(values[0])
            
            if previous_avg != 0:
                recent_improvement_rate = abs((recent_avg - previous_avg) / previous_avg)
            else:
                recent_improvement_rate = 0.0
        else:
            recent_improvement_rate = total_improvement
        
        return {
            "total_improvement": total_improvement,
            "recent_improvement_rate": recent_improvement_rate,
            "best_value": best_value,
            "worst_value": worst_value,
            "current_value": float(values[-1]),
            "value_range": best_value - worst_value,
            "value_std": float(np.std(values))
        }
    
    def _assess_convergence(self, measurements: pd.DataFrame, targets: List[str]) -> Dict:
        """
        评估收敛性
        """
        indicators = {
            "is_converging": False,
            "convergence_confidence": 0.0,
            "plateau_detected": False,
            "oscillation_detected": False
        }
        
        if len(measurements) < 5:
            return indicators
        
        # 检测plateau（平台期）
        for target in targets:
            if target in measurements.columns:
                recent_values = measurements[target].values[-5:]
                value_range = np.max(recent_values) - np.min(recent_values)
                value_mean = np.mean(recent_values)
                
                if value_mean != 0:
                    relative_range = value_range / abs(value_mean)
                    
                    if relative_range < 0.05:  # 变化小于5%
                        indicators["plateau_detected"] = True
        
        # 检测振荡
        for target in targets:
            if target in measurements.columns:
                values = measurements[target].values
                if len(values) >= 4:
                    # 计算一阶差分
                    diffs = np.diff(values)
                    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
                    
                    if sign_changes > len(diffs) * 0.6:  # 60%以上的点发生方向变化
                        indicators["oscillation_detected"] = True
        
        # 综合判断收敛性
        if indicators["plateau_detected"] and not indicators["oscillation_detected"]:
            indicators["is_converging"] = True
            indicators["convergence_confidence"] = 0.8
        elif indicators["plateau_detected"]:
            indicators["convergence_confidence"] = 0.5
        
        return indicators
    
    def _determine_phase(self, progress_analysis: Dict, iteration_number: int) -> str:
        """
        基于进展分析和迭代次数确定阶段
        """
        improvement_rate = progress_analysis.get("improvement_rate", 1.0)
        convergence_trend = progress_analysis.get("convergence_trend", "unknown")
        convergence_indicators = progress_analysis.get("convergence_indicators", {})
        
        # 基于迭代次数的基本阶段划分
        if iteration_number <= 3:
            base_phase = "exploration"
        elif iteration_number <= 7:
            base_phase = "intensification"
        elif iteration_number <= 15:
            base_phase = "exploitation"
        else:
            base_phase = "convergence"
        
        # 基于改进率的动态调整
        if improvement_rate > 0.15:
            # 快速改进中，保持或返回探索阶段
            return "exploration" if iteration_number <= 5 else "intensification"
        
        elif improvement_rate > 0.05:
            # 中等改进，使用基本阶段
            return base_phase
        
        elif improvement_rate < 0.02:
            # 改进缓慢，考虑收敛
            if convergence_indicators.get("is_converging", False):
                return "convergence"
            elif iteration_number > 10:
                return "exploitation"
            else:
                return base_phase
        
        else:
            return base_phase
    
    def _generate_progress_recommendation(self, analysis: Dict) -> str:
        """
        生成基于进展分析的建议
        """
        improvement_rate = analysis.get("improvement_rate", 0.0)
        convergence_trend = analysis.get("convergence_trend", "unknown")
        iterations = analysis.get("iterations_completed", 0)
        
        if improvement_rate > 0.15:
            return f"优化进展良好（改进率{improvement_rate:.1%}），建议继续当前策略"
        
        elif improvement_rate > 0.05:
            return f"优化稳定推进（改进率{improvement_rate:.1%}），可考虑调整为精细搜索"
        
        elif improvement_rate < 0.02 and iterations > 10:
            return f"改进速度放缓（{improvement_rate:.1%}），建议检查收敛状态或考虑停止优化"
        
        elif iterations < 5:
            return "优化初期，建议广泛探索参数空间"
        
        else:
            return f"当前改进率{improvement_rate:.1%}，建议{convergence_trend}"
    
    def get_strategy_summary(self) -> str:
        """
        获取当前策略摘要
        """
        phase = self.phases[self.current_phase]
        
        summary = f"""
🎯 **当前优化策略**

📊 **阶段**: {phase.name}
🔧 **获取函数**: {phase.acquisition_function}
📦 **批次大小**: {phase.batch_size}
🎨 **优化重点**: {phase.focus}

📝 **策略描述**: {phase.description}

📈 **适用轮次**: {phase.min_iterations} - {phase.max_iterations}

🔄 **历史阶段**: {len(self.phase_history)} 次策略调整
        """
        
        return summary
    
    def get_phase_history_dataframe(self) -> pd.DataFrame:
        """
        获取阶段历史记录DataFrame
        """
        if not self.phase_history:
            return pd.DataFrame(columns=["iteration", "phase", "timestamp"])
        
        return pd.DataFrame(self.phase_history)


# 使用示例
if __name__ == "__main__":
    print("🧪 自适应实验推荐策略演示\n")
    
    # 创建策略实例
    strategy = AdaptiveRecommendationStrategy()
    
    # 模拟不同阶段的策略选择
    print("=" * 60)
    print("模拟优化过程中的策略演变:")
    print("=" * 60)
    
    # 创建模拟的Campaign对象（简化版）
    class MockCampaign:
        def __init__(self):
            self.measurements = pd.DataFrame()
            self.objective = type('obj', (), {
                'targets': [
                    type('target', (), {'name': 'Target_alpha'})(),
                    type('target', (), {'name': 'Target_beta'})()
                ]
            })()
    
    campaign = MockCampaign()
    
    # 模拟10轮优化
    for iteration in range(1, 11):
        # 模拟实验数据
        if iteration == 1:
            campaign.measurements = pd.DataFrame({
                'Target_alpha': [80.0],
                'Target_beta': [100.0]
            })
        else:
            # 模拟改进（随迭代次数递减）
            improvement = 10 / iteration
            new_alpha = campaign.measurements['Target_alpha'].max() + improvement
            new_beta = campaign.measurements['Target_beta'].max() + improvement * 0.5
            
            new_row = pd.DataFrame({
                'Target_alpha': [new_alpha],
                'Target_beta': [new_beta]
            })
            campaign.measurements = pd.concat(
                [campaign.measurements, new_row], 
                ignore_index=True
            )
        
        # 选择策略
        strategy_config = strategy.select_strategy(campaign, iteration)
        
        print(f"\n轮次 {iteration}:")
        print(f"  阶段: {strategy_config['phase_name']}")
        print(f"  获取函数: {strategy_config['acquisition_function']}")
        print(f"  批次大小: {strategy_config['batch_size']}")
        print(f"  描述: {strategy_config['description']}")
        
        if strategy_config['progress_analysis']:
            progress = strategy_config['progress_analysis']
            print(f"  改进率: {progress['improvement_rate']:.3f}")
            print(f"  收敛趋势: {progress['convergence_trend']}")
    
    print("\n" + "=" * 60)
    print("策略历史记录:")
    print("=" * 60)
    print(strategy.get_phase_history_dataframe())


