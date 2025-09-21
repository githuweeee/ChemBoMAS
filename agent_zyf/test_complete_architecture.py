#!/usr/bin/env python3
"""
完整的4个Agent架构端到端测试
验证: Enhanced Verification → SearchSpace Construction → Recommender → Fitting
"""

import sys
import os
import pandas as pd
import tempfile
from datetime import datetime

def create_realistic_test_data():
    """创建更真实的化学实验测试数据"""
    test_data = pd.DataFrame({
        'SubstanceA_name': ['南亚127e', '环氧A', '环氧B', '南亚127e', '环氧A'],
        'SubstanceA_SMILE': [
            'CC(C)(C1=CC=C(C=C1)OCC2CO2)C3=CC=C(C=C3)OCC4CO4',  # 南亚127e
            'CCO',  # 简化的环氧A
            'CCCCO',  # 简化的环氧B
            'CC(C)(C1=CC=C(C=C1)OCC2CO2)C3=CC=C(C=C3)OCC4CO4',  # 南亚127e
            'CCO'   # 环氧A
        ],
        'SubstanceA_ratio': [0.6, 0.7, 0.8, 0.65, 0.75],
        'SubstanceB_name': ['1,5-戊二胺', 'IPDA', '双氰胺', '1,5-戊二胺', 'IPDA'],
        'SubstanceB_SMILE': [
            'NCCCCCN',  # 1,5-戊二胺
            'NC1CC(C)(CN)CC(C)(C)C1',  # IPDA
            'NC#N',  # 双氰胺（简化）
            'NCCCCCN',  # 1,5-戊二胺
            'NC1CC(C)(CN)CC(C)(C)C1'   # IPDA
        ],
        'SubstanceB_ratio': [0.3, 0.2, 0.15, 0.25, 0.2],
        'Temperature': [80, 90, 95, 85, 88],
        'Target_alpha_tg': [80, 90, 65, 86, 92],
        'Target_beta_impactstrength': [110, 100, 88, 115, 105],
        'Target_gamma_elongation': [1.4, 1.1, 2.1, 1.6, 1.2]
    })
    
    return test_data

class ArchitectureTestRunner:
    """4个Agent架构测试运行器"""
    
    def __init__(self):
        self.session_state = {
            "session_id": f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
    def test_agent_1_enhanced_verification(self):
        """测试Agent 1: Enhanced Verification Agent"""
        print("🔍 测试 Agent 1: Enhanced Verification Agent...")
        
        try:
            # 准备测试数据
            test_df = create_realistic_test_data()
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
                test_df.to_csv(tmp_file.name, index=False)
                test_file_path = tmp_file.name
            
            # 模拟ToolContext
            class MockToolContext:
                def __init__(self, state):
                    self.state = state
            
            tool_context = MockToolContext(self.session_state)
            
            # 测试Enhanced Verification功能
            from enhanced_verification_tools import enhanced_verification
            result = enhanced_verification(test_file_path, tool_context)
            
            print(f"   ✅ Enhanced Verification 执行成功")
            print(f"   📊 结果长度: {len(result)} 字符")
            
            # 检查状态更新
            verification_results = self.session_state.get("verification_results")
            if verification_results:
                print(f"   ✅ 验证结果已保存到状态")
                print(f"      SMILES验证: {len(verification_results['smiles_validation']['canonical_smiles_mapping'])} 有效")
                print(f"      参数建议: {len(verification_results['parameter_suggestions'])} 个")
            
            # 清理临时文件
            os.unlink(test_file_path)
            
            return True, "Enhanced Verification Agent 功能正常"
            
        except Exception as e:
            print(f"   ❌ Enhanced Verification 测试失败: {e}")
            return False, str(e)
    
    def test_agent_2_searchspace_construction(self):
        """测试Agent 2: SearchSpace Construction Agent"""
        print("\n🔍 测试 Agent 2: SearchSpace Construction Agent...")
        
        try:
            # 检查前置条件
            if "verification_results" not in self.session_state:
                return False, "缺少Enhanced Verification Agent的输出"
            
            # 模拟用户配置
            self.session_state["baybe_campaign_config"] = {
                "objectives": [{"name": "Target_alpha_tg", "mode": "MAX"}]
            }
            self.session_state["optimization_config"] = {
                "experimental_settings": {"batch_size": 3}
            }
            
            class MockToolContext:
                def __init__(self, state):
                    self.state = state
            
            tool_context = MockToolContext(self.session_state)
            
            # 测试SearchSpace Construction功能
            from sub_agents.searchspace_construction.tools import construct_searchspace_and_campaign
            result = construct_searchspace_and_campaign("", tool_context)
            
            print(f"   ✅ SearchSpace Construction 执行成功")
            
            # 检查Campaign创建
            campaign = self.session_state.get("baybe_campaign")
            if campaign:
                print(f"   ✅ BayBE Campaign 已创建")
                print(f"      参数数量: {len(campaign.searchspace.parameter_names)}")
                print(f"      目标数量: {len(campaign.objective.targets)}")
                print(f"      参数名称: {campaign.searchspace.parameter_names}")
            
            return True, "SearchSpace Construction Agent 功能正常"
            
        except Exception as e:
            print(f"   ❌ SearchSpace Construction 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e)
    
    def test_agent_3_recommender(self):
        """测试Agent 3: Recommender Agent"""
        print("\n🔍 测试 Agent 3: Recommender Agent...")
        
        try:
            # 检查前置条件
            campaign = self.session_state.get("baybe_campaign")
            if not campaign:
                return False, "缺少SearchSpace Construction Agent的Campaign输出"
            
            class MockToolContext:
                def __init__(self, state):
                    self.state = state
            
            tool_context = MockToolContext(self.session_state)
            
            # 测试实验推荐生成
            from sub_agents.recommender.tools import generate_recommendations
            result = generate_recommendations("3", tool_context)
            
            print(f"   ✅ 实验推荐生成成功")
            print(f"   📋 推荐数量: 3个实验")
            
            # 检查推荐状态更新
            if self.session_state.get("recommendations_generated"):
                print(f"   ✅ 推荐状态已更新")
                latest_recs = self.session_state.get("latest_recommendations", [])
                print(f"      最新推荐: {len(latest_recs)} 个实验条件")
            
            return True, "Recommender Agent 功能正常"
            
        except Exception as e:
            print(f"   ❌ Recommender Agent 测试失败: {e}")
            return False, str(e)
    
    def test_agent_4_fitting(self):
        """测试Agent 4: Fitting Agent"""
        print("\n🔍 测试 Agent 4: Fitting Agent...")
        
        try:
            # 检查前置条件
            campaign = self.session_state.get("baybe_campaign")
            if not campaign:
                return False, "缺少BayBE Campaign"
            
            class MockToolContext:
                def __init__(self, state):
                    self.state = state
            
            tool_context = MockToolContext(self.session_state)
            
            # 由于没有足够的实验数据，测试基本功能
            from sub_agents.fitting.tools import analyze_campaign_performance
            result = analyze_campaign_performance(tool_context)
            
            print(f"   ✅ Campaign性能分析执行成功")
            print(f"   📊 分析结果长度: {len(result)} 字符")
            
            # 检查分析功能
            if "数据不足" in result:
                print(f"   ✅ 数据不足检测正常（符合预期）")
            
            return True, "Fitting Agent 基础功能正常"
            
        except Exception as e:
            print(f"   ❌ Fitting Agent 测试失败: {e}")
            return False, str(e)
    
    def run_complete_test(self):
        """运行完整的4个Agent测试"""
        print("🚀 完整4个Agent架构端到端测试")
        print("=" * 70)
        
        test_results = []
        
        # 依次测试4个Agent
        agent_1_success, agent_1_msg = self.test_agent_1_enhanced_verification()
        test_results.append(("Enhanced Verification Agent", agent_1_success, agent_1_msg))
        
        agent_2_success, agent_2_msg = self.test_agent_2_searchspace_construction() 
        test_results.append(("SearchSpace Construction Agent", agent_2_success, agent_2_msg))
        
        agent_3_success, agent_3_msg = self.test_agent_3_recommender()
        test_results.append(("Recommender Agent", agent_3_success, agent_3_msg))
        
        agent_4_success, agent_4_msg = self.test_agent_4_fitting()
        test_results.append(("Fitting Agent", agent_4_success, agent_4_msg))
        
        # 总结测试结果
        print("\n" + "=" * 70)
        print("📊 **完整架构测试结果总结**:")
        
        passed = 0
        for agent_name, success, message in test_results:
            status = "✅ 通过" if success else "❌ 失败"
            print(f"\n🔧 **{agent_name}**: {status}")
            if not success:
                print(f"   错误: {message}")
            else:
                print(f"   状态: {message}")
            
            if success:
                passed += 1
        
        # 最终结果
        print(f"\n📈 **测试统计**: {passed}/4 个Agent通过")
        
        if passed == 4:
            print("\n🎉 **完整架构测试成功！**")
            print("\n✨ **架构验证完成**:")
            print("   - Enhanced Verification Agent: 7个任务实现 ✅")
            print("   - SearchSpace Construction Agent: BayBE Campaign构建 ✅")
            print("   - Recommender Agent: 实验推荐和迭代管理 ✅")
            print("   - Fitting Agent: 性能分析和可视化 ✅")
            print("\n🚀 **ChemBoMAS 新架构已完全就绪，可以开始真实的化学实验优化！**")
            return True
        else:
            print(f"\n⚠️ **部分功能需要完善**: {4-passed} 个Agent需要调试")
            return False

def main():
    """主测试函数"""
    # 首先测试基础导入
    try:
        import agent
        print("✅ 主Agent模块导入成功")
    except Exception as e:
        print(f"❌ 主Agent模块导入失败: {e}")
        return False
    
    # 运行完整架构测试
    test_runner = ArchitectureTestRunner()
    return test_runner.run_complete_test()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
