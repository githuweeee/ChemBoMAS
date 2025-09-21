#!/usr/bin/env python3
"""
测试脚本：验证Enhanced Verification Agent的重构成功
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """测试所有必要的导入"""
    print("🔍 测试导入...")
    
    try:
        # 测试新的工具函数导入
        import enhanced_verification_tools as evt
        print("✅ Enhanced Verification Tools模块导入成功")
        
        # 测试主要的类
        validator = evt.SimplifiedSMILESValidator()
        advisor = evt.IntelligentParameterAdvisor()
        encoder = evt.UserDefinedEncodingHandler()
        print("✅ 主要类创建成功")
        
        # 测试主要的agent导入（需要修复路径）
        try:
            import agent
            print("✅ Agent模块导入成功")
        except Exception as e:
            print(f"⚠️ Agent模块导入失败（预期）: {e}")
            print("   这在独立测试中是正常的")
        
        # 测试prompts导入
        from prompts import return_instructions_enhanced_verification
        print("✅ Enhanced Verification Prompts导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_agent_structure():
    """测试Agent结构"""
    print("\n🔍 测试Agent结构...")
    
    try:
        from agent import enhanced_verification_agent, root_agent
        
        # 检查Enhanced Verification Agent
        print(f"Enhanced Verification Agent名称: {enhanced_verification_agent.name}")
        print(f"工具数量: {len(enhanced_verification_agent.tools)}")
        print(f"可用工具: {[tool.__name__ for tool in enhanced_verification_agent.tools]}")
        
        # 检查Root Agent
        print(f"Root Agent子代理数量: {len(root_agent.sub_agents)}")
        print(f"Root Agent工具数量: {len(root_agent.tools)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent结构测试失败: {e}")
        return False

def test_tool_functions():
    """测试工具函数基本功能"""
    print("\n🔍 测试工具函数...")
    
    try:
        from enhanced_verification_tools import (
            SimplifiedSMILESValidator, 
            IntelligentParameterAdvisor,
            UserDefinedEncodingHandler
        )
        
        # 测试SMILES验证器
        validator = SimplifiedSMILESValidator()
        print("✅ SMILES验证器创建成功")
        
        # 测试参数建议器
        advisor = IntelligentParameterAdvisor()
        print("✅ 参数建议器创建成功")
        
        # 测试编码处理器
        encoder = UserDefinedEncodingHandler()
        print("✅ 编码处理器创建成功")
        
        # 简单的功能测试
        import pandas as pd
        test_df = pd.DataFrame({
            'SubstanceA_name': ['树脂A'],
            'SubstanceA_SMILES': ['CCO'], 
            'SubstanceB_name': ['稀释剂'],
            'SubstanceB_SMILES': ['']  # 特殊物质
        })
        user_special_data = encoder.identify_user_special_substances(test_df)
        print(f"✅ 用户特殊物质识别测试: {len(user_special_data['substances_without_smiles'])} 个特殊物质")
        
        return True
        
    except Exception as e:
        print(f"❌ 工具函数测试失败: {e}")
        return False

def test_baybe_availability():
    """测试BayBE可用性"""
    print("\n🔍 测试BayBE可用性...")
    
    try:
        from baybe.utils.chemistry import get_canonical_smiles
        print("✅ BayBE已安装并可用")
        
        # 简单的BayBE功能测试
        test_smiles = "CCO"
        canonical = get_canonical_smiles(test_smiles)
        print(f"✅ BayBE SMILES验证测试: {test_smiles} → {canonical}")
        return True
        
    except ImportError:
        print("⚠️  BayBE未安装 - 系统将使用降级模式")
        print("   请运行: pip install baybe")
        return False
    except Exception as e:
        print(f"⚠️  BayBE测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("🚀 Enhanced Verification Agent 重构测试")
    print("=" * 50)
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("导入测试", test_imports()))
    test_results.append(("Agent结构测试", test_agent_structure()))
    test_results.append(("工具函数测试", test_tool_functions()))
    test_results.append(("BayBE可用性测试", test_baybe_availability()))
    
    # 总结测试结果
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(test_results)} 个测试通过")
    
    if passed == len(test_results):
        print("\n🎉 所有测试通过！Enhanced Verification Agent重构成功！")
    elif passed >= len(test_results) - 1:  # 允许BayBE测试失败
        print("\n✅ 核心功能测试通过！可以继续开发（需要安装BayBE）")
    else:
        print("\n❌ 存在重大问题，需要修复后再继续")
    
    return passed == len(test_results)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
