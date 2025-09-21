# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""生成标准CSV模板的工具函数"""

from .enhanced_verification_tools import UserDefinedEncodingHandler

def generate_csv_template(tool_context, num_substances: int = 4, output_file: str = "standard_template.csv") -> str:
    """
    生成包含扩展列类型的标准CSV模板
    """
    try:
        handler = UserDefinedEncodingHandler()
        template_content = handler.generate_standard_csv_template(num_substances)
        
        # 保存模板文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        return f"""
✅ **标准CSV模板已生成**: {output_file}

📋 **模板包含的列类型**:
• **物质信息**: 名称、SMILES、比例、类型、供应商、等级
• **物理性质**: 密度、粘度  
• **成本信息**: 单价、可获得性
• **工艺参数**: 温度、时间、压力、转速
• **目标变量**: 多种性能指标

💡 **使用方式**:
1. 参考模板调整您的数据格式
2. 或保持现有格式，系统会智能识别
3. 两种方式都支持，推荐使用模板以获得最佳体验

🎯 **智能识别能力**: 
即使不使用标准格式，系统也能识别常见的:
- 中英文列名 (如: 密度/density, 供应商/supplier)
- 物质类型 (树脂、固化剂、催化剂、溶剂、添加剂等)
- 物理性质 (密度、粘度、熔点、玻璃化温度等)
- 工艺参数 (温度、时间、压力等)
        """
        
    except Exception as e:
        return f"生成模板时出错: {str(e)}"
