import json
import re
from typing import List, Dict, Any
from datetime import datetime

class NVDProcessor:
    def __init__(self):
        self.processed_files = []
    
    def parse_cve_item(self, cve_item: Dict) -> Dict[str, Any]:
        """解析单个CVE条目，提取关键信息"""
        cve_data = cve_item["cve"]
        
        # 基本信息
        cve_id = cve_data["id"]
        description = cve_data["descriptions"][0]["value"]
        published = cve_data["published"]
        last_modified = cve_data["lastModified"]
        status = cve_data.get("vulnStatus", "Unknown")
        
        # 提取CVSS评分
        cvss_info = self.extract_cvss_metrics(cve_data.get("metrics", {}))
        
        # 提取弱点信息（CWE）
        weaknesses = self.extract_weaknesses(cve_data.get("weaknesses", []))
        
        # 提取受影响的产品
        affected_products = self.extract_affected_products(cve_data.get("configurations", []))
        
        # 提取参考链接
        references = [ref["url"] for ref in cve_data.get("references", [])]
        
        return {
            "cve_id": cve_id,
            "description": description,
            "published_date": published,
            "last_modified_date": last_modified,
            "status": status,
            **cvss_info,
            "weaknesses": weaknesses,
            "affected_products": affected_products,
            "references": references
        }
    
    def extract_cvss_metrics(self, metrics: Dict) -> Dict:
        """提取CVSS评分信息"""
        cvss_info = {
            "cvss_version": None,
            "base_score": None,
            "severity": None,
            "vector_string": None
        }
        
        # 优先使用CVSS v3.1
        if "cvssMetricV31" in metrics and metrics["cvssMetricV31"]:
            cvss_data = metrics["cvssMetricV31"][0]["cvssData"]
            cvss_info.update({
                "cvss_version": "3.1",
                "base_score": cvss_data["baseScore"],
                "severity": cvss_data["baseSeverity"],
                "vector_string": cvss_data["vectorString"]
            })
        # 其次使用CVSS v3.0
        elif "cvssMetricV30" in metrics and metrics["cvssMetricV30"]:
            cvss_data = metrics["cvssMetricV30"][0]["cvssData"]
            cvss_info.update({
                "cvss_version": "3.0",
                "base_score": cvss_data["baseScore"],
                "severity": cvss_data["baseSeverity"],
                "vector_string": cvss_data["vectorString"]
            })
        # 最后使用CVSS v2.0
        elif "cvssMetricV2" in metrics and metrics["cvssMetricV2"]:
            cvss_data = metrics["cvssMetricV2"][0]["cvssData"]
            cvss_info.update({
                "cvss_version": "2.0",
                "base_score": cvss_data["baseScore"],
                "severity": metrics["cvssMetricV2"][0].get("baseSeverity"),
                "vector_string": cvss_data["vectorString"]
            })
        
        return cvss_info
    
    def extract_weaknesses(self, weaknesses: List) -> List[str]:
        """提取弱点信息（CWE）"""
        cwe_list = []
        for weakness in weaknesses:
            for desc in weakness.get("description", []):
                if desc["lang"] == "en":
                    value = desc["value"]
                    # 提取CWE编号
                    if value.startswith("CWE-"):
                        cwe_list.append(value)
                    elif "CWE-" in value:
                        # 从文本中提取CWE编号
                        cwe_matches = re.findall(r'CWE-\d+', value)
                        cwe_list.extend(cwe_matches)
        return list(set(cwe_list))
    
    def extract_affected_products(self, configurations: List) -> List[str]:
        """提取受影响的产品列表"""
        products = []
        for config in configurations:
            for node in config.get("nodes", []):
                for cpe_match in node.get("cpeMatch", []):
                    if cpe_match.get("vulnerable", False):
                        # 简化CPE字符串，使其更易读
                        cpe = cpe_match["criteria"]
                        # 从cpe:2.3:a:vendor:product:version:*:*:*:*:*:*:*提取产品名
                        parts = cpe.split(":")
                        if len(parts) >= 5:
                            vendor = parts[3]
                            product_name = parts[4]
                            products.append(f"{vendor}/{product_name}")
                        else:
                            products.append(cpe)
        return list(set(products))  # 去重
    
    def create_cve_content(self, parsed_cve: Dict) -> str:
        """为单个CVE创建完整的文本内容"""
        content_parts = []
        
        # 基本信息
        content_parts.append(f"CVE ID: {parsed_cve['cve_id']}")
        content_parts.append(f"描述: {parsed_cve['description']}")
        content_parts.append(f"状态: {parsed_cve['status']}")
        content_parts.append(f"发布日期: {parsed_cve['published_date']}")
        content_parts.append(f"最后修改: {parsed_cve['last_modified_date']}")
        
        # CVSS信息
        if parsed_cve.get('base_score') is not None:
            content_parts.append(f"CVSS版本: {parsed_cve.get('cvss_version')}")
            content_parts.append(f"基础评分: {parsed_cve.get('base_score')}")
            content_parts.append(f"严重程度: {parsed_cve.get('severity')}")
            content_parts.append(f"向量字符串: {parsed_cve.get('vector_string')}")
        
        # 弱点信息
        if parsed_cve['weaknesses']:
            content_parts.append(f"相关弱点: {', '.join(parsed_cve['weaknesses'])}")
        
        # 受影响的产品
        if parsed_cve['affected_products']:
            content_parts.append("受影响的产品:")
            for product in parsed_cve['affected_products'][:10]:  # 限制显示数量
                content_parts.append(f"  - {product}")
            if len(parsed_cve['affected_products']) > 10:
                content_parts.append(f"  ... 以及另外 {len(parsed_cve['affected_products']) - 10} 个产品")
        
        # 参考链接
        if parsed_cve['references']:
            content_parts.append("参考链接:")
            for ref in parsed_cve['references'][:5]:  # 限制显示数量
                content_parts.append(f"  - {ref}")
        
        return "\n".join(content_parts)
    
    def create_metadata(self, parsed_cve: Dict) -> Dict[str, Any]:
        """为CVE创建元数据"""
        metadata = {
            "cve_id": parsed_cve["cve_id"],
            "source": "NVD",
            "description": f"CVE-{parsed_cve['cve_id']}漏洞信息",
            "published_date": parsed_cve["published_date"],
            "last_modified_date": parsed_cve["last_modified_date"],
            "status": parsed_cve["status"]
        }
        
        # 添加CVSS相关元数据
        if parsed_cve.get('base_score') is not None:
            metadata.update({
                "cvss_score": parsed_cve.get('base_score'),
                "severity": parsed_cve.get('severity'),
                "cvss_version": parsed_cve.get('cvss_version')
            })
        
        # 添加弱点信息
        if parsed_cve['weaknesses']:
            metadata["weaknesses"] = parsed_cve['weaknesses']
        
        # 添加年份信息
        year_match = re.search(r'CVE-(\d{4})', parsed_cve["cve_id"])
        if year_match:
            metadata["year"] = int(year_match.group(1))
        
        return metadata
    
    def process_nvd_file(self, input_file: str, max_items: int = None) -> List[Dict]:
        """处理NVD JSON文件，转换为指定格式"""
        print(f"开始处理NVD文件: {input_file}")
        
        # 读取原始数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vulnerabilities = data.get("vulnerabilities", [])
        
        # 如果指定了最大处理数量，则截取
        if max_items and max_items < len(vulnerabilities):
            vulnerabilities = vulnerabilities[:max_items]
            print(f"限制处理前 {max_items} 个CVE条目")
        
        self.processed_files = []
        
        for i, vuln in enumerate(vulnerabilities):
            try:
                # 解析CVE
                parsed_cve = self.parse_cve_item(vuln)
                
                # 创建内容
                content = self.create_cve_content(parsed_cve)
                
                # 创建元数据
                metadata = self.create_metadata(parsed_cve)
                
                # 添加到结果列表
                self.processed_files.append({
                    "file": content,
                    "metadata": metadata
                })
                
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1} 个CVE条目...")
                    
            except Exception as e:
                cve_id = vuln.get("cve", {}).get("id", "Unknown")
                print(f"处理 {cve_id} 时出错: {e}")
                continue
        
        print(f"处理完成！共生成 {len(self.processed_files)} 个文件条目")
        return self.processed_files
    
    def save_processed_data(self, output_file: str = "processed_nvd_data.json"):
        """保存处理后的数据"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_files, f, indent=2, ensure_ascii=False)
        
        print(f"数据已保存到: {output_file}")
    
    def get_sample_data(self, count: int = 5) -> List[Dict]:
        """获取样本数据用于测试"""
        if not self.processed_files:
            # 如果没有处理过数据，返回测试数据
            return [
                {"file": "hello world, 网络安全测试", "metadata": {"description": "测试文件1"}},
                {"file": "第二条测试文本", "metadata": {"description": "测试文件2"}},
                {"file": "网络安全是指保护网络系统及其数据免受攻击、损坏或未经授权访问的过程。", 
                 "metadata": {"description": "网络安全定义"}},
                {"file": "防火墙是一种网络安全系统,用于监控和控制传入和传出的网络流量。", 
                 "metadata": {"description": "防火墙定义"}}
            ]
        
        return self.processed_files[:count]


# 使用示例
def main():
    # 创建处理器实例
    processor = NVDProcessor()
    
    # 处理NVD数据文件
    # 替换为你的实际文件路径
    input_file = r"D:\Exploration\LLM\rag\nvdcve-2.0-modified.json"  # 你的NVD JSON文件路径
    
    try:
        # 处理文件（可以设置max_items限制处理数量用于测试）
        files = processor.process_nvd_file(input_file, max_items=1000)  # 测试时只处理100条
        
        # 显示前几个样本
        print("\n=== 样本数据 ===")
        sample_data = processor.get_sample_data(3)
        for i, item in enumerate(sample_data):
            print(f"\n--- 样本 {i+1} ---")
            print(f"文件内容: {item['file'][:100]}...")  # 只显示前100字符
            print(f"元数据: {item['metadata']}")
        
        # 保存处理后的数据
        processor.save_processed_data("nvd_processed_output.json")
        
        # 验证输出格式
        print(f"\n=== 格式验证 ===")
        print(f"总条目数: {len(files)}")
        if files:
            first_item = files[0]
            print(f"每个条目包含键: {list(first_item.keys())}")
            print(f"metadata包含键: {list(first_item['metadata'].keys())}")
            
        # 输出统计信息
        severities = {}
        for item in files:
            severity = item['metadata'].get('severity', 'UNKNOWN')
            severities[severity] = severities.get(severity, 0) + 1
        
        print(f"\n=== 严重程度统计 ===")
        for severity, count in severities.items():
            print(f"{severity}: {count}")
            
    except FileNotFoundError:
        print(f"文件 {input_file} 未找到，使用样本数据")
        # 使用样本数据
        files = processor.get_sample_data()
        print("样本数据:")
        for item in files:
            print(json.dumps(item, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()