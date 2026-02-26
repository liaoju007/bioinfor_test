import pandas as pd
from Bio import Entrez
import os
import ssl
from google import genai
from google.genai import types
import time
import json

# ==========================================
# 1. 全局配置与初始化
# ==========================================

# 解决国内网络环境/代理导致的 SSL 证书报错
ssl._create_default_https_context = ssl._create_unverified_context

# NCBI 配置
Entrez.email = "your.email@example.com"
HISTORY_FILE = "downloaded_pmids.txt"
DATABASE_FILE = "ai_aggregator_database.csv"

# Gemini API 配置 (请务必替换为你的真实 Key)
GEMINI_API_KEY = "AIzaSyCPIVh_0zbO7W2EnzR2wdepdNbJZaZvi0I"
client = genai.Client(api_key=GEMINI_API_KEY)

# 期刊影响因子字典 (可自行补充)
JOURNAL_IF_MAP = {
    "nature": 50.5, "science": 44.7, "cell": 45.5,
    "american journal of respiratory and critical care medicine": 19.3,
    "european respiratory journal": 16.6, "circulation": 35.5,
    "chest": 9.6, "hypertension": 7.7,
    "gut": 23.0, "microbiome": 13.8, "cell host & microbe": 20.6,
    "bioinformatics": 4.4, "briefings in bioinformatics": 6.8
}


# ==========================================
# 2. 核心功能函数
# ==========================================

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return set()
    with open(HISTORY_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())


def save_to_history(new_pmids):
    if not new_pmids:
        return
    with open(HISTORY_FILE, "a") as f:
        for pmid in new_pmids:
            f.write(f"{pmid}\n")


def analyze_with_gemini(title, abstract):
    prompt = f"""
    你是一个资深的生信分析专家和医学文献研究员。请仔细阅读以下医学论文的标题和摘要，并进行深度的结构化信息提取。
    请严格以 JSON 格式输出，必须且只能包含以下四个字段：
    - "chinese_summary": 详细总结核心机制（特别是肠道菌群与肺动脉高压的关联），150-300字。
    - "sequencing_method": 详细提取使用的所有测序技术（如 16S rRNA, 宏基因组等），未提及写"未提及"。
    - "bioinfo_tools": 详细列出所有生信分析工具、编程语言、R包（如 QIIME 2, DADA2 等），未提及写"未提及"。
    - "innovation": 深入分析该研究在机制探索或生信方法学上的创新点，不少于 80 字。

    标题: {title}
    摘要: {abstract}
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3
            )
        )
        return response.text
    except Exception as e:
        print(f"  [!] Gemini API 请求失败: {e}")
        return '{"chinese_summary": "提取失败", "sequencing_method": "未知", "bioinfo_tools": "未知", "innovation": "未知"}'


# ==========================================
# 3. 主程序流水线
# ==========================================

def run_ai_pipeline(search_term, min_if=5.0, days_recent=730):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 启动 AI 聚合体流水线...")

    # 步骤 A: 搜索最新文献 ID
    print(f"正在检索最近 {days_recent} 天的文献...")
    handle = Entrez.esearch(db="pubmed", term=search_term, retmax=100,
                            sort="date", datetype="pdat", reldate=days_recent)
    record = Entrez.read(handle)
    handle.close()

    id_list = record.get("IdList", [])
    if not id_list:
        print("未检索到相关文献，任务结束。")
        return

    # 步骤 B: 去重过滤
    history_pmids = load_history()
    new_id_list = [pmid for pmid in id_list if pmid not in history_pmids]

    print(f"搜到 {len(id_list)} 篇文献，其中全新未处理的共 {len(new_id_list)} 篇。")
    if not new_id_list:
        print("所有文献均已在本地数据库中，无需重复抓取。任务结束。")
        return

    # 步骤 C: 下载详细内容并进行影响因子初步筛选
    print("正在下载全新文献详情并进行影响因子筛选...")
    handle = Entrez.efetch(db="pubmed", id=new_id_list, retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    valid_papers = []
    for article in records['PubmedArticle']:
        try:
            medline = article['MedlineCitation']
            article_data = medline['Article']
            pmid = str(medline['PMID'])

            journal_title = article_data.get('Journal', {}).get('Title', '').lower()
            impact_factor = JOURNAL_IF_MAP.get(journal_title, 0.0)

            if impact_factor < min_if:
                continue

            title = article_data.get('ArticleTitle', 'No Title')
            abstract_list = article_data.get('Abstract', {}).get('AbstractText', [])
            abstract_text = " ".join([str(text) for text in abstract_list]) if abstract_list else 'No Abstract'

            pub_date_info = article_data.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
            pub_date = f"{pub_date_info.get('Year', '')} {pub_date_info.get('Month', '')}".strip()

            valid_papers.append({
                'PMID': pmid, 'Date': pub_date, 'Journal': journal_title.title(),
                'ImpactFactor': impact_factor, 'Title': title, 'Abstract': abstract_text
            })
        except Exception:
            continue

    # 先把抓取过的 ID 存入历史账本，防止程序中断导致下次重复抓取
    save_to_history(new_id_list)
    print(f"保留了 {len(valid_papers)} 篇高分文献进入 AI 分析环节。")

    if not valid_papers:
        print("新文献均未达到分数线，任务结束。")
        return

    # 步骤 D: Gemini AI 自动化处理
    print("\n开始调用 Gemini 进行深度结构化分析...")
    for index, paper in enumerate(valid_papers):
        print(f"  -> [{index + 1}/{len(valid_papers)}] 分析中: {paper['Title'][:45]}...")

        if paper['Abstract'] == 'No Abstract':
            paper.update({'中文详细总结': '无摘要', '测序与实验技术': '无', '生信分析工具': '无', '创新点与启示': '无'})
            continue

        ai_response_str = analyze_with_gemini(paper['Title'], paper['Abstract'])

        try:
            parsed_data = json.loads(ai_response_str)
            paper['中文详细总结'] = parsed_data.get("chinese_summary", "解析缺失")
            paper['测序与实验技术'] = parsed_data.get("sequencing_method", "解析缺失")
            paper['生信分析工具'] = parsed_data.get("bioinfo_tools", "解析缺失")
            paper['创新点与启示'] = parsed_data.get("innovation", "解析缺失")
        except json.JSONDecodeError:
            paper.update(
                {'中文详细总结': '格式错误', '测序与实验技术': '错误', '生信分析工具': '错误', '创新点与启示': '错误'})

        # 频率控制，保护免费 API 额度
        time.sleep(4)

        # 步骤 E: 结果保存 (追加到数据库)
    df = pd.DataFrame(valid_papers)
    # 移除冗余的长英文摘要，保持最终数据库清爽
    df = df.drop(columns=['Abstract'])

    header = not os.path.exists(DATABASE_FILE)
    df.to_csv(DATABASE_FILE, mode='a', index=False, encoding="utf-8-sig", header=header)
    print(f"\n[成功] 完工！本次新增 {len(valid_papers)} 篇深度分析结果，已汇入 {DATABASE_FILE}")


if __name__ == "__main__":
    # 设定你的专属科研检索式
    query = '("pulmonary hypertension"[Title/Abstract]) AND ("gut microbiome"[Title/Abstract] OR "gut microbiota"[Title/Abstract] OR )'

    # 运行流水线 (筛选 5分以上，近两年的文献)
    run_ai_pipeline(query, min_if=0.0, days_recent=3000)