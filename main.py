# -*- coding: utf-8 -*-
from langchain_core.pydantic_v1 import BaseModel, Field
from zhipuai import ZhipuAI
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, ToolMessage,AIMessage,SystemMessage
from test_for_chromadb import get_newpoint
from dotenv import dotenv_values
# from papers import paper_base
import sqlite3
import os
import sys
import json
import pdftotext
import random
config = dotenv_values(".env")
# llm2 = ChatOpenAI(model="GLM-4-Plus", openai_api_key="18b0066eec241318a9f17093c3ebe250.PBvbMvwpKFgI9MbT", openai_api_base="https://open.bigmodel.cn/api/paas/v4/", temperature=0.1)
llm = ChatOpenAI(model="deepseek-chat",openai_api_key=config["DEEPSEEK_KEY"],openai_api_base="https://api.deepseek.com", temperature=1.0)
tech_extract_agent_prompt="""你是一个文献技术提取专家，我将给出一份文献摘要，你需要提取其中的关键技术与创新点，并在一段内完成输出，语言尽量连贯、准确：
文献摘要：{summary}
你需要在一段内完成输出，且以“文献核心技术：”为开头，输出尽量连贯
"""
summary_agent_primary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一位文献摘要生成专家，接下来我将提供一份文献，你需要针对文献内容生成一份2000字左右的文献摘要，在生成过程中不要分段或分点，并尽量保持摘要的连贯性和精炼,在生成过程中不要分段或分点，且不要输出其他信息      "
        ),
        ("placeholder", "{messages}"),
    ]
)
summary_prompt= """
    你是一位文献摘要生成专家，接下来我将提供一份文献，你需要针对文献内容生成一份2000字左右的文献摘要，在生成过程中不要分段或分点，并尽量保持摘要的连贯性和精炼：

    文献内容：{content}

    请在分析过程中重点关注文献中的技术内容，直接生成一份2000字的文献摘要，在生成过程中不要分段或分点，且不要输出其他信息,直接输出摘要内容即可,请以"摘要:"作为输出的起始；
    """
judge_prompt=""" 
{num}你是一位文献处理专家，你需要将我提供的文献进行解读分析，并与提供的查新点进行逐一比对，判断是否与提供的查新点密切相关，而后输出相关的查新点：

查新点名称：{newpoint_name}

查新点的解释：{newpoint_content}

文献名称：{name}

{main_tech}

请你严格按照输出格式进行输出，仅输出与文献相关的最为查新点名称，不要输出其他内容，判断相关性的标准尽量高，且输出一个最相关的查新点,
"""
subcontent_prompt="""
你是一个核心技术概括专家，我将给出一份文献核心技术，你需要提取其中的关键技术，输出其中的核心信息，并在一段内完成输出，不超过80字，语言尽量连贯、准确：
输出示例1:文献4对中欧班列运输网络的韧性分析研究，有助于认识网络在攻击或干扰状态下的响应过程，识别出班列运输网络中的关键节点和抵抗突发事件能力薄弱的环节，并期望找出网络修复过程中的最佳修复策略。
输出示例2:文献12建立了城市轨道交通网络级联失效模型,并以网络失效规模和破坏程度两个指标对其进行评估,仿真了级联失效过程并分析了不同失效策略下城市轨道交通网络的级联失效抗毁性。
输出示例3:文献42针对城市交通运输网络中节点的差异性，基于节点特征给出了其容量系数的取值方法，并构建级联失效情形下变容量系数的城市交通运输网络抗毁性模型，同时给出求解算法及仿真验证。
输出示例4:文献5提出了一种基于级联失效对加权铁路网的抗毁性研究的方法，该级联失效模型综合考虑了客流重分配时客流会考虑站点的邻接站点的剩余容量和该站点与邻接站点的连接权值，基于ER随机网络、NW小世界网络和BA无标度网络这三种经典网络和实际加权铁路换乘网，对网络的抗毁性进行了分析。
{main_tech}
你需要在一段内完成输出，且以“文献{paper_num}”为开头，输出尽量连贯
"""
conclusion_prompt="""你是一个文献查新结论生成专家，接下来，我将给出文献查新的技术摘要、结论话术、目标查新点与查新点解释，你需要参考样例，先对技术摘要中涉及的技术进行总结，而后参考结论话术进行结论生成：

技术摘要:{abstract}

结论话术：1.检出文献未见与本查新点技术特征相同的研究报道2.检出文献未见与本查新点研究特征相同的报道3.检出文献已见(技术内容)的报道，但（项目方面）等与本项目不同。检出文献未见（目标查新点）的报道。

目标查新点：{target_newpoint}

查新点解释：{target_newpoint_content}

参考样例1:检出文献已见自动引导系统、微机自动控制技术、自动循迹控制系统、自动寻路与动态避障算法、无人自动驾驶仪决策系统、路面自适应MPC轨迹跟踪控制技术等自动驾驶自适应控制技术的报道，以及基于激光雷达、相机多传感器等信息融合的自动驾驶目标检测、目标跟踪等技术的报道，但技术路线、自适应控制方法、融合的感知数据、信息融合算法、适用环境等与本项目不同。检出文献未见寒区冰雪路面跨境无人驾驶货运车辆自适应控制及信息感知融合方法的报道。

参考样例2:检出文献已见研究提出在“一带一路”倡议背景下，交通运输发挥的作用、面临问题、发展建议等的报道，但研究背景、研究对象、研究视角、研究结论等与本项目不同。检出文献未见从交通强国视角提出了交通运输推进“一带一路”的路径及“十四五”国际合作发展重点的报道。

请在在一段内完成输出，且不要分点给出回答，请直接生成查新结论，不需要给出其他辅助性文字。

"""
newpoint_extract_prompt="""你是一个信息抽取专家，接下来我将提供若干查新点及其对应的解释，你需要从中抽取查新点进行输出，并以###进行分割，

输入示例:1.查新点1。查新点1的解释 2.查新点2。查新点2的解释 3.查新点3。查新点3的解释

输出示例:查新点1###查新点2###查新点3

待抽取的查新点及解释:{input}

请你对待抽取的查新点及解释中的查新点进行抽取，并严格按照输出示例的格式进行抽取，请不要输出无关内容。"""
tech_extract_agent_prompt_template=ChatPromptTemplate.from_template(tech_extract_agent_prompt)
summary_prompt_template = ChatPromptTemplate.from_template(summary_prompt)
judge_prompt_template = ChatPromptTemplate.from_template(judge_prompt)
newpoint_extract_prompt_template = ChatPromptTemplate.from_template(newpoint_extract_prompt)
subcontent_prompt_template = ChatPromptTemplate.from_template(subcontent_prompt)
conclusion_prompt_template = ChatPromptTemplate.from_template(conclusion_prompt)
def load_pdf(input_file):
    # Load your PDF
    with open(input_file, "rb") as f:
        pdf = pdftotext.PDF(f)
    output=""
    for page in pdf:
        # 去除换行符
        page_without_newlines = page.replace('\n', ' ')
        # f.write(page_without_newlines + ' ')  # 在每页内容后添加空格以分隔不同页面的内容
        output+=page_without_newlines
    return output
def get_content(name,list):
    for k in list:
        if name in k:
            return k
    return ""
def process_single_paper(paper_num,paper_name,paper_content,newpoint_name,newpoint_content):
    main_tech_prompt_messages = tech_extract_agent_prompt_template.format_messages(summary=paper_content)
    main_tech=llm.invoke(main_tech_prompt_messages).content
    print(main_tech)
    judege_prompt_messages = judge_prompt_template.format_messages(num=random.randint(0,800000),newpoint_name=newpoint_name,newpoint_content=newpoint_content,name=paper_name,main_tech=main_tech)
    related_newpoint=llm.invoke(judege_prompt_messages).content
    output_newpoint=""
    print(related_newpoint)
    for newpoint in newpoint_list:
        if newpoint in related_newpoint:
            subcontent_prompt_messages=subcontent_prompt_template.format_messages(main_tech=main_tech,paper_num=paper_num)
            subcontent=llm.invoke(subcontent_prompt_messages).content
            output_newpoint=newpoint
            print(subcontent)
    return main_tech,subcontent,output_newpoint
# newpoint_content="""1.基于供需平衡的轴辐式江海联运协同布局规划技术。围绕平陆运河“内部转运+外部分流”的轴辐式江海联运网络结构和功能特征，提出江海联运货运枢纽选址与布局协同优化模型，建立需求与能力适配的货运通道走廊优化方法，能实现组织协同、功能匹配、运转高效的平陆运河江海联运系统的构建。
# 2.面向“集散换装”的江海联运韧性运输组织优化技术。围绕平陆运河江海联运“集散换装”需求，结合航道建设规模、航道尺度、航道线路方案情况，剖析常规条件、不确定扰动和极端中断情形条件下的平陆运河煤炭、粮食等重点货种运输组织动态响应特性，构建起多目标江海联运韧性运输组织优化模型与智能求解算法，能降低干散货“集散换装”运输组织下的多因素扰动影响，强化支撑重点货种平陆运河江海联运的可靠性和物流服务水平。
# 3.基于外部扰动的多要素耦合情境下江海联运中转多体多维协同智能控制技术。基于多式联运工艺和智能控制全过程中的多模块组合技术，综合考虑运河码头复杂特殊作业环境及装卸转运过程中外部扰动的多因素协同作用影响，提出适用于平陆运河江海联运特点的柔性中转工艺方案和多体多维智能控制技术方案，突破运河关键控制节点的中转效率瓶颈，提升平陆运河江海联运的整体通过能力。
# 4.多源数据融合下面向通航安全和效率的江海联运交通一体化控制技术。构建船舶安全与效率目标驱动的港口水域动态链网模型，建立适用于平陆运河的江海联运船舶交通组织新模式，提出基于多源数据融合的高精度船舶交通一体化管控策略和方法，建立平陆运河江海联运交通模拟系统，突破江海联运混合交通流一体化调度的瓶颈，能实现江海联运不同组织模式下船舶通航安全与效率优化。"""
# newpoint_list=['基于供需平衡的轴辐式江海联运协同布局规划技术','面向“集散换装”的江海联运韧性运输组织优化技术','基于外部扰动的多要素耦合情境下江海联运中转多体多维协同智能控制技术','多源数据融合下面向通航安全和效率的江海联运交通一体化控制技术']
newpoint_content=get_newpoint()
# import ipdb
# ipdb.set_trace()
newpoint_extract_message=newpoint_extract_prompt_template.format_messages(input=newpoint_content)
newpoint_list_tmp=llm.invoke(newpoint_extract_message).content
newpoint_list=newpoint_list_tmp.split("###")
print(newpoint_list)
newpoint_name=""
for k in range(len(newpoint_list)):
    newpoint_name+=newpoint_list[k]
    if k!=(len(newpoint_list)-1):
        newpoint_name+='、'
print(newpoint_name)

newpoint_content_list=newpoint_content.split('\n')
# txt_content=load_pdf("test/基于电子海图的船舶交通流模拟研究_赵鹏.pdf")
# summary_prompt_messages = summary_prompt_template.format_messages(content=txt_content)
# result = llm.invoke(summary_prompt_messages).content
# 连接到 SQLite 数据库
conn = sqlite3.connect('papers.db')
# 创建一个游标对象
cursor = conn.cursor()
# 查询数据
cursor.execute('SELECT * FROM papers')
# 获取所有结果
rows = cursor.fetchall()

random.seed()
output_list = {key: [] for key in newpoint_list}
for i in range(len(rows)):
    paper_num=i+1
    paper_name=rows[i][1]
    paper_content=rows[i][2]
    main_tech,subcontent,output_newpoint=process_single_paper(paper_num,paper_name,paper_content,newpoint_name,newpoint_content)
    if output_newpoint in newpoint_list:
        output_list[output_newpoint] .append(subcontent) 
    print(output_list)
conclusion_list=[]
output_conclusion=""
count=1
for k in output_list:
    print(k)
    print(output_list[k])
    output_conclusion+=(f"        {count}.检出“{k}”的相关文献共{len(output_list[k])}篇。"+"".join(output_list[k])+'\n'+'        ')
    conclusion=llm.invoke(conclusion_prompt_template.format_messages(abstract="".join(output_list[k]),target_newpoint=k,target_newpoint_content=get_content(k,newpoint_content_list))).content
    conclusion_list.append(conclusion)
    output_conclusion+=(conclusion+'\n')
    count+=1
print(conclusion_list)
output_conclusion+="        综上所述可知，国内外尚未见与本委托项目综合技术特点相同的研究报道，本委托项目的以下查新内容在国内外具有新颖性："+'\n'+newpoint_content
print(output_conclusion)