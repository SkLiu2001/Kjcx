import chromadb
from typing import Annotated, Literal, Optional
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from zhipuai import ZhipuAI
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
from typing import Callable
from utils import chromadb_client,ZhiPuEmbeddingFunction,get_retriever
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, ToolMessage,AIMessage,SystemMessage
from dotenv import dotenv_values
config = dotenv_values(".env")
# llm = ChatOpenAI(model="GLM-4-Flash",openai_api_key="18b0066eec241318a9f17093c3ebe250.PBvbMvwpKFgI9MbT",openai_api_base="https://open.bigmodel.cn/api/paas/v4/", temperature=0)
def get_docs():
    current_path = os.path.dirname(__file__)
    # import ipdb 
    # ipdb.set_trace()
    f = open(os.path.join(current_path,'product_info.txt'),'r')
    lines = f.read().splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    f.close()
    return lines
newpoint_prompt="""你是一个信息抽取专家，接下来我将提供一段包含若干查新点及其对应的查新要求的文字，你需要从中抽取查新点和查新要求进行输出，并以换行符进行分割，

输出示例:1、查新点1。查新点1对应的查新要求
2、查新点2。查新点2对应的查新要求
3、查新点3。查新点3对应的查新要求
....

待抽取的查新点及解释:{input}

请你对待抽取的查新点及解释中的查新点进行抽取，并严格按照输出示例的格式进行抽取，请不要输出无关内容。"""
def get_newpoint():
    newpoint_prompt_template=ChatPromptTemplate.from_template(newpoint_prompt)
    # texts_li = get_docs()#
    llm = ChatOpenAI(model="deepseek-chat",openai_api_key=config["DEEPSEEK_KEY"],openai_api_base="https://api.deepseek.com", temperature=1.0)
    loader = UnstructuredFileLoader('data.doc')#
    docs = loader.load()#
    splited_docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)#
    collection_name = 'fangan'
    texts_li=[]#
    for doc in splited_docs:#
        texts_li.append(doc.page_content)#

    collection = chromadb_client.get_or_create_collection(collection_name,embedding_function=ZhiPuEmbeddingFunction(),metadata={"hnsw:space":"cosine"})
    ids = list(range(len(texts_li)))#
    ids = [str(i) for i in ids]#
    batch_size=10#
    for i in range(0, len(ids), batch_size):#
        collection.add(documents=texts_li[i:i + batch_size],ids=ids[i:i + batch_size])#
        print(i)#
    include=["documents"]
    retriever = get_retriever(collection,5,include=include)
    simility_prompts = retriever("请你为我列举项目的查新点与查新要求是什么")
    related_data="".join(simility_prompts['documents'])
# import ipdb
# ipdb.set_trace()
# print(simility_prompts)
# print(related_data)
    conclusion=llm.invoke(newpoint_prompt_template.format_messages(input=related_data))
    return conclusion.content
# print(conclusion)
# if __name__ == '__main__':
#     paragraphs = extract_text_from_pdf_chinese(
#         "sora.pdf",
#         min_line_length=10
#     )
#     print(paragraphs)
#     print(len(paragraphs))

#     # 创建一个向量数据库
#     path = r"Chroma" # 本地存放rag的目录

#     create_data(path, "demo1", paragraphs)