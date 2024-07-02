import asyncio
import huggingface_hub
#huggingface_hub.login(token='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')
from dotenv import dotenv_values
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.legacy.llms import OpenAILike as OpenAI
from qdrant_client import models
from tqdm.asyncio import tqdm

from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.qa import read_jsonl, save_answers
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval
from BCEmbedding.tools.llama_index import BCERerank

def post_run():
    config = dotenv_values(".env")
    llm = OpenAI(
        api_key=config["GLM_KEY"],
        model="glm-4",
        api_base="https://open.bigmodel.cn/api/paas/v4/",
        is_chat_model=True,
    )
    # embeding = HuggingFaceEmbedding(
    #     model_name="BAAI/bge-small-zh-v1.5",
    #     cache_folder="./",
    #     embed_batch_size=128,
    # )

    embeding = HuggingFaceEmbedding(
            model_name='/home/liusk-s24/data/model/bce-embedding', # 替换为本地路径
            cache_folder="./",
            max_length=512,
            embed_batch_size=128,
        )
    Settings.embed_model = embeding
    # Settings

    reranker_args = {'model': '/home/liusk-s24/data/model/bce-embedding',  # 替换为本地路径
                     'top_n': 5,
                     "device":"cuda:2"}
    reranker = BCERerank(**reranker_args)

    # 初始化 数据ingestion pipeline 和 vector store
    print("初始化数据库")
    if (config["IS_DOCUMENT"]=='False'):
        print("not doc class")
        client, vector_store = build_vector_store(config, reindex=False)

        collection_info = client.get_collection(
            config["COLLECTION_NAME"] 
        )

        if collection_info.points_count == 0:
            print(f"The collection {config['COLLECTION_NAME']} exists but has no points.")
            data = read_data(config["SOURCE_FILE"]) # 加载数据
            pipeline = build_pipeline(llm, embeding, vector_store=vector_store)
            # 暂时停止实时索引
            client.update_collection(
                collection_name=config["COLLECTION_NAME"] or "aiops24",
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
            )
            pipeline.run(documents=data, show_progress=True, num_workers=1)
            # 恢复实时索引
            client.update_collection(
                collection_name=config["COLLECTION_NAME"] or "aiops24",
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
            )
            print(len(data))
            return llm,reranker,retrievers
    else:
        print('doc class')
        client, vector_stores = build_vector_store(config, reindex=False,is_doc=True)
        print("vector stores",vector_stores)
        retrievers = {}
        doc_list = ["director","emsplus","rcp","umac"]
        for doc,vector_store in zip(doc_list,vector_stores):
            
            collection_info = client.get_collection(
                config["COLLECTION_NAME"] + doc
            )
            if collection_info.points_count == 0: 
                data = read_data(config["SOURCE_FILE"]+"/" +doc)
                pipeline = build_pipeline(llm, embeding, vector_store=vector_store)
                # 暂时停止实时索引
                client.update_collection(
                    collection_name=config["COLLECTION_NAME"]+doc,
                    optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
                )
                pipeline.run(documents=data, show_progress=False, num_workers=1)
                # 恢复实时索引
                client.update_collection(
                    collection_name=config["COLLECTION_NAME"]+doc,
                    optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
                )
                print(len(data))
            # retriever = QdrantRetriever(vector_store, embeding, similarity_top_k=3)
            retriever = QdrantRetriever(vector_store, embeding, similarity_top_k=50)
            retrievers[doc] = retriever
        # retriever = QdrantRetriever(vector_store, embeding, similarity_top_k=3)
        retriever = QdrantRetriever(vector_store, embeding, similarity_top_k=50)
        return llm,reranker,retrievers
        
async def main(llm,reranker,retrievers):
    config = dotenv_values(".env")
    print(config)
    queries = read_jsonl("question.jsonl")
    # 生成答案
    print("Start generating answers...")
    results = []
    recalls = []
    if (config["IS_DOCUMENT"]=='False'):
        for query in tqdm(queries, total=len(queries)):
            result,recall = await generation_with_knowledge_retrieval(
                query["query"], retrievers, llm
            )
            results.append(result)
            recalls.append(recall)
    else:
        for query in tqdm(queries, total=len(queries)):
            result,recall = await generation_with_knowledge_retrieval(
                query["query"], retrievers[query['document']], llm, reranker=reranker
            )
            results.append(result)
            recalls.append(recall)

        # 处理结果
    save_answers(queries, results, recalls,"submit_hybrid.jsonl")
                

if __name__ == "__main__":
    post_run()
    asyncio.run(main())
