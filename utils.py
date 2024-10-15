from typing import Annotated, Literal, Optional
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import tools_condition,ToolNode
from zhipuai import ZhipuAI
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
from typing import Callable
from dotenv import dotenv_values
config = dotenv_values(".env")
chromadb_client = chromadb.PersistentClient(path='Chroma_database')
# chromadb_client = chromadb.HttpClient(host='47.95.203.141', port=11596)
class ZhiPuEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model_client = ZhipuAI(api_key=config["GLM_KEY"]) 
        self.model_name = "embedding-2"
        super(ZhiPuEmbeddingFunction,self).__init__()
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        response = self.model_client.embeddings.create(
            model=self.model_name,
            input = input,
        )
        embeddings = [resp.embedding for resp in response.data]
        return embeddings
        # return
def get_retriever(collection,top_k=5,include=None):
    if not include:
        include=["documents","metadatas"]
    def fn(query_text):
        res = collection.query(query_texts=[query_text],n_results=top_k,include=include)
        new_res = {}
        for key in include:
            new_res[key] = res[key][0]
        return new_res
    return fn
