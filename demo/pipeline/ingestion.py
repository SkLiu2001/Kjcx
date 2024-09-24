from llama_index.core import SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.extractors import SummaryExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms.llm import LLM
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, MetadataMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores import VectorStoreQueryResult
from qdrant_client import AsyncQdrantClient, models,QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from custom.template import SUMMARY_EXTRACT_TEMPLATE
from custom.transformation import CustomFilePathExtractor, CustomTitleExtractor

from typing import Any, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from FlagEmbedding import BGEM3FlagModel
import textract
import os

from tqdm import tqdm

def read_data(path: str = "data") -> list[Document]:
    print("path",path)
    reader = SimpleDirectoryReader(
        input_dir=path,
        recursive=True,
        required_exts=[
            ".pdf",
        ],
    )
    return reader.load_data()

# def read_data(path: str = "data") -> list[Document]:
#     print("path", path)

#     # 获取目录中的所有 .doc 文件
#     doc_files = []
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             if file.endswith(".doc"):
#                 doc_files.append(os.path.join(root, file))

#     # 使用 textract 读取每个 .doc 文件的内容
#     documents = []
#     for doc_file in doc_files:
#         text = textract.process(doc_file)
#         documents.append(Document(text.decode('utf-8')))

#     return documents


# sparse_tokenizer = AutoTokenizer.from_pretrained(
#     "/home/liusk-s24/data/model/bge_m3"
# )
# sparse_model = AutoModelForMaskedLM.from_pretrained(
#     "/home/liusk-s24/data/model/bge_m3"
# )

sparse_model = BGEM3FlagModel('/home/liusk-s24/data/model/bge_m3', use_fp16=True) 

doc_tokenizer = AutoTokenizer.from_pretrained(
    "/home/liusk-s24/data/model/efficient-splade-VI-BT-large-doc"
)
doc_model = AutoModelForMaskedLM.from_pretrained(
    "/home/liusk-s24/data/model/efficient-splade-VI-BT-large-doc"
)

query_tokenizer = AutoTokenizer.from_pretrained(
    "/home/liusk-s24/data/model/efficient-splade-VI-BT-large-query"
)
query_model = AutoModelForMaskedLM.from_pretrained(
    "/home/liusk-s24/data/model/efficient-splade-VI-BT-large-query"
)
# def sparse_vectors_fn(
#     texts: List[str],
# ) -> Tuple[List[List[int]], List[List[float]]]:
#     """
#     Computes vectors from logits and attention mask using ReLU, log, and max operations.
#     """
#     tokens = sparse_tokenizer(
#         texts, truncation=True, padding=True, return_tensors="pt"
#     )
#     if torch.cuda.is_available():
#         tokens = tokens.to("cpu")

#     output = sparse_model(**tokens)
#     logits, attention_mask = output.logits, tokens.attention_mask
#     relu_log = torch.log(1 + torch.relu(logits))
#     weighted_log = relu_log * attention_mask.unsqueeze(-1)
#     tvecs, _ = torch.max(weighted_log, dim=1)

#     # extract the vectors that are non-zero and their indices
#     indices = []
#     vecs = []
#     for batch in tvecs:
#         indices.append(batch.nonzero(as_tuple=True)[0].tolist())
#         vecs.append(batch[indices[-1]].tolist())

#     return indices, vecs

def sparse_vectors_fn(
    texts: List[str],
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Computes vectors from logits and attention mask using ReLU, log, and max operations.
    """
    outputs = sparse_model.encode(texts, return_dense=False, return_sparse=True, return_colbert_vecs=False)

    # extract the vectors that are non-zero and their indices
    indices = []
    vecs = []
    
    for sentence in outputs["lexical_weights"]:
        key_list = [int(key) for key in sentence.keys()]
        value_list = list(sentence.values())
        indices.append(key_list)
        vecs.append(value_list)

    return indices, vecs

# def sparse_vectors_fn(texts: List[str], batch_size: int = 1) -> Tuple[List[List[int]], List[List[float]]]:
#     """
#     Computes vectors from logits and attention mask using ReLU, log, and max operations.
#     """
#     indices = []
#     vecs = []

#     if torch.cuda.is_available():
#         sparse_model.to("cuda:1")

#     for i in tqdm(range(0, len(texts), batch_size)):
#         batch_texts = texts[i:i + batch_size]
#         tokens = sparse_tokenizer(
#             batch_texts, truncation=True, padding=True, return_tensors="pt"
#         )
#         if torch.cuda.is_available():
#             tokens = {key: value.to("cuda:1") for key, value in tokens.items()}

#         output = sparse_model(**tokens)
#         logits, attention_mask = output.logits, tokens['attention_mask']
#         relu_log = torch.log(1 + torch.relu(logits))
#         weighted_log = relu_log * attention_mask.unsqueeze(-1)
#         tvecs, _ = torch.max(weighted_log, dim=1)

#         for batch in tvecs:
#             indices.append(batch.nonzero(as_tuple=True)[0].tolist())
#             vecs.append(batch[indices[-1]].tolist())

#         # 释放不再需要的张量
#         del tokens, output, logits, attention_mask, relu_log, weighted_log, tvecs
#         torch.cuda.empty_cache()

#     return indices, vecs

def sparse_doc_vectors(texts: List[str], batch_size: int = 1) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Computes vectors from logits and attention mask using ReLU, log, and max operations.
    """
    indices = []
    vecs = []

    if torch.cuda.is_available():
        doc_model.to("cuda:1")

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        tokens = doc_tokenizer(
            batch_texts, truncation=True, padding=True, return_tensors="pt"
        )
        if torch.cuda.is_available():
            tokens = {key: value.to("cuda:1") for key, value in tokens.items()}

        output = doc_model(**tokens)
        logits, attention_mask = output.logits, tokens['attention_mask']
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        tvecs, _ = torch.max(weighted_log, dim=1)

        for batch in tvecs:
            indices.append(batch.nonzero(as_tuple=True)[0].tolist())
            vecs.append(batch[indices[-1]].tolist())

        # 释放不再需要的张量
        del tokens, output, logits, attention_mask, relu_log, weighted_log, tvecs
        torch.cuda.empty_cache()

    return indices, vecs

def sparse_query_vectors(texts: List[str], batch_size: int = 1) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Computes vectors from logits and attention mask using ReLU, log, and max operations.
    """
    indices = []
    vecs = []

    if torch.cuda.is_available():
        query_model.to("cuda:1")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = query_tokenizer(
            batch_texts, truncation=True, padding=True, return_tensors="pt"
        )
        if torch.cuda.is_available():
            tokens = {key: value.to("cuda:1") for key, value in tokens.items()}

        output = query_model(**tokens)
        logits, attention_mask = output.logits, tokens['attention_mask']
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        tvecs, _ = torch.max(weighted_log, dim=1)

        for batch in tvecs:
            indices.append(batch.nonzero(as_tuple=True)[0].tolist())
            vecs.append(batch[indices[-1]].tolist())

        # 释放不再需要的张量
        del tokens, output, logits, attention_mask, relu_log, weighted_log, tvecs
        torch.cuda.empty_cache()

    return indices, vecs


def build_pipeline(
    llm: LLM,
    embed_model: BaseEmbedding,
    template: str = None,
    vector_store: BasePydanticVectorStore = None,
) -> IngestionPipeline:
    transformation = [
        SentenceSplitter(chunk_size=4096, chunk_overlap=50),
        CustomTitleExtractor(metadata_mode=MetadataMode.EMBED),
        CustomFilePathExtractor(last_path_length=4, metadata_mode=MetadataMode.EMBED),
        # SummaryExtractor(
        #     llm=llm,
        #     metadata_mode=MetadataMode.EMBED,
        #     prompt_template=template or SUMMARY_EXTRACT_TEMPLATE,
        # ),
        embed_model,
    ]

    return IngestionPipeline(transformations=transformation, vector_store=vector_store)


def build_vector_store(
    config: dict, reindex: bool = False,is_doc:bool =False
) -> tuple[AsyncQdrantClient, QdrantVectorStore]:
    # client = AsyncQdrantClient(
    #     # url=config["QDRANT_URL"],
    #     #location=":memory:"
    #     path="vs"
    # )
    client = QdrantClient(
        # url=config["QDRANT_URL"],
        #location=":memory:"
        path="vs"
    )
    if (is_doc == False):
        collection_name = config["COLLECTION_NAME"] 
        if reindex:
            try:
                client.delete_collection(config["COLLECTION_NAME"] or "aiops24")
            except UnexpectedResponse as e:
                print(f"Collection not found: {e}")

        try:
            # Check if the collection exists
            client.get_collection(collection_name)
            print(f"{collection_name} already exists")
        except ValueError as e:
            # If the collection does not exist, create it
            client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "text-dense":models.VectorParams(size=config["VECTOR_SIZE"] or 1024, distance=models.Distance.DOT)
                        },
                    sparse_vectors_config={
                        "text-sparse-new": models.SparseVectorParams(index=models.SparseIndexParams())
                    },
                )
            print(f"{collection_name} created")
        return client, QdrantVectorStore(
                client  = client,
                collection_name=collection_name,
                parallel=4,
                batch_size=32,
                enable_hybrid=True,
                sparse_doc_fn=sparse_vectors_fn,
                sparse_query_fn=sparse_vectors_fn,
            )
    else:
        collection_names = [
        config["COLLECTION_NAME"] + "director",
        config["COLLECTION_NAME"] + "emsplus",
        config["COLLECTION_NAME"] + "rcp",
        config["COLLECTION_NAME"] + "umac"
        ]
        vector_stores = []
        for collection_name in collection_names:
            if reindex:
                try:
                    client.delete_collection(collection_name)
                except UnexpectedResponse as e:
                    print(f"{collection_name} not found: {e}")

            try:
                # Check if the collection exists
                client.get_collection(collection_name)
                print(f"Collection {collection_name} already exists")
            except ValueError as e:
                # If the collection does not exist, create it
                print(f"Unexpected response when checking collection {collection_name}: {e}")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "text-dense":models.VectorParams(size=config["VECTOR_SIZE"] or 1024, distance=models.Distance.DOT)
                        },
                    sparse_vectors_config={
                        "text-sparse-new": models.SparseVectorParams(index=models.SparseIndexParams())
                    },
                )
                print(f"Collection {collection_name} created")
            # vector_store = QdrantVectorStore(
            #     aclient=client,
            #     collection_name=collection_name,
            #     parallel=4,
            #     batch_size=32,
            # )
            vector_store = QdrantVectorStore(
                client  = client,
                collection_name=collection_name,
                parallel=4,
                batch_size=32,
                enable_hybrid=True,
                sparse_doc_fn=sparse_vectors_fn,
                sparse_query_fn=sparse_vectors_fn,
            )
            vector_stores.append(vector_store)
        
        return client, vector_stores
