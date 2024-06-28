from llama_index.core import SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.extractors import SummaryExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms.llm import LLM
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, MetadataMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from custom.template import SUMMARY_EXTRACT_TEMPLATE
from custom.transformation import CustomFilePathExtractor, CustomTitleExtractor

from typing import Any, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def read_data(path: str = "data") -> list[Document]:
    print("path",path)
    reader = SimpleDirectoryReader(
        input_dir=path,
        recursive=True,
        required_exts=[
            ".txt",
        ],
    )
    return reader.load_data()



# sparse_tokenizer = AutoTokenizer.from_pretrained(
#     "BAAI/bge-m3"
# )
# sparse_model = AutoModelForMaskedLM.from_pretrained(
#     "BAAI/bge-m3"
# )


def sparse_vectors_fn(
    texts: List[str],
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Computes vectors from logits and attention mask using ReLU, log, and max operations.
    """
    tokens = sparse_tokenizer(
        texts, truncation=True, padding=True, return_tensors="pt"
    )
    if torch.cuda.is_available():
        tokens = tokens.to("cuda:0")

    output = sparse_model(**tokens)
    logits, attention_mask = output.logits, tokens.attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    tvecs, _ = torch.max(weighted_log, dim=1)

    # extract the vectors that are non-zero and their indices
    indices = []
    vecs = []
    for batch in tvecs:
        indices.append(batch.nonzero(as_tuple=True)[0].tolist())
        vecs.append(batch[indices[-1]].tolist())

    return indices, vecs


def sparse_vectors_fn(
    texts: List[str],
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Computes vectors from logits and attention mask using ReLU, log, and max operations.
    """
    tokens = sparse_tokenizer(
        texts, truncation=True, padding=True, return_tensors="pt"
    )
    if torch.cuda.is_available():
        tokens = tokens.to("cuda:0")

    output = sparse_model(**tokens)
    logits, attention_mask = output.logits, tokens.attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    tvecs, _ = torch.max(weighted_log, dim=1)

    # extract the vectors that are non-zero and their indices
    indices = []
    vecs = []
    for batch in tvecs:
        indices.append(batch.nonzero(as_tuple=True)[0].tolist())
        vecs.append(batch[indices[-1]].tolist())

    return indices, vecs



def build_pipeline(
    llm: LLM,
    embed_model: BaseEmbedding,
    template: str = None,
    vector_store: BasePydanticVectorStore = None,
) -> IngestionPipeline:
    transformation = [
        SentenceSplitter(chunk_size=1024, chunk_overlap=50),
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


async def build_vector_store(
    config: dict, reindex: bool = False,is_doc:bool =False
) -> tuple[AsyncQdrantClient, QdrantVectorStore]:
    client = AsyncQdrantClient(
        # url=config["QDRANT_URL"],
        #location=":memory:"
        path="vs"
    )
    if (is_doc == False):
        collection_name = config["COLLECTION_NAME"] 
        if reindex:
            try:
                await client.delete_collection(config["COLLECTION_NAME"] or "aiops24")
            except UnexpectedResponse as e:
                print(f"Collection not found: {e}")

        try:
            # Check if the collection exists
            await client.get_collection(collection_name)
            print("Collection already exists")
        except UnexpectedResponse:
            # If the collection does not exist, create it
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=config["VECTOR_SIZE"] or 1024, distance=models.Distance.DOT
                ),
            )
            print("Collection created")
        return client, QdrantVectorStore(
            aclient=client,
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            parallel=4,
            batch_size=32,
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
                    await client.delete_collection(collection_name)
                except UnexpectedResponse as e:
                    print(f"Collection not found: {e}")

            try:
                # Check if the collection exists
                await client.get_collection(collection_name)
                print(f"Collection {collection_name} already exists")
            except ValueError as e:
                # If the collection does not exist, create it
                print(f"Unexpected response when checking collection {collection_name}: {e}")
                await client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=config["VECTOR_SIZE"] or 1024, distance=models.Distance.DOT
                    ),
                )
                print(f"Collection {collection_name} created")
            vector_store = QdrantVectorStore(
                aclient=client,
                collection_name=collection_name,
                parallel=4,
                batch_size=32,
            )
            # vector_store = QdrantVectorStore(
            #     aclient=client,
            #     collection_name=collection_name,
            #     parallel=4,
            #     batch_size=32,
            #     enable_hybrid=True,
            #     sparse_doc_fn=sparse_vectors_fn,
            #     sparse_query_fn=sparse_vectors_fn,
            # )
            vector_stores.append(vector_store)
        
        return client, vector_stores
