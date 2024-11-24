import os
from urllib.request import urlretrieve

from common.model_parameters import ModelParameters

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from huggingface_hub import snapshot_download

os.environ['HF_HOME'] = ModelParameters.embed_cache_folder
os.environ['TRANSFORMERS_CACHE'] = ModelParameters.embed_cache_folder

def build_query_engine(input_dir_path = "../docs/docs"):
    params = ModelParameters()

    Settings.llm = params.llm
    Settings.embed_model = params.embed_model

    # load data
    loader = SimpleDirectoryReader(
        input_dir = input_dir_path,
        required_exts=[".mdx"],
        recursive=True
    )
    docs = loader.load_data()

    index = VectorStoreIndex.from_documents(docs, show_progress=True)

    index.storage_context.persist(params.persist_dir)


# curl -OL https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
def download_embed_model():
    print(f"Downloading embed model to {ModelParameters.embed_cache_folder}")

    dir = snapshot_download("BAAI/bge-large-en-v1.5",
        local_dir= ModelParameters.embed_cache_folder,
        allow_patterns=[
            "*.json",
            "vocab.txt",
            "onnx",
            "1_Pooling",
            "model.safetensors"
        ]
    )

    print(f"Downloaded model to {dir}")


def download_llm():
    print(f"Downloading llm to {ModelParameters.llm_cache_folder}")

    os.mkdir(ModelParameters.llm_cache_folder)

    llm_name = "Llama-3.2-1B-Instruct"
    llm_path = f"{llm_name}-Q4_K_M.gguf"
    llm_download_url = f"https://huggingface.co/bartowski/{llm_name}-GGUF/resolve/main/{llm_path}"
    llm_dir = urlretrieve(llm_download_url, f"{ModelParameters.llm_cache_folder}/{llm_path}")

    print(f"Download model to {llm_dir[0]}")

download_embed_model()
download_llm()
build_query_engine()