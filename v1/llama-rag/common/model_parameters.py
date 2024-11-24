import os

from llama_index.core import ChatPromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class ModelParameters:
  # Lazily loaded llm
  llm = None

  # Lazily loaded embed model
  embed_model: HuggingFaceEmbedding = None

  # Set the location that we will persist our embeds
  persist_dir = "query_engine_vectors"

  # Set the location to cache the embed model
  embed_cache_folder = "vector_model"

  # Set the location to store the llm
  llm_cache_folder = "model"

  # Create the prompt query templates to sanitise hallucinations
  prompt_template = ChatPromptTemplate.from_messages([
    (
      "system",
      "If the context is not useful, respond with 'I'm not sure'.",
    ),
    (
      "user",
      (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge "
        "answer the question: {query_str}\n."
      )
    ),
  ])

  def __init__(self):
    # Lazily load the locally stored Llama model
    self._llm = None
    # Lazily load the Embed from Hugging Face model
    self._embed_model = None

  @property
  def llm(self):
    from llama_index.llms.llama_cpp import LlamaCPP

    if self._llm is None:
      print("Initializing Llama CPP Model...")
      self._llm = LlamaCPP(
        model_url=None,
        model_path=f"{self.llm_cache_folder}/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        temperature=0.7,
        verbose=False,
      )
    return self._llm
  
  @property
  def embed_model(self):
    if self._embed_model is None:
      print("Initializing Embed Model...")
      self._embed_model = HuggingFaceEmbedding(
        model_name="./vector_model", 
        cache_folder=self.embed_cache_folder
      )
    return self._embed_model
