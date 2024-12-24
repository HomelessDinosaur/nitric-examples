import os


# Convert the messages into Llama 3.1 format
def messages_to_prompt(messages):
  prompt = ""
  for message in messages:
      if message.role == 'system':
        prompt += f"<|system|>\n{message.content}</s>\n"
      elif message.role == 'user':
        prompt += f"<|user|>\n{message.content}</s>\n"
      elif message.role == 'assistant':
        prompt += f"<|assistant|>\n{message.content}</s>\n"

  # ensure we start with a system prompt, insert blank if needed
  if not prompt.startswith("<|system|>\n"):
      prompt = "<|system|>\n</s>\n" + prompt

  # add final assistant prompt
  prompt = prompt + "<|assistant|>\n"

  return prompt

# Convert the completed prompt into Llama 3.1 format
def completion_to_prompt(completion):
  return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"

class ModelParameters:
  # Lazily loaded llm
  _llm = None

  # Lazily loaded embed model
  _embed_model = None

  # Lazily loaded tokenizer
  _tokenizer = None

  # Set the location that we will persist our embeds
  persist_dir = "./models/query_engine_db"

  # Set the location to cache the embed model
  embed_cache_folder = os.getenv("HF_CACHE") or "./models/vector_model_cache"

  # Set the location to store the llm
  llm_cache_folder = "./models/llm_cache"

  # Create the prompt query templates to sanitise hallucinations
  prompt_template = (
    "Context information is below. If the context is not useful, respond with 'I'm not sure'. "
    "Given the context information and not prior knowledge answer the prompt.\n"
    "{context_str}\n"
  )


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
        model_path=f"{self.llm_cache_folder}/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        temperature=0.7,
        # Increase for longer responses
        max_new_tokens=512,
        context_window=3900,
        generate_kwargs={},
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": 1},
        # transform inputs into Llama3.1 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=False,
      )
    return self._llm
  
  @property
  def embed_model(self):
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    if self._embed_model is None:
      print("Initializing Embed Model...")
      self._embed_model = HuggingFaceEmbedding(
        model_name=self.embed_cache_folder, 
        cache_folder=self.embed_cache_folder
      )
    return self._embed_model
  
  @property
  def tokenizer(self):
    from transformers import AutoTokenizer

    if self._tokenizer is None:
      print("Initializing Tokenizer")
      self._tokenizer = AutoTokenizer.from_pretrained(
        "pcuenq/Llama-3.2-1B-Instruct-tokenizer"
      ).encode
    return self._tokenizer