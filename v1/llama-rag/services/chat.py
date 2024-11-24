import os

from common.model_parameters import ModelParameters

from nitric.resources import websocket
from nitric.context import WebsocketContext
from nitric.application import Nitric
from llama_index.core import StorageContext, load_index_from_storage, Settings

os.environ['HF_HOME'] = ModelParameters.embed_cache_folder
os.environ['TRANSFORMERS_CACHE'] = ModelParameters.embed_cache_folder

socket = websocket("socket")

@socket.on("connect")
async def on_connect(ctx):
  # handle connections
  print(f"socket connected with {ctx.req.connection_id}")
  return ctx


@socket.on("disconnect")
async def on_disconnect(ctx):
  # handle disconnections
  print(f"socket disconnected with {ctx.req.connection_id}")
  return ctx


@socket.on("message")
async def on_message(ctx: WebsocketContext):
  response = await query_model(ctx.req.data.decode("utf-8"))

  await socket.send(ctx.req.connection_id, response.encode("utf-8"))


async def query_model(prompt: str) -> str:
  params = ModelParameters()

  Settings.llm = params.llm
  Settings.embed_model = params.embed_model

  # Get the model from the stored local context
  if os.path.exists(ModelParameters.persist_dir):
    print("Loading model from storage...")
    storage_context = StorageContext.from_defaults(persist_dir=params.persist_dir)

    index = load_index_from_storage(storage_context)
  else:
    print("model does not exist")
    return

  # Get the query engine from the index, and use the prompt template for santisation.
  query_engine = index.as_query_engine(
    streaming=False, 
    similarity_top_k=4, 
    text_qa_template=params.prompt_template
  )

  print(f"Querying model: \"{prompt}\"")

  # Query the model
  query_resp = query_engine.query(prompt)

  print(f"Response: \n{query_resp}")

  return query_resp.response

Nitric.run()