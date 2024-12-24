import os

from common.model_parameters import ModelParameters
from common.resources import chat_topic, socket, connections

from nitric.context import MessageContext
from nitric.application import Nitric
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.llms import MessageRole, ChatMessage
from llama_index.core.chat_engine import ContextChatEngine

read_write_connections = connections.allow("get", "set")

@chat_topic.subscribe()
async def query_model(ctx: MessageContext) -> str:
  params = ModelParameters()

  Settings.llm = params.llm
  Settings.embed_model = params.embed_model
  Settings.tokenizer = params.tokenizer

  connection_id = ctx.req.data.get("connection_id")
  prompt = ctx.req.data.get("prompt")

  connection_metadata = await read_write_connections.get(connection_id)

  # Get the model from the stored local context
  if os.path.exists(ModelParameters.persist_dir):
    print("Loading model from storage...")
    storage_context = StorageContext.from_defaults(persist_dir=params.persist_dir)

    index = load_index_from_storage(storage_context)
  else:
    print("model does not exist")
    ctx.res.success = False
    return ctx

  # Create a list of chat messages from the chat history
  chat_history = []
  for chat in connection_metadata.get("context"):
    chat_history.append(
      ChatMessage(
        role=chat.get("role"), 
        content=chat.get("content")
      )
    )
  
  # Create the chat engine
  retriever = index.as_retriever(
    similarity_top_k=4,
  )

  chat_engine = ContextChatEngine.from_defaults(
    retriever=retriever,
    chat_history=chat_history,
    context_template=params.prompt_template,
    streaming=False,
  )

  # Query the model
  assistant_response = chat_engine.chat(f"{prompt}")

  print(f"Response: {assistant_response}")

  # Send the response to the socket connection
  await socket.send(
    connection_id, 
    assistant_response.response.encode("utf-8")
  )

  # Add the context to th connections store
  await read_write_connections.set(connection_id, {
      "context": [
        *connection_metadata.get("context"), 
        {
          "role": MessageRole.USER,
          "content": prompt, 
        }, 
        {
          "role": MessageRole.ASSISTANT,
          "content": assistant_response.response
        }
      ]
    }
  )

  return ctx

Nitric.run()