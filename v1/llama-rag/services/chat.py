from common.resources import socket, chat_topic, connections

from nitric.context import WebsocketContext
from nitric.application import Nitric

publishable_chat_topic = chat_topic.allow("publish")
connections_store = connections.allow("set", "delete")

@socket.on("connect")
async def on_connect(ctx: WebsocketContext):
  # handle connections
  await connections_store.set(ctx.req.connection_id, {
    # Store the context related to the connection here
    "context": []
  })
  print(f"socket connected with {ctx.req.connection_id}")
  return ctx


@socket.on("disconnect")
async def on_disconnect(ctx: WebsocketContext):
  # handle disconnections
  await connections_store.delete(ctx.req.connection_id)

  print(f"socket disconnected with {ctx.req.connection_id}")
  return ctx


@socket.on("message")
async def on_message(ctx: WebsocketContext):
  print(f"socket message with {ctx.req.connection_id}")

  await publishable_chat_topic.publish({
    "connection_id": ctx.req.connection_id,
    "prompt": ctx.req.data.decode("utf-8") 
  })

  return ctx

Nitric.run()