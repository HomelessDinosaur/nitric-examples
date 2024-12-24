from nitric.resources import websocket, topic, kv

socket = websocket("socket")
chat_topic = topic("chat")
connections = kv("connections")