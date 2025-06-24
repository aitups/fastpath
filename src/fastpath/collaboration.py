from typing import Any, List

class Message:
    def __init__(self, sender: str, content: Any):
        self.sender = sender
        self.content = content

class MessageBus:
    """
    Bus de mensajes para colaboración en tiempo real entre agentes.
    """
    def __init__(self):
        self.messages: List[Message] = []

    def post(self, message: Message):
        self.messages.append(message)

    def get_all(self) -> List[Message]:
        return self.messages 