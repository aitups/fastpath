from fastpath.collaboration import Message, MessageBus

def test_message_creation():
    msg = Message(sender="agente1", content="hola")
    assert msg.sender == "agente1"
    assert msg.content == "hola"

def test_message_bus_post_and_get():
    bus = MessageBus()
    msg1 = Message(sender="a1", content="msg1")
    msg2 = Message(sender="a2", content="msg2")
    bus.post(msg1)
    bus.post(msg2)
    messages = bus.get_all()
    assert messages[0].content == "msg1"
    assert messages[1].sender == "a2" 