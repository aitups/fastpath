import pytest
import asyncio
from unittest.mock import MagicMock
from fastpath.intent_modeler import IntentModeler


def run_async(coro):
    return asyncio.run(coro)


def test_extract_user_intent_valid_json():
    llm = MagicMock()
    llm.generate.return_value = (
        '{"intent_type": "investigation", "required_resources": ["web", "kb"], "urgency": "high"}'
    )
    modeler = IntentModeler(llm_provider=llm)

    result = run_async(modeler.extract_user_intent("Investiga las tendencias de IA"))

    assert result == {
        "intent_type": "investigation",
        "required_resources": ["web", "kb"],
        "urgency": "high",
    }
    # Se usó generate() (API síncrona del provider real)
    llm.generate.assert_called_once()


def test_extract_user_intent_json_in_markdown():
    """El parseo robusto debe extraer JSON envuelto en bloques de markdown."""
    llm = MagicMock()
    llm.generate.return_value = (
        "Aquí está el análisis:\n"
        "```json\n"
        '{"intent_type": "management", "required_resources": ["calendar"], "urgency": "medium"}\n'
        "```"
    )
    modeler = IntentModeler(llm_provider=llm)

    result = run_async(modeler.extract_user_intent("Organiza mi agenda"))

    assert result["intent_type"] == "management"
    assert result["required_resources"] == ["calendar"]
    assert result["urgency"] == "medium"


def test_extract_user_intent_invalid_response_uses_default():
    llm = MagicMock()
    llm.generate.return_value = "No tengo ni idea de qué quieres."
    modeler = IntentModeler(llm_provider=llm)

    result = run_async(modeler.extract_user_intent("algo raro"))

    assert result == {
        "intent_type": "general",
        "required_resources": [],
        "urgency": "medium",
    }


def test_extract_user_intent_python_dict_from_mock():
    """Algunos mocks devuelven str(dict) en vez de JSON; se soporta vía literal_eval."""
    llm = MagicMock()
    llm.generate.return_value = str({"intent_type": "booking", "required_resources": [], "urgency": "low"})
    modeler = IntentModeler(llm_provider=llm)

    result = run_async(modeler.extract_user_intent("Reserva una sala"))

    assert result["intent_type"] == "booking"
    assert result["urgency"] == "low"


def test_extract_user_intent_with_history():
    llm = MagicMock()
    llm.generate.return_value = '{"intent_type": "analysis", "required_resources": [], "urgency": "medium"}'
    modeler = IntentModeler(llm_provider=llm)

    run_async(modeler.extract_user_intent("Analiza", history=[{"role": "user", "content": "previo"}]))

    # El historial debe incluirse en el prompt
    prompt = llm.generate.call_args[0][0]
    assert "Tarea principal: Analiza" in prompt


def test_analyze_sentiment_valid():
    llm = MagicMock()
    llm.generate.return_value = '{"sentiment": "positive", "score": 0.9}'
    modeler = IntentModeler(llm_provider=llm)

    result = run_async(modeler.analyze_sentiment("Me encanta esta herramienta"))

    assert result == {"sentiment": "positive", "score": 0.9}


def test_analyze_sentiment_fallback_on_error():
    llm = MagicMock()
    llm.generate.side_effect = Exception("LLM down")
    modeler = IntentModeler(llm_provider=llm)

    result = run_async(modeler.analyze_sentiment("texto"))

    assert result == {"sentiment": "neutral", "score": 0.5}


def test_analyze_sentiment_angle_brackets_stripped():
    """El parseo robusto extrae JSON aunque venga precedido de texto y con indentación."""
    llm = MagicMock()
    llm.generate.return_value = (
        "El análisis es:\n"
        '{\n  "sentiment": "negative",\n  "score": -0.6\n}'
    )
    modeler = IntentModeler(llm_provider=llm)

    result = run_async(modeler.analyze_sentiment("Esto es horrible"))

    assert result["sentiment"] == "negative"
    assert result["score"] == -0.6