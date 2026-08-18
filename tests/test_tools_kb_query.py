import pytest
from unittest.mock import MagicMock
from fastpath.tools import KnowledgeBaseQueryTool


def test_kb_query_tool_init_defaults():
    tool = KnowledgeBaseQueryTool()
    assert tool.name == "knowledge_base_query"
    assert "Base de Conocimiento" in tool.description


def test_kb_query_tool_formats_results():
    # KB fake que devuelve resultados estructurados
    fake_kb = MagicMock()
    fake_kb.query.return_value = [
        {"text": "FastPath es un framework de agentes.", "score": 0.95},
        {"text": "Soporta memoria persistente.", "score": 0.82},
    ]

    tool = KnowledgeBaseQueryTool(knowledge_base=fake_kb)
    result = tool.run("¿Qué es FastPath?")

    fake_kb.query.assert_called_once_with("¿Qué es FastPath?", top_k=5)
    assert "FastPath es un framework de agentes." in result
    assert "score: 0.9500" in result
    assert "Soporta memoria persistente." in result


def test_kb_query_tool_empty_results():
    fake_kb = MagicMock()
    fake_kb.query.return_value = []

    tool = KnowledgeBaseQueryTool(knowledge_base=fake_kb)
    result = tool.run("nada")
    assert result == "No se encontraron resultados en la Knowledge Base."


def test_kb_query_tool_empty_kb_error():
    # KnowledgeBase.query lanza ValueError si no hay datos ingeridos
    fake_kb = MagicMock()
    fake_kb.query.side_effect = ValueError("Knowledge base is empty. Ingest data first.")

    tool = KnowledgeBaseQueryTool(knowledge_base=fake_kb)
    result = tool.run("pregunta")
    assert result.startswith("Error:")


def test_kb_query_tool_custom_top_k():
    fake_kb = MagicMock()
    fake_kb.query.return_value = [
        {"text": "Resultado 1", "score": 0.9},
        {"text": "Resultado 2", "score": 0.8},
        {"text": "Resultado 3", "score": 0.7},
    ]

    tool = KnowledgeBaseQueryTool(knowledge_base=fake_kb, top_k=3)
    result = tool.run("consulta", top_k=2)

    # top_k explícito en run() tiene prioridad
    fake_kb.query.assert_called_once_with("consulta", top_k=2)
    # El resultado contiene los 3 items (separados por líneas en blanco)
    non_empty_lines = [l for l in result.splitlines() if l.strip()]
    assert len(non_empty_lines) == 3
