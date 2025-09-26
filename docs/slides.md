# FastPath — Overview & Usage

## 1. Objetivo
Framework "agentic" para orquestar agentes habilitados con LLM, herramientas y memoria.
Modos soportados: sequential, collaborative, smart, hierarchical.

---

## 2. Arquitectura (visual)
(Puedes exportar la imagen desde docs/diagrams/architecture.mmd)

```mermaid
%% included via docs/diagrams/architecture.mmd
```

---

## 3. Diagrama de clases (resumen)
(Puedes exportar la imagen desde docs/diagrams/classes.mmd)

```mermaid
%% included via docs/diagrams/classes.mmd
```

---

## 4. Flujo jerárquico (ejecución)
(Puedes exportar la imagen desde docs/diagrams/sequence.mmd)

```mermaid
%% included via docs/diagrams/sequence.mmd
```

---

## 5. Quick start (para slides)
- Requisitos: Poetry, .env con HUGGINGFACE_API_TOKEN, SERPER_API_KEY (si usas Serper).
- Instalar dependencias:
  - PowerShell:
    - poetry install
- Ejecutar ejemplo (main.py):
  - poetry run python src/main.py
- Ejecutar tests unitarios:
  - poetry run pytest -q
  - (Las tests de integración requieren Docker; están marcadas y pueden fallar sin Docker)

---

## 6. Cómo usar Orchestrator (ejemplo)
- Inicializar:
  - Orchestrator(agents=[...], mode="hierarchical" | "smart" | "sequential" | "collaborative")
- Smart / Hierarchical:
  - smart: Orchestrator tiene LLM que genera plan JSON. Strategy asigna y ejecuta.
  - hierarchical: plan con subtareas {id,name,depends_on}, se valida grafo, se ejecuta en orden topológico.
- Logging: timestamps y nombre de clase por defecto.

---

## 7. Notas para la presentación
- Mantener las imágenes en docs/images/ y referenciarlas con `![...](/docs/images/archivo.png)`.
- Para mostrar pasos: usa la diapositiva "Quick start" y la de "Cómo usar Orchestrator".
