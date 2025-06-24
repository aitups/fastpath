import tempfile
import yaml
from fastpath.config import Config

def test_config_get():
    data = {"llm_provider": "ollama", "llm_config": {"model": "llama3:8b"}}
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as tmp:
        yaml.dump(data, tmp)
        tmp.flush()
        config = Config(path=tmp.name)
        assert config.get("llm_provider") == "ollama"
        assert config.get("llm_config")["model"] == "llama3:8b" 