# Comandos comuns. Use `make` para listar.

.PHONY: help install run test seed lint format clean

help:
	@echo "Alvos disponíveis:"
	@echo "  install  - instala dependências (requirements.txt)"
	@echo "  run      - sobe o app Streamlit"
	@echo "  test     - roda pytest"
	@echo "  seed     - sobe parquet local de embeddings para o Snowflake (one-time)"
	@echo "  lint     - lint via ruff"
	@echo "  format   - formata código via ruff"
	@echo "  clean    - remove caches"

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

test:
	pytest tests/

seed:
	python scripts/seed_snowflake.py

lint:
	ruff check src tests app.py streamlit_app scripts

format:
	ruff format src tests app.py streamlit_app scripts

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
