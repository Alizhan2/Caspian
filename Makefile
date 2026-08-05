.PHONY: dev test lint format train demo down
dev:
	docker compose up --build
test:
	cd backend && python -m pytest
lint:
	cd backend && ruff check app tests
	cd frontend && npm run lint
format:
	cd backend && ruff format app tests training
train:
	python -m training.train
demo:
	docker compose up --build
down:
	docker compose down
