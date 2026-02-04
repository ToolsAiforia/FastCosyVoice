stop:
	@docker compose stop tts
down:
	@docker compose down tts
build:
	@docker compose build tts

run: stop down build
	@docker compose up -d --remove-orphans tts

logs:
	@docker compose logs -f tts

trt:
	@docker compose up trt
