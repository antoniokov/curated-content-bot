-include .env.dev

ifndef SERVER
  $(error SERVER is not set. Create .env.dev with SERVER=<your-server-ip>)
endif

.PHONY: deploy deploy-rebuild deploy-csv deploy-cache test

deploy:
	ssh root@$(SERVER) 'cd /opt/curated-content-bot && git pull && systemctl restart curated-content-bot'

deploy-rebuild:
	ssh root@$(SERVER) 'cd /opt/curated-content-bot && git pull && systemctl restart curated-content-bot && systemctl start daily-refresh'

CSV ?= data/creators.csv
NAME ?= $(notdir $(CSV))

deploy-csv:
	scp $(CSV) root@$(SERVER):/opt/curated-content-bot/data/$(NAME)
	ssh root@$(SERVER) 'chown deploy:deploy /opt/curated-content-bot/data/$(NAME) && cd /opt/curated-content-bot && rm -rf cache/* && sudo -u deploy .venv/bin/python3 bot.py --refresh'

deploy-cache:
	scp cache/.youtube_cache.json cache/.youtube_embeddings.npz cache/.podcast_cache.json cache/.podcast_embeddings.npz root@$(SERVER):/opt/curated-content-bot/cache/
	ssh root@$(SERVER) 'chown deploy:deploy /opt/curated-content-bot/cache/.*'

test:
	python3 -m pytest tests/test_bot.py -v
