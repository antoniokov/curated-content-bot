-include .env.dev

ifndef SERVER
  $(error SERVER is not set. Create .env.dev with SERVER=<your-server-ip>)
endif

.PHONY: deploy deploy-rebuild deploy-csv test

deploy:
	ssh root@$(SERVER) 'cd /opt/curated-content-bot && git pull && systemctl restart curated-content-bot'

deploy-rebuild:
	ssh root@$(SERVER) 'cd /opt/curated-content-bot && git pull && systemctl restart curated-content-bot && systemctl start daily-refresh'

CSV ?= data/creators.csv
NAME ?= $(notdir $(CSV))

deploy-csv:
	scp $(CSV) root@$(SERVER):/opt/curated-content-bot/data/$(NAME)
	ssh root@$(SERVER) 'chown deploy:deploy /opt/curated-content-bot/data/$(NAME) && cd /opt/curated-content-bot && rm -rf cache/* && sudo -u deploy .venv/bin/python3 bot.py --refresh'

test:
	python3 -m pytest tests/test_bot.py -v
