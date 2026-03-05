# Functionality

## Done

[x] Remove the limit of 7 results per source. Return 10 most relevant results across all sources.
[x] Add the date a video or podcast episode was published to the Telegram reply. If it's not stored in the cache yet, adjust this.
[x] Add /rebuild command for full cache rebuild (vs incremental /refresh). Fix YouTube date display and add end-to-end test.
[x] Add video/podcast duration and YouTube view count next to the published date
[x] Make embedding model truly local (local_files_only=True) and add /updatemodel command
[x] Limit results to at most 3 per creator to ensure variety

## To do

[] Include the list of the available commands (together with their descriptions) with the Telegram bot (to be read in Telegram)
[] Prepare for multi-user
    [] Upload a list of creators via Telegram (limit to 500 at most)
    [] Share cache across all users (make it per-channel?)
    [] Do not assume the size of creators file when calculating YouTube quota
[] Allow to add a creator from Telegram bot (does it need AI to enrich things?)

# Development

## Done

[x] Optimize YouTube quota usage via caching
[x] Add tests to ensure future changes don't break the bot
[x] Reorganize project into src/, tests/, data/ folders
[x] Store local logs to help with future troubleshooting
[x] Production hardening: --dev/prod split, auth check, XXE fix, HTTP timeouts, graceful shutdown, HTML escaping, pinned deps
[x] Cleanup after ONNX migration: remove double normalization, unused kwargs, redundant astype, cache token_type_ids check
[x] Restore embedding progress bar lost during ONNX migration
[x] Fix creators.csv: add check_creators.py CLI tool, fix 33 YouTube channel IDs, replace 3 expired Pushkin+ feeds with public Omnycontent feeds
[x] Move cache files from data/ to separate cache/ directory
[x] Add extract_subscriptions.py script to convert YouTube subscriptions HTML export to creators.csv format
[x] Include creators_sample.csv file in the repo
[x] Add --refresh-youtube and --refresh-podcasts CLI flags to refresh caches independently


## To do

[] After every cache refresh or rebuild, add a logging message showing how much each cache file weighs and the total size of all the files.
[] Add a command to check the current daily YouTube quota available (dev only)

# Deployment

## Done

[x] Prepare for DigitalOcean deployment: --refresh CLI flag, systemd service + daily timer, cache size warning, README deployment guide
[x] Upgrade Droplet to 2 GB RAM and 50 GB SSD
[x] Upgrade Python to 3.14 and upgrade related dependencies (numpy)
[x] Deploy bot to the upgraded Droplet
[x] Update bot remotely from my dev computer (git pull + restart, rebuild if necessary)
[x] Add `make deploy-csv` command to upload a CSV and rebuild caches on the server
[x] Fix OOM issue on cache rebuild

## Done

[x] Pin embedding model revision and add `make deploy-cache` to upload pre-built caches from dev machine

## To do


