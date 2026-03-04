# Plan

## Done

[x] Optimize YouTube quota usage via caching
[x] Add tests to ensure future changes don't break the bot
[x] Reorganize project into src/, tests/, data/ folders
[x] Store local logs to help with future troubleshooting
[x] Remove the limit of 7 results per source. Return 10 most relevant results across all sources.
[x] Add the date a video or podcast episode was published to the Telegram reply. If it's not stored in the cache yet, adjust this.
[x] Add /rebuild command for full cache rebuild (vs incremental /refresh). Fix YouTube date display and add end-to-end test.
[x] Add video/podcast duration and YouTube view count next to the published date
[x] Make embedding model truly local (local_files_only=True) and add /updatemodel command
[x] Production hardening: --dev/prod split, auth check, XXE fix, HTTP timeouts, graceful shutdown, HTML escaping, pinned deps

## TODO
[] Ensure  easy updates with minimalistic CI/CD
[] Deploy to my personal DigitalOcean server
[] Include the list of the available commands (together with their descriptions) with the Telegram bot (to be read in Telegram)
[] Swap private podcast feeds that are unreachable with their public equivalents
[] Add a command to check the current daily YouTube quota available
[] Allow to add a creator from Telegram bot (does it need AI to enrich things?)
