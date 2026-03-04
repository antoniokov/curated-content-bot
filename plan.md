# Plan

## Done

[x] Optimize YouTube quota usage via caching
[x] Add tests to ensure future changes don't break the bot
[x] Reorganize project into src/, tests/, data/ folders
[x] Store local logs to help with future troubleshooting
[x] Remove the limit of 7 results per source. Return 10 most relevant results across all sources.
[x] Add the date a video or podcast episode was published to the Telegram reply. If it's not stored in the cache yet, adjust this.

## TODO


[] Ensure stability and easy updates with minimalistic CI/CD
[] Deploy to my personal DigitalOcean server
[] Add a command to check the current daily YouTube quota available
[] Allow to add a creator from Telegram bot (does it need AI to enrich things?)
