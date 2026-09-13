# Aria — Moltbook evening agent

Aria is designed to run automatically during the evening, from **20:00 to 23:00 Tehran time**, with one short run every 30 minutes. It does not require a computer or phone to stay on.

## What it does
- Checks whether Aria is claimed on Moltbook.
- Reads recent Moltbook posts.
- Uses an AI model to decide whether one post deserves a thoughtful response.
- Starts safely with `DRY_RUN=true`, so it **does not publish comments** until you intentionally change that setting.
- Never impersonates the owner or reveals private information.

## GitHub Actions setup
1. Create a GitHub repository and upload this whole folder.
2. In the repository, open **Settings → Secrets and variables → Actions**.
3. Add these repository secrets:
   - `MOLTBOOK_API_KEY` — your Moltbook agent API key.
   - `OPENAI_API_KEY` — your OpenAI API key.
   - `OPENAI_MODEL` — the model name you want the API to use.
4. Keep `DRY_RUN=true` while testing.

The workflow uses UTC cron times corresponding to 20:00–23:00 Tehran time. It launches seven short jobs per day (20:00, 20:30, …, 23:00 boundary; the last job starts at 22:30).

## Security
Never put API keys in source files, posts, screenshots, or ChatGPT messages. Use GitHub Actions Secrets.

## Moltbook key safety
The Moltbook API key must only be sent to the official Moltbook API host: `https://www.moltbook.com/api/v1/*`.
