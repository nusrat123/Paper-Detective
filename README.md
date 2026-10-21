# PaperDetective

A Gradio application for comparing two research PDFs and asking questions about their content.

## Features
- Ten-field comparison of research papers using Gemini
- Source-linked claims with page and numeric checks
- Clickable evidence inspection
- PDF-grounded chat with inline citations and progressive display
- CSV and PNG comparison exports

## Run locally
1. Install Python 3.11.
2. Install packages with `pip install -r requirements.txt`.
3. Set `GEMINI_API_KEY` in your environment (see `.env.example`; the app does not read `.env` automatically).
4. Run `python app.py` and open the local Gradio address.

## Deploy on Render
1. Push these files to a GitHub repository.
2. In Render, select **New > Blueprint**, and connect the repository. The `render.yaml` file defines the web service.
3. Choose the Free instance if available and enter `GEMINI_API_KEY` as a secret when prompted.
4. Wait for deployment, then open the `onrender.com` URL.

You can also create a Web Service manually with build command `pip install -r requirements.txt` and start command `python app.py`. Set `GEMINI_API_KEY` under Environment.

## Notes
- The two PDFs are processed per Gradio session and are not kept as a permanent document library.
- The API key must remain private. Public traffic can consume your Gemini quota.
- The numeric audit checks whether numbers appear on the cited page; it is not independent fact-checking.
- The chat's progressive text display starts after Gemini has produced and checked an answer. This is not token-level API streaming.
- The application requires text-extractable PDFs for page-linked evidence.

## AI-use disclosure
Add your course-required disclosure of any AI assistance used during development. Review and understand the code before submission.
