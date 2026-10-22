# PaperDetective

Compare two research PDFs and ask questions with source-page citations using Gemini.

## Run locally

1. Install Python 3.11 and run `pip install -r requirements.txt`.
2. Set `GEMINI_API_KEY` in your environment (do not commit it).
3. Run `python app.py` and open `http://localhost:7860`.

## Deploy on Render

Use the included `render.yaml` or create a Python web service.
Build command: `pip install -r requirements.txt`.
Start command: `python app.py`.
Add `GEMINI_API_KEY` as a secret environment variable.
The optional `GEMINI_MODEL` variable selects the Gemini model.

## Features

- Two-PDF analysis using native PDF input to Gemini.
- Ten-field comparison with original-page evidence inspection.
- CSV and PNG exports.
- Profile-first question answering with inline citation validation.
- Review flags for claims that fail structural evidence checks.

## Limitations

A valid page number is not proof that a claim is factually correct. Gemini results may vary, and scanned PDFs without extractable text are not supported. Uploaded PDFs are sent to the configured Gemini API for analysis. Free Render services may sleep when inactive. The model name must be available to your API project.
