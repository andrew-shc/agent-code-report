# Formatting Guidelines

This repo uses ACM manuscript formatting for reports and papers.
The goal is consistency, not perfection — these guidelines will
grow as conventions settle.

## Reports (LaTeX / PDF)

- Use the `acmart` document class with `[manuscript,review]` options
  for internal review drafts.
- Keep reports under `GREENFIELD/` (source `.tex`) with compiled
  PDFs in `ASSETS/`.
- Current report: `GREENFIELD/restir/eval_report.tex` →
  `ASSETS/restir/eval_report/STANFORD_ORB_EVAL_REPORT.pdf`

## Code Listings

- Use the `listings` package with Python language settings.
- Keep listings focused: one class or one function per listing,
  not entire files.
- Use the `float` option so the listing can drift to a convenient
  page in the PDF.

## Markdown (non-LaTeX)

- No strict rules yet — use standard GitHub-flavored Markdown.
- Tables should be pipe-delimited with alignment markers.
- Code blocks should specify the language for syntax highlighting.
- Keep line length under 100 characters for readability.

## WIP

All of this is provisional. When a new convention emerges, add it
here rather than leaving it implicit.
