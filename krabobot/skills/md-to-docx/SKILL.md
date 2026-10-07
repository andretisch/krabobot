---
name: md-to-docx
description: Converts Markdown (.md) files to Microsoft Word (.docx) format using Pandoc for high-quality output, including correct table formatting and structure. Use when the user needs to transform documentation or reports from Markdown to a professional Word document.
---

# MD to DOCX Converter

This skill provides a reliable way to convert Markdown files to DOCX format using the Pandoc engine.

## Usage

To convert a file, use the `scripts/convert.py` script:

1. Identify the path to the source `.md` file.
2. Define the desired output path for the `.docx` file.
3. Execute the conversion script.

## Technical Details

- **Engine**: Pandoc (via `pypandoc` wrapper).
- **Quality**: High. Pandoc is the industry standard for document conversion and handles complex Markdown elements (like tables and nested lists) much better than basic Python libraries.
- **Dependencies**: Requires `pandoc` to be installed on the system. The script attempts to ensure installation via `pypandoc.ensure_pandoc_installed()`.
