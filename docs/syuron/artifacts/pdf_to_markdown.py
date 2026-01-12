#!/usr/bin/env python3
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from google import genai
from google.genai import types


@dataclass
class Config:
    api_key: str
    model_name: str = "gemini-3-flash-preview"


def new_config() -> Config:
    api_key_val = os.environ.get("GEMINI_API_KEY")
    if not api_key_val:
        err_msg = "GEMINI_API_KEY environment variable is not set"
        raise ValueError(err_msg)
    return Config(api_key=api_key_val)


def convert_page_to_markdown(client: genai.Client, uploaded_file: types.File, page_num: int, model_name: str) -> str:
    """Converts a single page of a PDF to markdown."""
    print(f"Converting page {page_num}...")
    prompt = (
        f"Convert page {page_num} of this PDF document into high-quality Markdown. "
        "Preserve the structure, headers, tables, and mathematical formulas (using LaTeX if possible). "
        "Only output the Markdown content for this specific page. Do not include any preamble or conclusion."
    )

    response = client.models.generate_content(
        model=model_name,
        contents=[uploaded_file, prompt],
        config=types.GenerateContentConfig(system_instruction="You are a professional document converter. Output ONLY valid Markdown."),
    )

    if not response.text:
        return f"\n\n<!-- Failed to convert page {page_num} -->\n\n"

    return response.text


def convert_pdf_to_markdown(pdf_path: Path, config: Config) -> str:
    """Converts a PDF file to markdown page by page using Gemini API."""
    client = genai.Client(api_key=config.api_key)

    # Get page count locally
    pdf = pdfium.PdfDocument(str(pdf_path))
    num_pages = len(pdf)
    pdf.close()
    print(f"Total pages to convert: {num_pages}")

    # Upload the file once
    print(f"Uploading {pdf_path}...")
    uploaded_file = client.files.upload(file=str(pdf_path))

    all_markdown = []
    for i in range(1, num_pages + 1):
        page_md = convert_page_to_markdown(client, uploaded_file, i, config.model_name)
        all_markdown.append(page_md)

    return "\n\n".join(all_markdown)


def main() -> None:
    min_args = 2
    if len(sys.argv) < min_args:
        print("Usage: python pdf_to_markdown.py <path_to_pdf>")
        return

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Error: File {pdf_path} not found")
        return

    try:
        config = new_config()
        markdown_content = convert_pdf_to_markdown(pdf_path, config)

        output_path = pdf_path.with_suffix(".md")
        output_path.write_text(markdown_content, encoding="utf-8")
        print(f"Successfully converted {pdf_path} to {output_path}")
    except ValueError as e:
        print(f"Configuration error: {e}")
    except RuntimeError as e:
        print(f"API or Runtime error: {e}")
    except OSError as e:
        print(f"File system error: {e}")


if __name__ == "__main__":
    main()
