#!/usr/bin/env python3
"""
Markdown Hierarchical Chunking Script for RAG Pipelines

This script parses Markdown files by their header structure (# through ######),
groups text paragraphs, tables, and code blocks under their corresponding header paths,
and chunks them based on character size limits while preserving tables and code blocks intact.
"""

import argparse
import json
import os
import re


def parse_markdown_to_sections(file_path):
    """
    Parses a markdown file into hierarchical sections.
    Each section contains the text lines and the active header hierarchy at that point.
    """
    sections = []
    current_headers = {f"h{i}": "" for i in range(1, 7)}
    current_content = []

    # Matches markdown headers like: # Header, ## Header, etc.
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

    with open(file_path, "r", encoding="utf-8") as f:
        in_code_block = False
        for line in f:
            stripped = line.strip()

            # Identify code blocks to avoid parsing header symbols inside code blocks
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                current_content.append(line)
                continue

            if not in_code_block:
                header_match = header_pattern.match(stripped)
                if header_match:
                    # Flush previous content if it contains any non-empty lines
                    if any(c.strip() for c in current_content):
                        sections.append(
                            {
                                "headers": current_headers.copy(),
                                "content_lines": current_content,
                            }
                        )
                        current_content = []

                    # Update active headers based on the current header level
                    hashes = header_match.group(1)
                    level = len(hashes)
                    title = header_match.group(2).strip()

                    current_headers[f"h{level}"] = title
                    # Reset all lower level headers
                    for l in range(level + 1, 7):
                        current_headers[f"h{l}"] = ""
                    continue

            current_content.append(line)

    # Flush remaining content at the end of the file
    if any(c.strip() for c in current_content):
        sections.append(
            {"headers": current_headers.copy(), "content_lines": current_content}
        )

    return sections


def group_lines_into_blocks(lines):
    """
    Groups lines of a section into coherent blocks like tables, code blocks,
    and paragraphs to prevent splitting key structures across chunk boundaries.
    """
    blocks = []
    current_block = []
    block_type = None  # 'table', 'code', 'text'
    in_code = False

    for line in lines:
        stripped = line.strip()

        # 1. Code Block detection
        if stripped.startswith("```"):
            if not in_code:
                if current_block:
                    blocks.append(
                        {"type": block_type or "text", "text": "".join(current_block)}
                    )
                    current_block = []
                in_code = True
                block_type = "code"
                current_block.append(line)
            else:
                current_block.append(line)
                blocks.append({"type": "code", "text": "".join(current_block)})
                current_block = []
                in_code = False
                block_type = None
            continue

        if in_code:
            current_block.append(line)
            continue

        # 2. Table detection
        is_table_line = stripped.startswith("|")
        if is_table_line:
            if block_type != "table":
                if current_block:
                    blocks.append(
                        {"type": block_type or "text", "text": "".join(current_block)}
                    )
                    current_block = []
                block_type = "table"
            current_block.append(line)
        else:
            if stripped == "":
                # Use empty lines as paragraph boundaries to split cleanly
                if block_type == "table":
                    blocks.append({"type": "table", "text": "".join(current_block)})
                    current_block = []
                    block_type = None
                elif block_type == "text":
                    if current_block:
                        blocks.append({"type": "text", "text": "".join(current_block)})
                        current_block = []
                    block_type = None
            else:
                if block_type == "table":
                    blocks.append({"type": "table", "text": "".join(current_block)})
                    current_block = []
                    block_type = "text"
                elif block_type is None:
                    block_type = "text"
                current_block.append(line)

    if current_block:
        blocks.append({"type": block_type or "text", "text": "".join(current_block)})

    return [b for b in blocks if b["text"].strip()]


def split_large_text(text, max_size, overlap):
    """
    Helper function to split a single text block by lines/characters
    if it exceeds the max_size.
    """
    lines = text.splitlines(keepends=True)
    chunks = []
    current_lines = []
    current_len = 0

    for line in lines:
        line_len = len(line)
        if current_len + line_len > max_size:
            if current_lines:
                chunks.append("".join(current_lines))
                # Gather overlapping lines from current_lines
                overlap_lines = []
                overlap_len = 0
                for l in reversed(current_lines):
                    if overlap_len + len(l) <= overlap:
                        overlap_lines.insert(0, l)
                        overlap_len += len(l)
                    else:
                        break
                current_lines = overlap_lines
                current_len = overlap_len

            # If a single line is still larger than max_size, split by characters
            if line_len > max_size:
                for i in range(0, line_len, max_size - overlap):
                    chunks.append(line[i : i + max_size])
                continue

        current_lines.append(line)
        current_len += line_len

    if current_lines:
        chunks.append("".join(current_lines))

    return chunks


def create_chunks_from_blocks(blocks, max_chunk_size=800, overlap=100):
    """
    Consolidates atomic blocks into chunks while staying under max_chunk_size
    and applying the overlap.
    """
    chunks = []
    current_chunk_blocks = []
    current_len = 0

    for block in blocks:
        block_text = block["text"]
        block_len = len(block_text)

        if block_len > max_chunk_size:
            # Flush existing chunk
            if current_chunk_blocks:
                chunks.append("".join(b["text"] for b in current_chunk_blocks))
                current_chunk_blocks = []
                current_len = 0

            if block["type"] == "text":
                # Split large text block
                sub_chunks = split_large_text(block_text, max_chunk_size, overlap)
                chunks.extend(sub_chunks)
            else:
                # Keep table/code blocks intact even if they exceed max_chunk_size
                chunks.append(block_text)
        else:
            if current_len + block_len > max_chunk_size:
                chunks.append("".join(b["text"] for b in current_chunk_blocks))

                # Retrieve overlapping block candidates
                overlap_blocks = []
                overlap_len = 0
                for b in reversed(current_chunk_blocks):
                    if overlap_len + len(b["text"]) <= overlap:
                        overlap_blocks.insert(0, b)
                        overlap_len += len(b["text"])
                    else:
                        break
                current_chunk_blocks = overlap_blocks
                current_len = overlap_len

            current_chunk_blocks.append(block)
            current_len += block_len

    if current_chunk_blocks:
        chunks.append("".join(b["text"] for b in current_chunk_blocks))

    return chunks


def format_header_path(headers):
    """
    Returns a unified header path string (e.g., 'H1 > H2 > H3')
    """
    parts = [headers[f"h{i}"] for i in range(1, 7) if headers.get(f"h{i}")]
    return " > ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Chunk markdown files for RAG pipelines."
    )
    parser.add_argument("--input", required=True, help="Path to input markdown file")
    parser.add_argument(
        "--output", required=True, help="Path to output JSON or JSONL file"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=800,
        help="Maximum character size per chunk content",
    )
    parser.add_argument(
        "--overlap", type=int, default=100, help="Character overlap between chunks"
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="json",
        help="Output format (json or jsonl)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    print(f"Parsing '{args.input}'...")
    sections = parse_markdown_to_sections(args.input)

    chunks_data = []
    chunk_id = 1

    for section in sections:
        headers = section["headers"]
        content_lines = section["content_lines"]

        # 1. Group lines into semantic blocks (paragraphs, tables, code blocks)
        blocks = group_lines_into_blocks(content_lines)

        # 2. Chunk blocks based on size constraints
        content_chunks = create_chunks_from_blocks(
            blocks, max_chunk_size=args.max_size, overlap=args.overlap
        )

        # 3. Create metadata and formats for each chunk
        header_path = format_header_path(headers)
        for content in content_chunks:
            # Clean content string
            cleaned_content = content.strip()
            if not cleaned_content:
                continue

            # Create full context text (crucial for dense embedding matching)
            full_text = (
                f"Header: {header_path}\n\n{cleaned_content}"
                if header_path
                else cleaned_content
            )

            chunks_data.append(
                {
                    "id": chunk_id,
                    "headers": {k: v for k, v in headers.items() if v},
                    "header_path": header_path,
                    "content": cleaned_content,
                    "full_text": full_text,
                    "char_count": len(cleaned_content),
                }
            )
            chunk_id += 1

    # Save output
    print(f"Saving {len(chunks_data)} chunks to '{args.output}'...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if args.format == "json":
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    else:  # jsonl
        with open(args.output, "w", encoding="utf-8") as f:
            for item in chunks_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Print stats
    if chunks_data:
        char_counts = [item["char_count"] for item in chunks_data]
        avg_char = sum(char_counts) / len(char_counts)
        print("\n--- Chunking Statistics ---")
        print(f"Total Chunks: {len(chunks_data)}")
        print(f"Max Chunk Size (chars): {max(char_counts)}")
        print(f"Min Chunk Size (chars): {min(char_counts)}")
        print(f"Avg Chunk Size (chars): {avg_char:.1f}")
        print(f"Output File: {args.output}")
        print("---------------------------")
    else:
        print("No chunks were generated.")


if __name__ == "__main__":
    main()
