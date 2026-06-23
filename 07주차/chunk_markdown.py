#!/usr/bin/env python3
import argparse
import json
import os
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def main():
    parser = argparse.ArgumentParser(description="Chunk markdown files for RAG pipelines using LangChain Splitters.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--format", choices=["json", "jsonl"], default="json")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # 1. 헤더 기반 마크다운 1차 분할
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
        ("#####", "h5"),
        ("######", "h6"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    header_splits = markdown_splitter.split_text(markdown_text)

    # 2. 800자 초과 청크 대상 2차 재귀 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.max_size,
        chunk_overlap=args.overlap
    )
    splits = text_splitter.split_documents(header_splits)

    # 3. 기존 JSON/JSONL 스키마 양식 포맷팅
    chunks_data = []
    chunk_id = 1

    for split in splits:
        headers = split.metadata
        header_parts = [headers.get(f"h{i}") for i in range(1, 7) if headers.get(f"h{i}")]
        header_path = " > ".join(header_parts)
        content = split.page_content.strip()
        full_text = f"Header: {header_path}\n\n{content}" if header_path else content

        chunks_data.append({
            "id": chunk_id,
            "headers": headers,
            "header_path": header_path,
            "content": content,
            "full_text": full_text,
            "char_count": len(content)
        })
        chunk_id += 1

    print(f"Saving {len(chunks_data)} chunks to '{args.output}'...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if args.format == "json":
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            for item in chunks_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if chunks_data:
        char_counts = [item["char_count"] for item in chunks_data]
        print(f"Total Chunks: {len(chunks_data)}")
        print(f"Avg Chunk Size: {sum(char_counts) / len(char_counts):.1f} chars")

if __name__ == "__main__":
    main()
