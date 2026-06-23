#!/usr/bin/env python3
import csv
import os
import re


def parse_qa_markdown(filepath):
    qa_list = []

    current_category = ""
    current_id = ""
    current_type = ""
    current_question = []
    current_answer = []

    state = None  # Track state: 'Q' or 'A'

    category_pattern = re.compile(r"^##\s+(카테고리\s+[A-D]\s+—\s+[^—\n]+)")
    item_pattern = re.compile(r"^###\s+([A-D]\d+)\s+\[([^\]]+)\]")

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            # Skip horizontal rules and reset accumulator state
            if stripped == "---":
                state = None
                continue

            # Check for category header
            cat_match = category_pattern.match(stripped)
            if cat_match:
                # Flush previous item if exists before changing category
                if current_id:
                    qa_list.append(
                        {
                            "Category": current_category,
                            "ID": current_id,
                            "Type": current_type,
                            "Question": "\n".join(current_question).strip(),
                            "Answer": "\n".join(current_answer).strip(),
                        }
                    )
                    current_id = ""
                current_category = cat_match.group(1).strip()
                continue

            # Check for Q/A item header
            item_match = item_pattern.match(stripped)
            if item_match:
                # Flush previous item if exists
                if current_id:
                    qa_list.append(
                        {
                            "Category": current_category,
                            "ID": current_id,
                            "Type": current_type,
                            "Question": "\n".join(current_question).strip(),
                            "Answer": "\n".join(current_answer).strip(),
                        }
                    )
                current_id = item_match.group(1).strip()
                current_type = item_match.group(2).strip()
                current_question = []
                current_answer = []
                state = None
                continue

            if stripped.startswith("**Q:**"):
                state = "Q"
                # Extract text after **Q:**
                parts = line.split("**Q:**", 1)
                q_part = parts[1].rstrip() if len(parts) > 1 else ""
                current_question.append(q_part)
                continue
            elif stripped.startswith("**A:**"):
                state = "A"
                # Extract text after **A:**
                parts = line.split("**A:**", 1)
                a_part = parts[1].rstrip() if len(parts) > 1 else ""
                current_answer.append(a_part)
                continue

            # Accumulate multi-line questions or answers
            if state == "Q":
                current_question.append(line.rstrip())
            elif state == "A":
                current_answer.append(line.rstrip())

        # Flush last item
        if current_id:
            qa_list.append(
                {
                    "Category": current_category,
                    "ID": current_id,
                    "Type": current_type,
                    "Question": "\n".join(current_question).strip(),
                    "Answer": "\n".join(current_answer).strip(),
                }
            )

    return qa_list


def save_to_csv(qa_list, output_filepath):
    fieldnames = ["Category", "ID", "Type", "Question", "Answer"]
    with open(output_filepath, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in qa_list:
            writer.writerow(row)


def main():
    input_file = "kakao-tech-bootcamp-rag-qa-set.md"
    output_file = "kakao-tech-bootcamp-rag-qa-set.csv"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Parsing {input_file}...")
    qa_list = parse_qa_markdown(input_file)
    print(f"Parsed {len(qa_list)} QA pairs.")

    print(f"Saving to {output_file}...")
    save_to_csv(qa_list, output_file)
    print("Done!")


if __name__ == "__main__":
    main()
