import re
import sys

def clean_text(text):
    # 한글, 영문, 숫자, 문장부호 및 마크다운 기호를 제외한 나머지 특수문자 제거
    text = re.sub(r'[^\w\s\.\?\!\,\(\)\[\]\"\'\-\#\*\`\>\:\=\+\|]', '', text)
    
    # 문장 중간에서 단어를 끊어놓은 불필요한 줄바꿈 보정
    text = re.sub(r'([가-힣])\n([가-힣])', r'\1 \2', text)
    
    # 두 번 이상 중복된 일반 공백을 하나로 축소
    text = re.sub(r' +', ' ', text)
    
    # 세 번 이상 연속된 줄바꿈을 하나의 줄바꿈으로 변경
    text = re.sub(r'\n{3,}', '\n', text)
    
    return text.strip()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ocr_noise_cleaner.py <input_path> <output_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        with open(input_path, 'r', encoding='utf-8') as input_file:
            original_text = input_file.read()

        result = clean_text(original_text)

        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(result)

        print("Success: Processed the file.")

    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)