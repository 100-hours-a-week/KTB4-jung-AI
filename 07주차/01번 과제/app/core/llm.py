from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings


class LLMManager:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", google_api_key=settings.GEMINI_API_KEY
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 카카오테크 부트캠프 4기 교육생(크루)들을 돕는 전문 운영 지원 AI 어시스턴트입니다.
제공된 [교육 규칙 및 규정 문서]의 내용을 절대적인 기준(Context)으로 삼아, 교육생의 질문에 친절하고 정확하게 답변해 주세요.

[작성 지침]
1. 반드시 제공된 [규칙 규정 (Context)]에 기반하여 사실에 입각한 답변을 작성해 주세요. Context에 없는 자의적인 내용이나 거짓 정보를 지어내지 마세요.
2. 만약 제공된 Context에서 질문에 대한 답을 찾을 수 없거나 부족하다면, 억지로 꾸며내지 말고 "제공된 규정 문서에서 관련 내용을 찾을 수 없습니다. 운영팀(헬퍼라이언)에 문의하시기 바랍니다."라고 답변하세요.
3. 질문에 명확한 수치나 기한(예: 3일 전까지, 80% 이상, 3회 누적 등)이 있을 경우 이를 명확하게 표기해 주세요.
4. 존댓말과 격식 있고 친절한 톤앤매너를 유지해 주세요.""",
                ),
                (
                    "human",
                    """[규칙 규정 (Context)]
{context}

[교육생 질문]
{question}

[답변]""",
                ),
            ]
        )

        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def get_chain(self):
        return self.chain
