from flask import Flask, request, jsonify
import os

app = Flask(__name__)

from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import StringPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
# from langchain.chains import ConversationChain
# from langchain.chains.conversation.memory import ConversationBufferMemory

from transformers import pipeline


api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.8,)

PROMPT_TEMPLATE = """
context: {context}
history: {history}
input: {input}

Generate an answer to an input that takes into account context and history.
answer:
"""

# Define the custom prompt template class
class CustomPromptTemplate(StringPromptTemplate):
    context: str
    template: str

    def format(self, **kwargs) -> str:
        kwargs['context'] = self.context
        return self.template.format(**kwargs)

# Create an instance of the custom prompt template
PROMPT = CustomPromptTemplate(
    input_variables=["history", "input"],
    template=PROMPT_TEMPLATE,
    context="""
    너는 중고등학생들과 대화를 나누는 비밀 친구야.
    너의 이름은 Feelow야.
    답변을 할 때, 이모지를 많이 사용해줘.
    답변 마지막에는 항상 학생들에게 질문을 해줘.
    항상 한국어로 답변해줘.
    항상 반말을 사용해줘.
    항상 두 문장 이하로 답변해주고, 질문은 오직 하나만 해줘.
    """
    # context="""
    # You're a secret friend who talks with middle and high school students.
    # Your name is Feelow.
    # When responding, use lots of appropriate emojis.
    # At the end of your answer, always ask students.
    # Please always answer in Korean.
    # Always use semi-finished sentences like "어", "맞아", "좋아", "해", "이야", "봐".
    # Always answer in two sentences or less, and only ask one question.
    # """
)

# Define the conversation memory
# memory = ConversationBufferMemory(
#     ai_prefix="Secret Friend",
#     human_prefix='Student'
# )

# conversation = ConversationChain(
#     prompt = PROMPT,
#     llm=llm,
#     memory=memory,
# )

# global_history = ""

user_histories = {}

# user_history 요약
text_splitter = CharacterTextSplitter()

summarize_prompt = """다음 텍스트를 간결하게 한국말로 요약해줘:
{text}
SUMMARY:"""
SUMMARIZE_PROMPT = PromptTemplate(template=summarize_prompt, input_variables=["text"])
summarize_chain = LLMChain(llm=llm, prompt=SUMMARIZE_PROMPT)
stuff_chain = StuffDocumentsChain(llm_chain=summarize_chain, document_variable_name="text")

# 감정 분석
#감정 분석 모델 불러오기

huggingface_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

senti = pipeline(
    "text-classification",
    model="matthewburke/korean_sentiment",
)

def senti_score_json(input):
  score = senti(input, top_k=None)

  for item in score:
    if item['label'] == 'LABEL_0':
        item['label'] = 'negative'
    elif item['label'] == 'LABEL_1':
        item['label'] = 'positive'

  # senti_score_json = json.dumps(score, ensure_ascii=False, indent=2)
  print(senti_score_json)

  return score


# Define the Flask route
@app.route("/api/chat", methods=['POST'])
def chat_endpoint():
    try:
        global user_histories

        input_text = request.json['input']
        user_name = request.json['user_name']

        if user_name not in user_histories:
            user_histories[user_name] = ""
        user_history = user_histories[user_name]

        # Format the prompt using the conversation template
        formatted_prompt = PROMPT.format(history=user_history, input=input_text)

        # Use the language model to predict the response
        response = llm.predict(text=formatted_prompt)
        user_history += f"\n{user_name}: {input_text}\nFeelow: {response}"
        user_histories[user_name] = user_history
        # global_history += f"\nStudent: {input_text}\nSecret Friend: {response}"

        # user history summarization
        texts = text_splitter.split_text(user_history)
        docs = [Document(page_content=t) for t in texts]
        history_sum = stuff_chain.run(docs)

        # sentiment analysis
        senti_score = senti_score_json(input_text)

        return jsonify({'input': input_text, 'response': response, 'history': user_history, 'history_sum': history_sum, 'senti_score': senti_score})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define a simple hello route for testing
@app.route("/")
def hello():
    return "Hello, World!"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
