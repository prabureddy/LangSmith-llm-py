from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from langsmith import Client
import uuid

run_id = str(uuid.uuid4())

ls_client = Client()

openai_client = wrap_openai(OpenAI())

@traceable(run_type="retriever")
def retriever(query: str):
    results = ["Harrison worked at Kensho"]
    return results

@traceable(metadata={"llm": "gpt-4o-mini"})
def rag(question):
    docs = retriever(question)
    system_message = """Answer the users question using only the provided information below:
        {docs}""".format(docs="\n".join(docs))

    ls_client.create_feedback(
        run_id,
        key="user-score",
        score=0.8,
        comment="Great answer!",
    )

    return openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
        model="gpt-4o-mini",
    )

rag(
    "where did harrison work",
    langsmith_extra={"run_id": run_id, "metadata": {"user_id": "harrison"}}
)

