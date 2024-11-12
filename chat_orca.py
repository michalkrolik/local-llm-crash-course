import argparse
from ctransformers import AutoModelForCausalLM

# parser = argparse.ArgumentParser(description="Przykład skryptu przyjmującego argumenty")
# parser.add_argument("prompt", type=str)
# question = parser.parse_args().prompt

llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")


def get_prompt(instruction: str, history: list[str] = None) -> str:
    system = "You are an AI assistant that gives helpful answers. You answers are consistent, and as short as possible"
    prompt = f"### System:\n{system}\n\n### User:\n"
    if history is not None:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    return prompt


history = []
answer = ""

question = "Who is the president of Poland?"
for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)
    answer += word
print()

history.append(answer)

question = "who is of Russia?"
for word in llm(get_prompt(question, history), stream=True):
    print(word, end="", flush=True)
print()
