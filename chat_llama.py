import argparse
from ctransformers import AutoModelForCausalLM

parser = argparse.ArgumentParser(description="Przykład skryptu przyjmującego argumenty")
parser.add_argument("prompt", type=str)
question = parser.parse_args().prompt

# llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q4_K_M.gguf")


def get_prompt(instruction: str) -> str:
    system = "You are an AI assistant that gives helpful answers. You answers are consistent, and as short as possible"
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
    return prompt


for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)
print()
