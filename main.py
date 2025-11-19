import os
import json
import csv
from typing import List
import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic
from huggingface_hub import InferenceClient


# ================================
# 1. API key / Client setting
# ================================

# ⚠️ only in Colab:
%env OPENAI_API_KEY= ... 
%env GOOGLE_API_KEY= ...
%env ANTHROPIC_API_KEY= ...

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
HUGGINGFACE_API_KEY = "HUGGINGFACE_API_KEY"

client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)
llama_client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=HUGGINGFACE_API_KEY)


# ================================
# 2. Prompt
# ================================

def prompt1():
    return "You are prompt 1 ..."

def prompt2():
    return "You are prompt 2 ..."

def prompt3():
    return """
    You are a smart and context-aware conversational agent.
    ... (여기에 아까 긴 prompt3 내용) ...
    """.strip()

def prompt4():
    return "You are prompt 4 ..."

def prompt5():
    return "You are prompt 5 ..."


# ================================
# 3. Original data (json)-> prompt
# ================================

def build_prompt(base_prompt: str, dialogue: list) -> str:
    lines = []
    for turn in dialogue:
        speaker = turn["speaker"]
        utt = turn["utterance"]
        lines.append(f"{speaker}: {utt}")
    dialogue_text = "\n".join(lines)
    return base_prompt + "\n\n" + "Dialogue:\n" + dialogue_text + "\n\nContinue the dialogue:"


# ================================
# 4. Calling models
# ================================

def call_gpt4o(prompt: str) -> str:
    resp = client_openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7,
    )
    return resp.choices[0].message.content


def call_gemini(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-pro")
    resp = model.generate_content(prompt)
    return resp.text


def call_claude(prompt: str) -> str:
    resp = client_anthropic.messages.create(
        model="claude-3-5-sonnet-20240620", 
        max_tokens=1000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def call_llama(prompt: str) -> str:
    response = llama_client.text_generation(
        prompt,
        max_new_tokens=1000,
        temperature=0.7,
        do_sample=True,          
        repetition_penalty=1.0,  
    )
    return response


def model_call(model_name: str, prompt: str) -> str:
    if model_name == "gpt-4o":
        return call_gpt4o(prompt)
    elif model_name == "gemini":
        return call_gemini(prompt)
    elif model_name == "claude":
        return call_claude(prompt)
    elif model_name == "llama3":
        return call_llama3(prompt)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# ================================
# 5. Result & Save
# ================================

def parse_agent_reply(reply: str) -> list:
    turns = []
    for line in reply.splitlines():
        if ":" in line:
            speaker, utt = line.split(":", 1)
            turns.append({
                "index": "",
                "speaker": speaker.strip(),
                "utterance": utt.strip(),
                "LRE": "",
                "source": "agent",
            })
    return turns


def save_dialogue_csv(original_dialogue: list, agent_dialogue: list, output_filename: str):
    final_output = [
        {
            "index": t.get("index", ""),
            "speaker": t["speaker"],
            "utterance": t["utterance"],
            "LRE": t.get("LRE", ""),
            "source": "original",
        }
        for t in original_dialogue
    ] + agent_dialogue

    with open(output_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["index", "speaker", "utterance", "LRE", "source"]
        )
        writer.writeheader()
        writer.writerows(final_output)


# ================================
# 6. Final
# ================================

def agent_dialogue(json_filename: str, num_cycles: int, prompt, models: list):

    with open(json_filename, "r", encoding="utf-8") as f:
        dialogue = json.load(f)

    original_dialogue = dialogue.copy()
    base_prompt = prompt()                        
    full_prompt = build_prompt(base_prompt, original_dialogue)
    prompt_name = prompt.__name__                  

    for model_name in models:
        print(f"\n🚀 Generating using: {model_name}")
        agent_turns_all_cycles = []

        for cycle in range(num_cycles):
            print(f"🔄 Cycle {cycle + 1} / {num_cycles} ...")
            reply = model_call(model_name, full_prompt)
            agent_turns = parse_agent_reply(reply)
            agent_turns_all_cycles.extend(agent_turns)

        base_name = os.path.splitext(os.path.basename(json_filename))[0]
        output_filename = f"agent_{base_name}_{prompt_name}_{model_name}.csv"

        save_dialogue_csv(original_dialogue, agent_turns_all_cycles, output_filename)
        print(f"✅ Saved as: {output_filename}")


# ================================
# 7. Processing
# ================================

# phase1
prompts = [prompt1, prompt2, prompt3, prompt4, prompt5]

for p in prompts:
    agent_dialogue(
        json_filename="high_math.json",
        num_cycles=20,
        prompt=p,
        models=["gpt-4o"])
    
for p in prompts:
    agent_dialogue(
        json_filename="high_econ.json",
        num_cycles=20,
        prompt=p,
        models=["gpt-4o"])
    
# phase2
models = ["gpt-4o", "gemini", "claude", "llama3"]

agent_dialogue(
    json_filename="high_math.json",
    num_cycles=20,
    prompt=prompt3,
    models=models)

agent_dialogue(
    json_filename="high_econ.json",
    num_cycles=20,
    prompt=prompt3,
    models=models)
