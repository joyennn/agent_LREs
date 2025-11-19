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
    return """
            You are a smart and context-aware conversational agent.

            Below is a dialogue between one or more students (S1, S2, ...), total students (S) and a teacher (T).
            Utterances with the same index represent one or more Language-Related Episodes (LREs).
            Note that identical utterances may appear under different indices.
            Treat duplicate utterances as a single utterance for the purpose of this task.

            Without any additional background, infer the topic and context of the conversation solely based on the exchange.
            Then, continue the conversation in a natural and contextually appropriate way.

            Generate the same number of dialogue turns as the input.
            Maintain the same alternating speaker pattern and conversational style.

            Continue the dialogue:
            """.strip()

def prompt2():
    return """
            You are a smart and context-aware conversational agent.

            Below is a dialogue between one or more students (S1, S2, ...), total students (S) and a teacher (T).

            Utterances with the same index represent one or more Language-Related Episodes (LREs).
            Note that identical utterances may appear under different indices.
            Treat duplicate utterances as a single utterance for the purpose of this task.

            The class is being conducted in English (L2), but all participants are Korean (L1) speakers.
            The students are at an intermediate English proficiency level (e.g., TEPS 550 or TOEFL 70), and they are also beginners in the subject matter being taught.

            Without any external background, infer the topic and situation of the lesson
            based solely on the exchange provided. Then, naturally continue the dialogue
            in a way that aligns with the context, the students' proficiency, and the academic level.

            Maintain the following constraints:
            - Generate the same number of dialogue turns as the input.
            - Follow the same alternating speaker pattern and conversational style.
            - Ensure that each utterance is similar in length and complexity to the original turns.

            Continue the dialogue:
            """.strip()

def prompt3():
    return """
            You are a smart and context-aware conversational agent.

            Below is a dialogue between one or more students (S1, S2, ...), total students (S) and a teacher (T).

            Utterances with the same index represent one or more Language-Related Episodes (LREs).
            Note that identical utterances may appear under different indices.
            Treat duplicate utterances as a single utterance for the purpose of this task.

            The class is being conducted in English (L2), but all participants are Korean (L1) speakers.
            The students are at an intermediate English proficiency level (e.g., TEPS 550 or TOEFL 70), so their utterances may contain grammatical or lexical errors.
            They are also beginners in the subject matter being taught.

            The teacher provides support and scaffolding based on the following instructional strategies:

            (1) Teacher corrective feedback (CF) strategy types
            1. Identify students’ attitudes toward CF and set shared goals appropriate to the context.
            2. Provide CF confidently, as it supports both accuracy and fluency development.
            3. Use focused CF targeting specific language forms relevant to the lesson.
            4. Make it clear when CF is being given, especially in spoken interactions.
            5. Adapt CF to the learner by starting implicitly and shifting to more explicit forms if needed.
            6. Vary the timing of oral CF (immediate or delayed); written CF is typically delayed.
            7. Allow space for learner uptake without forcing immediate correction.
            8. Tailor CF methods to each learner’s cognitive and emotional needs.
            9. Be willing to correct the same error multiple times for self-regulation.
            10. Monitor learners’ anxiety and adjust CF to ensure it remains supportive.

            (2) Students uptake types
            1. Uptake is initiated by the student.
            2. It is optional—students are not required to respond after receiving CF.
            3. It typically follows a moment when the learner reveals a knowledge gap (e.g., through an error or question).
            4. It responds to a teacher move that provides linguistic information, either explicitly or implicitly.

            Without any external background, infer the topic and situation of the lesson based solely on the exchange provided.
            Then, naturally continue the dialogue in a way that aligns with the context, the students' proficiency, academic level and instructional strategies.

            Maintain the following constraints:
            - Generate the same number of dialogue turns as the input.
            - Follow the same alternating speaker pattern and conversational style.
            - Ensure that each utterance is similar in length and complexity to the original turns.

            Continue the dialogue:
            """.strip()

def prompt4_econ():
    return """
    You are a smart and context-aware conversational agent.

    Below is a dialogue between one or more students (S1, S2, ...), total students (S) and a teacher (T).
    Utterances with the same index represent one or more Language-Related Episodes (LREs).
    Note that identical utterances may appear under different indices.
    Treat duplicate utterances as a single utterance for the purpose of this task.

    Students are as follows:
    Korean L1, English L2
    TOEFL iBT 60+ minimum requirement
    No overseas experience in English-speaking countries
    Admission based on: English interview + middle school English grades

    The teacher is like:
    Korean L1, English L2
    Education: Economics + Education double major (Bachelor's)
    Teaching experience: 5 years public high school → 3 years at this school
    TOEFL: above-average scores (all sections)
    EMI experience: Student (undergraduate) → Teacher (at this school)
    No overseas study experience
    No prior English-medium economics teaching experience
    Self-developed economics English textbook
    Teaching load: 4 classes (2 first-year, 2 second-year), twice weekly
    Professional development: school-provided teacher program


    The teacher provides support and scaffolding based on the following instructional strategies:

    (1) Teacher corrective feedback (CF) strategy types
    1. Identify students’ attitudes toward CF and set shared goals appropriate to the context.
    2. Provide CF confidently, as it supports both accuracy and fluency development.
    3. Use focused CF targeting specific language forms relevant to the lesson.
    4. Make it clear when CF is being given, especially in spoken interactions.
    5. Adapt CF to the learner by starting implicitly and shifting to more explicit forms if needed.
    6. Vary the timing of oral CF (immediate or delayed); written CF is typically delayed.
    7. Allow space for learner uptake without forcing immediate correction.
    8. Tailor CF methods to each learner’s cognitive and emotional needs.
    9. Be willing to correct the same error multiple times for self-regulation.
    10. Monitor learners’ anxiety and adjust CF to ensure it remains supportive.

    (2) Students uptake types
    1. Uptake is initiated by the student.
    2. It is optional—students are not required to respond after receiving CF.
    3. It typically follows a moment when the learner reveals a knowledge gap (e.g., through an error or question).
    4. It responds to a teacher move that provides linguistic information, either explicitly or implicitly.


    Without any external background, infer the topic and situation of the lesson based solely on the exchange provided.
    Then, naturally continue the dialogue in a way that aligns with the context, the profiles of the students and the teacher and instructional strategies.


    Maintain the following constraints:
    - Generate the same number of dialogue turns as the input.
    - Follow the same alternating speaker pattern and conversational style.
    - Ensure that each utterance is similar in length and complexity to the original turns.


    Continue the dialogue:
    """.strip()

def prompt4_math():
    return """
      You are a smart and context-aware conversational agent.

      Below is a dialogue between one or more students (S1, S2, ...), total students (S) and a teacher (T).
      Utterances with the same index represent one or more Language-Related Episodes (LREs).
      Note that identical utterances may appear under different indices.
      Treat duplicate utterances as a single utterance for the purpose of this task.

      Students are as follows:
      Korean L1, English L2
      TOEFL iBT 60+ minimum requirement
      No overseas experience in English-speaking countries
      Admission based on: English interview + middle school English grades

      The teacher is like:
      English L1 (Ireland)
      Education: Biology + Physiology double major (Bachelor's, UK) + Science & Maths Education (Postgraduate Diploma, UK)
      Teaching experience: 3 years secondary school (Ireland) → 2 years ESL elementary (Seoul) → private English academy → this school (since 2017)
      Subjects taught: Mathematics, Biology (EMI)
      Teaching load: Second-year mathematics, twice weekly
      Background: Science/maths teacher → ESL teacher → EMI teacher

      The teacher provides support and scaffolding based on the following instructional strategies:

      (1) Teacher corrective feedback (CF) strategy types
      1. Identify students’ attitudes toward CF and set shared goals appropriate to the context.
      2. Provide CF confidently, as it supports both accuracy and fluency development.
      3. Use focused CF targeting specific language forms relevant to the lesson.
      4. Make it clear when CF is being given, especially in spoken interactions.
      5. Adapt CF to the learner by starting implicitly and shifting to more explicit forms if needed.
      6. Vary the timing of oral CF (immediate or delayed); written CF is typically delayed.
      7. Allow space for learner uptake without forcing immediate correction.
      8. Tailor CF methods to each learner’s cognitive and emotional needs.
      9. Be willing to correct the same error multiple times for self-regulation.
      10. Monitor learners’ anxiety and adjust CF to ensure it remains supportive.

      (2) Students uptake types
      1. Uptake is initiated by the student.
      2. It is optional—students are not required to respond after receiving CF.
      3. It typically follows a moment when the learner reveals a knowledge gap (e.g., through an error or question).
      4. It responds to a teacher move that provides linguistic information, either explicitly or implicitly.

      Without any external background, infer the topic and situation of the lesson based solely on the exchange provided.
      Then, naturally continue the dialogue in a way that aligns with the context, the profiles of the students and the teacher and instructional strategies.

      Maintain the following constraints:
      - Generate the same number of dialogue turns as the input.
      - Follow the same alternating speaker pattern and conversational style.
      - Ensure that each utterance is similar in length and complexity to the original turns.

      Continue the dialogue:
    """.strip()    
    
def prompt5_econ():
    return """
      You are a smart and context-aware conversational agent.

      Below is a dialogue between one or more students (S1, S2, ...), total students (S) and a teacher (T).
      Utterances with the same index represent one or more Language-Related Episodes (LREs).
      Note that identical utterances may appear under different indices.
      Treat duplicate utterances as a single utterance for the purpose of this task.


      Students are as follows:
      Korean L1, English L2
      TOEFL iBT 60+ minimum requirement
      No overseas experience in English-speaking countries
      Admission based on: English interview + middle school English grades


      The teacher is like:
      Korean L1, English L2
      Education: Economics + Education double major (Bachelor's)
      Teaching experience: 5 years public high school → 3 years at this school
      TOEFL: above-average scores (all sections)
      EMI experience: Student (undergraduate) → Teacher (at this school)
      No overseas study experience
      No prior English-medium economics teaching experience
      Self-developed economics English textbook
      Teaching load: 4 classes (2 first-year, 2 second-year), twice weekly
      Professional development: school-provided teacher program


      Without any external background, infer the topic and situation of the lesson based solely on the exchange provided.
      Then, naturally continue the dialogue in a way that aligns with the context, the profiles of the students and the teacher.


      Maintain the following constraints:
      - Generate the same number of dialogue turns as the input.
      - Follow the same alternating speaker pattern and conversational style.
      - Ensure that each utterance is similar in length and complexity to the original turns.


      Continue the dialogue:
    """.strip()

def prompt5_math():
    return """
      You are a smart and context-aware conversational agent.

      Below is a dialogue between one or more students (S1, S2, ...), total students (S) and a teacher (T).
      Utterances with the same index represent one or more Language-Related Episodes (LREs).
      Note that identical utterances may appear under different indices.
      Treat duplicate utterances as a single utterance for the purpose of this task.


      Students are as follows:
      Korean L1, English L2
      TOEFL iBT 60+ minimum requirement
      No overseas experience in English-speaking countries
      Admission based on: English interview + middle school English grades


      The teacher is like:
      English L1 (Ireland)
      Education: Biology + Physiology double major (Bachelor's, UK) + Science & Maths Education (Postgraduate Diploma, UK)
      Teaching experience: 3 years secondary school (Ireland) → 2 years ESL elementary (Seoul) → private English academy → this school (since 2017)
      Subjects taught: Mathematics, Biology (EMI)
      Teaching load: Second-year mathematics, twice weekly
      Background: Science/maths teacher → ESL teacher → EMI teacher


      Without any external background, infer the topic and situation of the lesson based solely on the exchange provided.
      Then, naturally continue the dialogue in a way that aligns with the context, the profiles of the students and the teacher.


      Maintain the following constraints:
      - Generate the same number of dialogue turns as the input.
      - Follow the same alternating speaker pattern and conversational style.
      - Ensure that each utterance is similar in length and complexity to the original turns.


      Continue the dialogue:
    """.strip()

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

### phase1 ###
prompts_econ = [prompt1, prompt2, prompt3, prompt4_econ, prompt5_econ]
prompts_math = [prompt1, prompt2, prompt3, prompt4_math, prompt5_math]

for p in prompts:
    agent_dialogue(
        json_filename="high_econ.json",
        num_cycles=20,
        prompt=p,
        models=["gpt-4o"])
    
for p in prompts:
    agent_dialogue(
        json_filename="high_math.json",
        num_cycles=20,
        prompt=p,
        models=["gpt-4o"])


### phase2 ###
models = ["gpt-4o", "gemini", "claude", "llama3"]

agent_dialogue(
    json_filename="high_econ.json",
    num_cycles=20,
    prompt=prompt3,
    models=models)

agent_dialogue(
    json_filename="high_math.json",
    num_cycles=20,
    prompt=prompt3,
    models=models)
