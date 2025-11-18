import json
import csv
from openai import OpenAI

# API key
client = OpenAI(api_key="your-api-key")  

# Prompting
prompt1 = """
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

prompt2 = """
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

prompt3 = """
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

prompt4_econ = """
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


prompt4_math = """
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


prompt5_econ = """
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


prompt5_math = """
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


# Getting prompt
def get_prompt(prompt_version: str, dialogue: list) -> str:
    prompt_map = {
        "1": prompt1,
        "2": prompt2,
        "3": prompt3,
        "4_1": prompt4_econ,
        "4_2": prompt4_math,
        "5_1": prompt5_econ,
        "5_2": prompt5_math
    }

    if prompt_version not in prompt_map:
        raise ValueError(f"Unsupported prompt version: {prompt_version}")

    dialogue_text = ""
    for turn in dialogue:
        dialogue_text += f"{turn['speaker']}: {turn['utterance']}\n"

    return prompt_map[prompt_version] + "\n\n" + dialogue_text.strip()


# Generating agent discourse
def agent_dialogue(json_filename: str, num_cycles: int, prompt_version: str, output_filename: str = None):
    with open(json_filename, "r", encoding="utf-8") as f:
        dialogue = json.load(f)

    original_dialogue = dialogue.copy()
    prompt = get_prompt(prompt_version, original_dialogue)

    agent_dialogue = []
    for cycle in range(num_cycles):
        print(f"\nGenerating cycle {cycle+1}...")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )

        reply = response.choices[0].message.content.strip()
        lines = reply.split('\n')

        for line in lines:
            if ":" in line:
                speaker, utterance = line.split(":", 1)
                agent_dialogue.append({
                    "index": "",
                    "speaker": speaker.strip(),
                    "utterance": utterance.strip(),
                    "LRE": "",
                    "source": "agent"
                })

    final_output = [
        {
            "index": turn.get("index", ""),
            "speaker": turn["speaker"],
            "utterance": turn["utterance"],
            "LRE": turn.get("LRE", ""),
            "source": "original"
        }
        for turn in original_dialogue
    ] + agent_dialogue

    if output_filename is None:
        base_name = json_filename.replace(".json", "")
        output_filename = f"agent_{prompt_version}_{base_name}.csv"

    with open(output_filename, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "speaker", "utterance", "LRE", "source"])
        writer.writeheader()
        writer.writerows(final_output)

    print(f"\n✅ saved: {output_filename}")


# Processing
agent_dialogue("high_econ.json", num_cycles=20, prompt_version="1")
agent_dialogue("high_math.json", num_cycles=20, prompt_version="1")
agent_dialogue("high_econ.json", num_cycles=20, prompt_version="2")
agent_dialogue("high_math.json", num_cycles=20, prompt_version="2")
agent_dialogue("high_econ.json", num_cycles=20, prompt_version="3")
agent_dialogue("high_math.json", num_cycles=20, prompt_version="3")
agent_dialogue("high_econ.json", num_cycles=20, prompt_version="4_1")
agent_dialogue("high_math.json", num_cycles=20, prompt_version="4_2")
agent_dialogue("high_econ.json", num_cycles=20, prompt_version="5_1")
agent_dialogue("high_math.json", num_cycles=20, prompt_version="5_2")
