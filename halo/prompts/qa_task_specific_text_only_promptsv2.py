# this file contains bbox as inputs with corresponding language description of objects in the scene. The output is VQA of different types: counting, bounding box, frame index, etc.
eval_multi_bbox_only_prompt = '''
You are evaluating multiple candidate outputs for a single robot episode.
You will be given the SAME EPISODE CONTEXT as in the base prompt:
- A task instruction, task's detailed description, summary of task events labeled by frame numbers, and a list of information provided per-frame and per-camera about the visible objects in the scene, their locations, robot actions, etc.

After the task information, you will also be given a BUNDLE of candidate items to evaluate.
Each candidate item includes:
  - id (MUST be an integer)
  - 'query' (a memory-based question),
  - 'answer' (the model's answer),

Your job (per item):
1) Task alignment: Do query, instruction, and answer all help in solving the task as described in the task instruction and task description?
2) Grounding: Is the answer specific and consistent with the episode's referenced frame(s) indices and camera(s)? Is the answer correct? Verify for both.
3) Memory dependence: Does the query REQUIRE recalling earlier information about the visible objects in the scene, their locations, robot actions, etc.
4) Instruction quality: Does the instruction briefly prime what to remember?

Answer should be correct for high scores 3-5, both inclusive.
Scoring (integers only, 1–5):
5 = Excellent: query and answer clearly help in solving the task, grounded and answers are correct, memory-dependent question
4 = Good: query and answer help in solving the task; grounding and answer are correct; partially memory-dependent question
3 = Fair: query and answer help in solving the task; ground and answer are correct; not memory-dependent question
2 = Weak: the question-answer pair is not useful to remember for the task (e.g., asking about distractor objects, not relevant to the task); answer is correct;
1 = Poor: irrelevant to the task; answer is incorrect;

Decision rule:
- If score ≥ 3 → decision = "keep"
- If score ≤ 2 → decision = "skip"

INPUT FORMAT
You will receive:
1) The task information: a task instruction, task's detailed description, and a list of information provided per-frame and per-camera about the visible objects in the scene, their locations, robot actions, etc.
2) Then a bundle of candidate items:
[
    {{
      "id": "<integer>",
      "query": "<string>",
      "answer": "<string>",
    }},
    ...
]

STRICT OUTPUT FORMAT (no prose, no extra fields in the JSON). Mention one entry per ID for each entry in the candidate items. Do NOT skip any IDs:
json{{
  "results": [
    {{"reasoning": "<explain briefly why it will be useful to remember for the task>", "id": <integer>, "score": <integer>, "decision": "keep"|"skip"}},
    ...
  ]
}}

Important:
- 'id' in output MUST match the integer 'id' from input.
- Base your judgments ONLY on the provided task information (task instruction, task description, and provided information.) and the candidate fields.
- Do NOT include explanations, intermediate notes, or any fields other than id, score, and decision in the JSON.

TASK INSTRUCTION: {task_language}

TASK DESCRIPTION: {task_description}
'''

task_specific_bbox_only_prompt = '''
You are given a task instruction, task's description, summary of task events labeled by frame numbers, and a list of information provided per-frame and per-camera about the visible objects in the scene, their locations, and robot actions.
You are given a separate list of QUERY-FRAME INDEXES to generate the query and answer.

Your job:

1. Identify events relevant to the task.
  Use the task description, and summary of task events to understand the scene in the current episode.
  Using these information, describe what plausibly happens in the task for each of the given QUERY-FRAME INDEXES.

2. Write ONE memory-based query important for the task for each of the given QUERY-FRAME INDEXES
  The question must require recalling earlier information provided in the prompt for each of the given QUERY-FRAME INDEXES. It may include: 
    - object location seen earlier in the QUERY-FRAME INDEX (e.g., bounding box coordinates, etc.)
    - number of objects of a particular type seen earlier in the QUERY-FRAME INDEX (e.g., number of sponges, etc.)
    - few words about the task events achieved upto the QUERY-FRAME INDEX using the provided summary of task events (e.g., picked two sponges, placed one in sink)
    - time-frame of the task events which includes the QUERY-FRAME INDEX (e.g., frame 10 to frame 20)
    - important object relations to remember (e.g., object X is to the left of object Y)
  Rules (VERY IMPORTANT):
    - The answer should be relevant to the provided task instruction and help in solving the task.
    - The questions should be about the past events or observations upto the QUERY-FRAME INDEX, not the future events.
    - Include when, what, where, how much, how many, description, etc. in the question, especially depending on the task demands.
    - Avoid trivial questions like "what is in this image?" or asking about gripper action
    - Questions should NOT include generic references like 'first frame,' 'last frame' but rather use the QUERY-FRAME INDEX provided in the prompt to indicate the frame number in the question.
    - Mention the camera and the QUERY-FRAME INDEX in each question.
    - Do NOT include questions not used or important for completing the task.
  Example 1 (with QUERY-FRAME INDEX: 19), Query: "In the eye-in-hand camera, how many sponges were there in frame 19?"
  Example 2 (with QUERY-FRAME INDEX: 12), Query: "What was the location of the blue sponge in frame 12 in the agent-view camera?"
  Example 3 (with QUERY-FRAME INDEX: 38), Query: "Describe the task events up until frame 38 from both cameras?"
  Example 4 (with QUERY-FRAME INDEX: 24), Query: "What was the location all the sponges in frame 24 in the agent-view camera?"
  Example 5 (with QUERY-FRAME INDEX: 13, between frame 10 and frame 20), Query: "When did the robot pick up the blue sponge?"

3. Answer the query
  Provide the single best answer using only the provided information provided for each of the given QUERY-FRAME INDEXES and your event notes.
  The answer should be appropriate for the referenced QUERY-FRAME INDEX and the camera. If there's no answer, provide N/A
  The answer should be concise using very few words. Make sure if there are multiple answers to the query, provide all the answers.
  Example 1, 2 sponges
  Example 2, Bbox: (x_min, y_min, x_max, y_max) (if there are multiple bboxes, provide all the bboxes separated by a commas)
  Example 3, picked one sponge, placed one in trash, picked another sponge
  Example 4, Bbox: (x_min, y_min, x_max, y_max), (x_min, y_min, x_max, y_max), ...
  Example 5, Frame: 10 to frame 20

OUTPUT FORMAT:
The JSON should contain a single "results" field with a list of dictionaries, each containing the three required fields for each of the specified QUERY-FRAME INDEX:
json{{
  "results": [
    {{
      "description-of-query": "counting / task-events / location / etc, camera-name",
      "query": "<one memory-based question for the first QUERY-FRAME INDEX>",
      "answer": "<concise answer grounded in the frames for the first QUERY-FRAME INDEX>"
    }},
    {{
      "description-of-query": "counting / task-progress / location / etc, camera-name",
      "query": "<one memory-based question for the second QUERY-FRAME INDEX>",
      "answer": "<concise answer grounded in the frames for the second QUERY-FRAME INDEX>"
    }},
    ...
}}

IMPORTANT: 
- Only generate queries for the QUERY-FRAME INDEXES specified below.
- Answer should be short and concise
- Vary query types across different frame indices (e.g., location, counting, previous events, camera names, etc.). Do not use the same query type for consecutive QUERY-FRAME INDEXES.
- Do not include queries about things or events not relevant for solving the task

We provide the specific details below:

TASK INSTRUCTION: {task_language}

TASK DESCRIPTION: {task_description}
'''
