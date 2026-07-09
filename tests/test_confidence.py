from state import RoutingState
from evaluators.confidence import ConfidenceEvaluator

evaluator = ConfidenceEvaluator()

test_cases = [

    {
        "name": "Easy Math (Perfect)",
        "query": "What is 2 + 2?",
        "answer": "4",
        "logprobs": [-0.08, -0.10, -0.09],
        "difficulty": "easy",
        "task_type": "math",
    },

    {
        "name": "Hard Logic",
        "query": "Three friends own different pets. Deduce who owns the cat.",
        "answer": "Lee owns the cat.",
        "logprobs": [-1.4, -1.5, -1.2],
        "difficulty": "hard",
        "task_type": "logic",
    },

    {
        "name": "Low Confidence",
        "query": "Explain quantum gravity.",
        "answer": "Maybe.",
        "logprobs": [-2.8, -2.5, -2.9],
        "difficulty": "hard",
        "task_type": "factual",
    },

    {
        "name": "Code Generation",
        "query": "Write a Python function that returns factorial.",
        "answer": """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""",
        "logprobs": [-0.35, -0.30, -0.40],
        "difficulty": "medium",
        "task_type": "code_generation",
    },

    {
        "name": "NER",
        "query": "Extract entities from: Maria joined Google in London.",
        "answer": """
Maria : PERSON
Google : ORGANIZATION
London : LOCATION
""",
        "logprobs": [-0.18, -0.20, -0.15],
        "difficulty": "easy",
        "task_type": "ner",
    },

    {
        "name": "Bad Answer",
        "query": "Summarize this paragraph.",
        "answer": "I don't know.",
        "logprobs": [-0.05, -0.04, -0.06],
        "difficulty": "medium",
        "task_type": "summarization",
    },

]

print("=" * 70)

for test in test_cases:

    state = RoutingState(query=test["query"])

    state.local_answer = test["answer"]
    state.logprobs = test["logprobs"]
    state.difficulty = test["difficulty"]
    state.task_type = test["task_type"]

    state = evaluator.evaluate(state)

    print(f"\n{test['name']}")
    print("-" * 70)
    print(f"Task Type      : {state.task_type}")
    print(f"Difficulty     : {state.difficulty}")
    print(f"Route Score    : {state.route_score}")
    print(f"Breakdown      : {state.confidence_details}")

print("\n" + "=" * 70)