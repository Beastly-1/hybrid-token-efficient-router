from state import RoutingState
from decision.decision_engine import DecisionEngine

engine = DecisionEngine()

test_cases = [

    {
        "name": "Easy Math (Local)",
        "task_type": "math",
        "route_score": 0.96,
    },

    {
        "name": "Easy Math (Remote)",
        "task_type": "math",
        "route_score": 0.60,
    },

    {
        "name": "NER (Local)",
        "task_type": "ner",
        "route_score": 0.90,
    },

    {
        "name": "NER (Remote)",
        "task_type": "ner",
        "route_score": 0.65,
    },

    {
        "name": "Hard Logic",
        "task_type": "logic",
        "route_score": 0.70,
    },

    {
        "name": "Logic (Excellent)",
        "task_type": "logic",
        "route_score": 0.98,
    },

    {
        "name": "Code Generation",
        "task_type": "code_generation",
        "route_score": 0.91,
    },

    {
        "name": "Summarization",
        "task_type": "summarization",
        "route_score": 0.86,
    },

    {
        "name": "Factual",
        "task_type": "factual",
        "route_score": 0.81,
    },

    {
        "name": "Unknown Task",
        "task_type": "random_task",
        "route_score": 0.90,
    }

]

print("=" * 80)
print("DECISION ENGINE TEST")
print("=" * 80)

for test in test_cases:

    state = RoutingState(
        query="Dummy Question"
    )

    state.task_type = test["task_type"]
    state.route_score = test["route_score"]
    state.local_answer = "LOCAL MODEL ANSWER"

    state = engine.decide(state)

    print("\n" + "-" * 80)
    print(f"Test           : {test['name']}")
    print(f"Task Type      : {state.task_type}")
    print(f"Route Score    : {state.route_score}")
    print(f"Use Remote     : {state.use_remote}")
    print(f"Final Answer   : {state.final_answer}")
    print(f"Routing Reason : {state.routing_reason}")

print("\n" + "=" * 80)
print("Decision Engine Tests Completed")
print("=" * 80)