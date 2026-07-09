from analyzers.task_analyzer import TaskAnalyzer
from state import RoutingState

analyzer = TaskAnalyzer()

state = RoutingState(
    query="Summarize this paragraph in one sentence."
)

state = analyzer.analyze(state)

print(state.task_type)
print(state.difficulty)
print(state.tool_candidate)