from router import HybridRouter
from state import RoutingState

router = HybridRouter()

query = input("Query: ")

state = RoutingState(query=query)
state = router.route(state)

print("\nFinal Answer:")
print(state.final_answer)