from router import HybridRouter
from state import RoutingState


def run_test(title, query):

    print("=" * 80)
    print(title)
    print("=" * 80)

    state = RoutingState(query=query)

    router = HybridRouter()

    result = router.route(state)

    print("Query            :", query)
    print("Task Type        :", result.task_type)
    print("Difficulty       :", result.difficulty)
    print("Tool Candidate   :", result.tool_candidate)
    print("Tool Used        :", result.tool_used)
    print("Use Remote       :", result.use_remote)
    print("Route Score      :", result.route_score)
    print("Routing Reason   :", result.routing_reason)
    print("Final Answer     :", result.final_answer)

    print()


def main():

    tests = [

        (
            "Calculator",
            "sqrt(81)+factorial(4)-log(e)"
        ),

        (
            "JSON Validation",
            'Validate JSON {"name":"John","age":21}'
        ),

        (
            "Regex Email",
            "Verify email abc@gmail.com"
        ),

        (
            "Regex URL",
            "Verify url https://openai.com"
        ),

        (
            "Date Time",
            "What day of week is 2026-07-09"
        ),

        (
            "Summarization",
            """Summarize:
            AMD released a new GPU architecture
            focused on AI acceleration while
            reducing power consumption."""
        ),

        (
            "NER",
            """Extract named entities from:
            Sundar Pichai visited Bengaluru
            while meeting AMD executives."""
        ),

        (
            "Sentiment",
            """Classify sentiment:
            The phone performs extremely well
            although the battery is mediocre."""
        ),

        (
            "Logic",
            """Three friends own a cat,
            dog and bird.
            Sam doesn't own the bird.
            Lee owns the cat.
            Who owns the dog?"""
        ),

        (
            "Code Debug",
            """
            Debug:

            def maximum(nums):
                return nums[0]
            """
        ),

        (
            "Code Generation",
            """
            Write a Python function that
            returns the second largest
            element in a list.
            """
        ),

        (
            "Factual",
            """
            Explain how transformers
            differ from recurrent neural
            networks.
            """
        ),

        (
            "Hard Math (Should Escalate)",
            """
            A company grows revenue
            by 13% every quarter for
            seven quarters.
            Starting revenue is
            $18.4 million.
            Predict the revenue.
            """
        ),

        (
            "Hard Reasoning",
            """
            Five people sit around a circular
            table. Alice sits opposite Bob.
            Carol sits immediately left of Alice.
            Dave cannot sit next to Bob.
            Determine every valid arrangement.
            """
        ),

    ]

    for title, query in tests:

        run_test(title, query)


if __name__ == "__main__":
    main()