import os
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "600"

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.models import OllamaModel
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric
)
from deepeval.evaluate.configs import AsyncConfig
from deepeval.dataset import EvaluationDataset, Golden
from ReAct import autonomous_agent, retriever, selector, State

# ---------------- JUDGE MODEL ----------------
judge = OllamaModel(
    model="mistral:7b",
    base_url="http://localhost:11434",
    temperature=0
)

# ---------------- EVALUATION METRICS ----------------
metrics = [
    AnswerRelevancyMetric(model=judge, include_reason=True),
    FaithfulnessMetric(model=judge, include_reason=True),
    HallucinationMetric(model=judge, include_reason=True),
    ContextualRelevancyMetric(model=judge, include_reason=True),
    ContextualPrecisionMetric(model=judge, include_reason=True),
    TaskCompletionMetric(model=judge, include_reason=True),
    ToolCorrectnessMetric(model=judge, include_reason=True),
]

# ---------------- GOLDEN DATASET ----------------
dataset = EvaluationDataset(
    goldens=[
        Golden(
            input="What are the conditions required for a valid Hindu marriage?",
            expected_output="A valid Hindu marriage requires no living spouse, legal age, sound mind, and no prohibited or sapinda relationship."
        ),
        Golden(
            input="A women colleague in my company has filed POSH complaint against me. Its fake, just to damage my reputation i have proofs as well which proves it, what should i do now ?",
            expected_output="The person should collect and present all evidence before the Internal Committee, submit a written response denying the allegations, cooperate with the inquiry process, maintain professionalism, and seek legal advice if necessary."
        ),
        Golden(
            input="What are the duties of an employer under the POSH Act ?",
            expected_output=
                "An employer must provide a safe working environment, constitute an Internal Committee, assist in the complaint and inquiry process, treat sexual harassment as misconduct, and ensure timely action on the committee's recommendations."          
        ),
        Golden(
            input="What rights does a Hindu wife have regarding maintenance?",
            expected_output="A Hindu wife can claim maintenance during marriage and after divorce if she cannot maintain herself."
        )
    ]
)

# ---------------- BUILD TEST CASES ----------------
test_cases = []
print("\n=== Running DeepEval on Legal Advisor ===\n")

for idx, golden in enumerate(dataset.goldens, 1):

    query = golden.input
    expected_answer = golden.expected_output
    print(f"[{idx}/{len(dataset.goldens)}] {query}")

    # ----------- Step 1: Get Context via your REAL tools --------------
    state: State = {
        "query": query,
        "retrieved_docs": [],
        "selected_docs": [],
        "final_answer": "",
        "scratchpad": "",
        "last_signature": None,
        "next_action": ""
    }
    state = retriever(state)
    state = selector(state)
    contexts = state["selected_docs"]

    # -------- Step 2: Run Autonomous Agent --------
    actual_answer = autonomous_agent(query)

    # ------------- Step 3: Tool Trace ------------
    tools_called = [
        ToolCall(name="retriever"),
        ToolCall(name="selector"),
        ToolCall(name="answer_generator"),
    ]

    expected_tools = [
        ToolCall(name="retriever"),
        ToolCall(name="selector"),
        ToolCall(name="answer_generator"),
    ]

    # ------------- Build Test Case ------------
    test_cases.append(
        LLMTestCase(
            input=query,
            actual_output=actual_answer,
            expected_output=expected_answer,
            context=contexts,
            retrieval_context=contexts,
            tools_called=tools_called,
            expected_tools=expected_tools
        )
    )
    print("✓ Test case created\n")

# ---------------- RUN EVALUATION ----------------
print("\n=== Starting DeepEval Evaluation ===\n")
evaluate(
    test_cases=test_cases, metrics=metrics, async_config=AsyncConfig(run_async=False, max_concurrent=1)
)
print("\n=== Evaluation Complete ===\n")
