from openai import OpenAI
import json
from sklearn.metrics import f1_score, recall_score, precision_score
import dspy
from tqdm import tqdm
from collections import defaultdict
from dspy.teleprompt import MIPROv2
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
import random
from sklearn.model_selection import KFold

'''model = 'openai/meta-llama/Llama-3.3-70B-Instruct'               #70B on port 8001
modelname = '70B'
client = OpenAI(api_key='EMPTY', base_url='http://0.0.0.0:8001/v1/')'''

'''model = 'openai/Qwen2.5-32B-Instruct'
modelname = 'Qwen_32B'
client = OpenAI(api_key='EMPTY', base_url='http://0.0.0.0:8006/v1/')'''

model = 'openai/Qwen/Qwen2.5-7B-Instruct'
modelname = 'Qwen_7B'
client = OpenAI(api_key='EMPTY', base_url='http://0.0.0.0:8002/v1/')

class LoggingLM(dspy.LM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_prompt = None

    def __call__(self, *args, **kwargs):
        if self.model_type == "chat":
            self.last_messages = kwargs.get("messages", None)
        else:
            if 'prompt' in kwargs:
                self.last_prompt = kwargs['prompt']
            elif args:
                self.last_prompt = args[0]
        return super().__call__(*args, **kwargs)

not_logging_lm = dspy.LM(
    model=model, 
    api_base='http://0.0.0.0:8001/v1/',
    api_key='EMPTY', 
    model_type='chat'
)

lm = LoggingLM(
    model=model, 
    api_base='http://0.0.0.0:8002/v1/',
    api_key='EMPTY', 
    model_type='chat'
)
dspy.configure(lm=lm)

teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="medium", # Can choose between light, medium, and heavy optimization runs
)

# GLOBAL CONSTANT
exp_no = 15

class PredictClassificationLabel(dspy.Signature):
    "Classify whether the query is asking for information that the assistant has provided in previous interactions (A), or if it is asking for information that is found in the user’s own chat history and user provided details (B)."
    query = dspy.InputField(desc="User's natural language question")
    answer = dspy.OutputField(desc="A or B")

def load_dspy_examples():
    file_path = "dspy_examples.json"
    with open(file_path, "r") as f:
        data = json.load(f)
    return [dspy.Example(query=ex['query'], answer='A' if ex['label']==1 else 'B').with_inputs('query') for ex in data]

def old_evaluate_program(program, test_queries):
    for test in test_queries:
        query = test["query"]
        print(test)
        expected = 'A' if test.answer == 1 else 'B'
        result = program(query=query)
        print(f"Query: {query}")
        print(f"Predicted: {result.answer}, Expected: {expected}")
        print("Correct:", result.answer == expected)
        print("-" * 40)

def evaluate_program(program, test_queries, detailed_path, summary_path):
    results = []
    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for test in test_queries:
        #print(test)
        query = test["query"]
        true_label = test["answer"]
        predicted_label = program(query=query).answer
        is_correct = predicted_label == true_label

        # Save detailed result
        results.append({
            "query": query,
            "expected": true_label,
            "predicted": predicted_label,
            "correct": is_correct
        })

        # Update confusion counts per class
        for cls in ['A', 'B']:
            if true_label == cls and predicted_label == cls:
                class_stats[cls]['tp'] += 1
            elif true_label != cls and predicted_label == cls:
                class_stats[cls]['fp'] += 1
            elif true_label == cls and predicted_label != cls:
                class_stats[cls]['fn'] += 1

    # Compute per-class metrics
    per_class_metrics = {}
    total_correct = sum(1 for r in results if r['correct'])

    for cls, stats in class_stats.items():
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

        per_class_metrics[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn
        }

    # Final summary
    accuracy = total_correct / len(test_queries) if test_queries else 0.0
    summary = {
        "total": len(test_queries),
        "correct": total_correct,
        "accuracy": round(accuracy, 4),
        "per_class": per_class_metrics
    }

    # Write files
    with open(detailed_path, "a") as f:
        json.dump(results, f, indent=2)

    with open(summary_path, "a") as f:
        json.dump(summary, f, indent=2)

def convert_train_data_to_dspy_examples():
    filename = 'm_queries.json'

    with open(filename, 'r') as f:
        raw_data = json.load(f)
    
    dspy_examples =  [
        dspy.Example(query=item['question'], label=item['label'])
        for item in raw_data
    ]

    examples_dicts = [ex.toDict() for ex in dspy_examples]
    with open("dspy_examples.json", "w") as f:
        json.dump(examples_dicts, f, indent=2)

def load_test_queries(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def test_main():
    '''#dspy.configure(lm=lm)
    query = 'Can you remind me of the name of the romantic Italian restaurant in Rome you recommended for dinner?'
    classify = dspy.Predict(PredictClassificationLabel)
    result = classify(query=query)
    print(result)'''
    #convert_train_data_to_dspy_examples()

    query = 'Can you remind me of the name of the romantic Italian restaurant in Rome you recommended for dinner?'

    with open("dspy_examples.json", "r") as f:
        examples_data = json.load(f)
    
    examples = [dspy.Example(query=ex['query'], answer='A' if ex['label']==1 else 'B').with_inputs('query') for ex in examples_data]
    print(f'Examples loaded in DSPY format.')
    print(examples[0])

    optimized_program = teleprompter.compile(
        dspy.Predict(PredictClassificationLabel),
        trainset=examples,
        requires_permission_to_run=False
    )
    result=optimized_program(query=query)
    print(result.answer)
    for msg in lm.last_messages:
        print(f"{msg['role'].upper()}: {msg['content']}")

def main():
    detailed_path = 'optimized_results_detailed.json'
    summary_path = 'optimized_results_summary.json'
    # Load and shuffle all examples
    examples = load_dspy_examples()
    random.shuffle(examples)

    # Setup cross-validation
    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(examples)):
        print(f"\n=== Fold {fold_idx+1} ===")
        trainset = [examples[i] for i in train_idx]
        valset = [examples[i] for i in val_idx]

        # Compile the model using the train split
        optimized_program = teleprompter.compile(
            dspy.Predict(PredictClassificationLabel),
            trainset=trainset,
            requires_permission_to_run=False
        )

        # Evaluate on the validation split
        print("\nValidation Set Evaluation:")
        evaluate_program(optimized_program, valset, detailed_path, summary_path)
    
    with open("messages.txt", "w") as f:
        for msg in lm.last_messages:
            f.write(f"{msg['role'].upper()}: {msg['content']}\n")


    # Final evaluation on external test queries
    test_queries = examples
    final_program = teleprompter.compile(
        dspy.Predict(PredictClassificationLabel),
        trainset=examples,  # full training data
        requires_permission_to_run=False
    )
    detailed_path = 'final_results_detailed.json'
    summary_path = 'final_results_summary.json'
    print(f"\nFinal Evaluation on Test Queries dumped to:{summary_path}")
    evaluate_program(final_program, test_queries, detailed_path, summary_path)

if __name__ == "__main__":
    main()