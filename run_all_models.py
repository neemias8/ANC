import subprocess
import sys
import json
import os

# Define the scripts to run
scripts = [
    'run_taeg.py',
    'test_other_models.py',
    'primera_standard_mds.py',
    'primera_event_consolidation.py'
]

# Run each generation script
for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in {script}: {result.stderr}")
        sys.exit(1)
    else:
        print(f"{script} completed successfully.")

# Now run the evaluator
print("Running evaluator...")
result = subprocess.run([sys.executable, 'evaluate_existing_outputs.py'], capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error in evaluator: {result.stderr}")
    sys.exit(1)
else:
    print("Evaluator completed successfully.")

# Load evaluation results
evaluation_dir = 'outputs/evaluation'
results = {}

for file in os.listdir(evaluation_dir):
    if file.endswith('.json'):
        model_name = file.replace('_results.json', '').replace('_', ' ').title()
        with open(os.path.join(evaluation_dir, file), 'r') as f:
            data = json.load(f)
            results[model_name] = data

# Prepare and print table
print("\nFinal Results Table:")
print(f"{'Model':<25} {'ROUGE-1 F1':<12} {'ROUGE-2 F1':<12} {'ROUGE-L F1':<12} {'METEOR':<10} {'BERTScore F1':<14} {'Kendall Tau':<12}")
print("-" * 110)

for model, metrics in results.items():
    rouge1_f1 = metrics.get('rouge', {}).get('rouge1', {}).get('f1', 'N/A')
    rouge2_f1 = metrics.get('rouge', {}).get('rouge2', {}).get('f1', 'N/A')
    rougeL_f1 = metrics.get('rouge', {}).get('rougeL', {}).get('f1', 'N/A')
    meteor = metrics.get('meteor', 'N/A')
    bertscore_f1 = metrics.get('bertscore', {}).get('f1', 'N/A')
    kendall_tau = metrics.get('kendall_tau', 'N/A')
    
    # Format as strings with 4 decimals if float
    def fmt(val):
        return f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
    
    print(f"{model:<25} {fmt(rouge1_f1):<12} {fmt(rouge2_f1):<12} {fmt(rougeL_f1):<12} {fmt(meteor):<10} {fmt(bertscore_f1):<14} {fmt(kendall_tau):<12}")