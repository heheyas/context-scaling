import os
import ast
import pandas as pd
import megfile
from rich.table import Table
from rich.console import Console

MODE = "EN" # MODE = "ZH"
RESULT_DIR = "results"
DATA_FILE = f"OneIG-Bench{'-ZH' if MODE == 'ZH' else ''}.csv"
SOURCE_ID = {
    "anime": "Anime_Stylization",
    "human": "Portrait",
    "object": "General_Object",
    "text": "Text_Rendering",
    "reasoning": "Knowledge_Reasoning",
    "multilingualism": "Multilingualism"
}

KEYS = ["alignment", "diversity", "reasoning", "style", "text"]

# === Load Base Data ===
df = pd.read_csv(DATA_FILE, dtype=str)
model_names = os.listdir(RESULT_DIR)
# If you run several models and save results in one csv, you can replace 
# '''file_path = f"{RESULT_DIR}/{model_name}/{key}_prompt_score_{MODE}.csv"'''
# to '''file_path = f"{RESULT_DIR}/xxx/{key}_prompt_score_{MODE}.csv"'''

# === Display Utilities ===
def display_model_scores_table(raw_data: dict, categories: list, title: str):
    console = Console()
    table = Table(title=title, show_lines=True)
    table.add_column("Model", justify="left", style="bold")
    for cat in categories:
        table.add_column(cat, justify="center")

    csv_rows = []
    for model, scores in raw_data.items():
        row = [model]
        csv_row = {"Model": model}
        for cat in categories:
            score = scores.get(cat, {}).get('score')
            val = f"{score:.3f}" if score is not None else "-"
            row.append(val)
            csv_row[cat] = val
        table.add_row(*row)
        csv_rows.append(csv_row)

    console.print(table)
    pd.DataFrame(csv_rows).to_csv(f"{title}.csv", index=False)

# === Score Aggregation ===
def compute_scores(model_name, file_path, filter_fn):
    data = pd.read_csv(file_path, dtype=str)
    scores = [
        float(row[model_name])
        for _, row in data.iterrows()
        if filter_fn(df, row)
    ]
    return {
        "score": sum(scores) / len(scores) if scores else None,
        "num": len(scores)
    }

# === Handlers ===
def handle_prompt_based_metric(key):
    result = {}
    for model_name in model_names:
        file_path = megfile.smart_glob(f"{RESULT_DIR}/{model_name}/{key}_prompt_score_{MODE}*.csv")[0]
        if not file_path:
            return f"File not found for {model_name} in {key} metric."
        result[model_name] = {}
        for tag in ["short", "middle", "long"]:
            result[model_name][tag] = compute_scores(
                model_name, file_path,
                lambda df_, row: df_[(df_["category"] == SOURCE_ID[row["Unnamed: 0"].split('_')[0]]) & 
                                     (df_["id"] == row["Unnamed: 0"].split('_')[1])]["prompt_length"].iloc[0] == tag
            )
        for tag in ["T, P", "NP"]:
            result[model_name][tag] = compute_scores(
                model_name, file_path,
                lambda df_, row: df_[(df_["category"] == SOURCE_ID[row["Unnamed: 0"].split('_')[0]]) & 
                                     (df_["id"] == row["Unnamed: 0"].split('_')[1])]["type"].iloc[0] == tag
            )
    display_model_scores_table(result, ['NP', 'T, P', 'short', 'middle', 'long'], f"{key.capitalize()} Model Scores")

def handle_reasoning():
    result = {}
    subjects = ["geography", "computer science", "biology", "mathematics", "physics", "chemistry", "common sense"]
    for model_name in model_names:
        result[model_name] = {}
        file_path = megfile.smart_glob(f"{RESULT_DIR}/{model_name}/reasoning_prompt_score_{MODE}*.csv")[0]
        if not file_path:
            return f"File not found for {model_name} in {key} metric."
        data = pd.read_csv(file_path, dtype=str)
        for subject in subjects:
            scores = [
                float(row[model_name]) for _, row in data.iterrows()
                if df[(df["category"] == SOURCE_ID["reasoning"]) & (df["id"] == row["Unnamed: 0"])]["class"].iloc[0] == subject
            ]
            result[model_name][subject] = {
                "score": sum(scores)/len(scores) if scores else None,
                "num": len(scores)
            }
    display_model_scores_table(result, subjects, "Reasoning Model Scores")

def handle_style():
    style_types = {
        "traditional": ['abstract_expressionism', 'art_nouveau', 'baroque', 'chinese_ink_painting', 'cubism', 'fauvism', 'impressionism', 'line_art', 'minimalism', 'pointillism', 'pop_art', 'rococo', 'ukiyo-e'],
        "media": ['clay', 'crayon', 'graffiti', 'lego', 'comic', 'pencil_sketch', 'stone_sculpture', 'watercolor'],
        "anime": ['celluloid', 'chibi', 'cyberpunk', 'ghibli', 'impasto', 'pixar', 'pixel_art', '3d_rendering']
    }
    result = {}
    for model_name in model_names:
        result[model_name] = {}
        file_path = megfile.smart_glob(f"{RESULT_DIR}/{model_name}/style_style_score_{MODE}*.csv")[0]
        if not file_path:
            return f"File not found for {model_name} in {key} metric."
        data = pd.read_csv(file_path, dtype=str)
        print(data)
        for style_cat, styles in style_types.items():
            flat_scores = [float(row[style]) for _, row in data.iterrows() for style in styles]
            result[model_name][style_cat] = {
                "score": sum(flat_scores)/len(flat_scores) if flat_scores else None
            }
    display_model_scores_table(result, list(style_types.keys()), "Style Model Scores")

def handle_text():
    metrics = ["ED", "CR", "WAC"]
    lengths = ["short", "middle", "long"]
    result = {}
    for model_name in model_names:
        result[model_name] = {}
        file_path = megfile.smart_glob(f"{RESULT_DIR}/{model_name}/text_prompt_score_{MODE}*.csv")[0]
        if not file_path:
            return f"File not found for {model_name} in {key} metric."
        data = pd.read_csv(file_path, dtype=str)
        for length in lengths:
            filtered = [
                ast.literal_eval(row[model_name]) for _, row in data.iterrows()
                if df[(df["category"] == SOURCE_ID["text"]) & (df["id"] == row["Unnamed: 0"])]["prompt_length"].iloc[0] == length
            ]
            if filtered:
                avg = [sum(metric)/len(metric) for metric in zip(*filtered)]
                for i, m in enumerate(metrics):
                    result[model_name][f"{length}_{m}"] = {"score": avg[i]}
    display_model_scores_table(result, [f"{l}_{m}" for m in metrics for l in lengths], "Text Model Scores")

# === Main Execution ===
if __name__ == "__main__":
    if MODE == "EN":
        for key in KEYS:
            if key in ["alignment", "diversity"]:
                handle_prompt_based_metric(key)
            elif key == "reasoning":
                handle_reasoning()
            elif key == "style":
                handle_style()
            elif key == "text":
                handle_text()
    elif MODE == "ZH":
        for key in KEYS:
            if key == "reasoning":
                handle_reasoning()
            elif key == "style":
                handle_style()