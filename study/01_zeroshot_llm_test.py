import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentence_transformers import SentenceTransformer, util
from validator import validate_label, validate_reasoning
from promptloader import PromptLoader
from dataloader import DataLoader
from llm import LLM
import itertools
import json
import random


## Load API Key
from dotenv import load_dotenv
load_dotenv()
GWDG_KEY = os.getenv("GWDG_KEY")  

## LLM

dataloader = DataLoader()
promptloader = PromptLoader()

SPLIT_SEED = 42


# sentence transformers
# st_model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')


import concurrent.futures

def run_with_timeout(llm, prompt, seed, timeout=300):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(llm.predict, prompt, seed)
        try:
            output, duration = future.result(timeout=timeout)
            return output, duration
        except concurrent.futures.TimeoutError:
            raise TimeoutError("Vorhersage hat länger als 20 Sekunden gedauert.")


def get_embeddings_dict(text_list):
    """
    Get embeddings for the given list of texts using the SentenceTransformer model. return a dictionary with the text and its corresponding embedding.
    """
    embeddings = {}
    for text in text_list:
        embedding = st_model.encode(text, convert_to_tensor=True)
        embeddings[text] = embedding
    return embeddings


def get_similarity(embedding1, embedding2):
    """
    Calculate the cosine similarity between two embeddings.
    """
    # Compute cosine similarity
    return util.pytorch_cos_sim(embedding1, embedding2)
    
    
def sort_fs_examples(example_test, fs_examples, fs_embeddings):
    """
    Sort the few-shot examples based on their similarity to the test example.
    Highest Simlarity last.
    """
    # Get the embedding for the test example
    embedding_test = st_model.encode(example_test["text"], convert_to_tensor=True)

    # Calculate similarities
    similarities = []
    for text in fs_embeddings.keys():
        embedding = fs_embeddings[text]
        similarity = get_similarity(embedding, embedding_test)
        similarities.append((similarity.item(), text))  # Store similarity and index
    # Sort by similarity
    similarities.sort(reverse=True, key=lambda x: x[0])
    
    # return closest examples from dataset_train. find examples in dataset_train via "text" key of example
    sorted_examples = []
    for _, text in similarities:
        for example in fs_examples:
            if example["text"] == text:
                sorted_examples.append(example)
                break
    return sorted_examples



def zero_shot(TASK, DATASET_NAME, DATASET_TYPE, LLM_BASE_MODEL, SEED, MODE, N_FEW_SHOT, SORT_EXAMPLES):

    print(f"TASK:", TASK)
    print(f"DATASET_NAME: {DATASET_NAME}")
    print(f"DATASET_TYPE: {DATASET_TYPE}")
    print(f"LLM_BASE_MODEL: {LLM_BASE_MODEL}")
    print(f"SEED: {SEED}")
    print(f"MODE: {MODE}")
    print(f"N_FEW_SHOT: {N_FEW_SHOT}")
    print(f"SORT_EXAMPLES: {SORT_EXAMPLES}")
    
    ## Load Model

    llm = LLM(LLM_BASE_MODEL, parameters=[
        {"name": "stop", "value": [")]"]}, 
        {"name": "num_ctx", "value": "4096"}
        ]) #8192

    ## Load Eval Dataset

    dataset_test = dataloader.load_data(name=DATASET_NAME, data_type=DATASET_TYPE, target=TASK)

    #Truncating the dataset_test for running purposes
    dataset_test = dataset_test[:100]
    
    ## Unique Aspect Categories

    unique_aspect_categories = sorted({aspect['aspect_category'] for entry in dataloader.load_data(name=DATASET_NAME, data_type="all", target=TASK) for aspect in entry['aspects']})
    if DATASET_NAME == "gerest" and not("food general" in unique_aspect_categories):
        unique_aspect_categories += ["food general"]
    unique_aspect_categories = sorted(unique_aspect_categories)
    predictions = []
    
    ## Load Few-Shot Dataset
    few_shot_split_0 = []

    if (N_FEW_SHOT > 0):
        dataset_train = dataloader.load_data(name=DATASET_NAME, data_type="train", target=TASK)
        few_shot_split_0 = dataloader.random_cross_validation_split(dataset_train, seed=SPLIT_SEED)[0] + dataloader.random_cross_validation_split(dataset_train, seed=SPLIT_SEED)[1] + dataloader.random_cross_validation_split(dataset_train, seed=SPLIT_SEED)[2] + dataloader.random_cross_validation_split(dataset_train, seed=SPLIT_SEED)[3] + dataloader.random_cross_validation_split(dataset_train, seed=SPLIT_SEED)[4]
        
        random.seed(SPLIT_SEED)
        few_shot_split_0 = few_shot_split_0[0:N_FEW_SHOT]
         
    fs_examples_ids = [int(example["id"].split("_")[0]) for example in few_shot_split_0]

    # Lade alle Zeilen aus der Datei
    # Load all lines from the file
    fs_examples_txt = ""
    with open(f"./datasets/{TASK}/{DATASET_NAME}/train.txt", "r") as f:
        lines = f.readlines()

        # Füge die Zeilen zusammen, deren Index in fs_examples_ids enthalten ist
        # Merge the lines whose index is contained in fs_examples_ids
        fs_examples_txt = "".join(lines[i] for i in fs_examples_ids if 0 <= i < len(lines))

    # Erstelle den Zielpfad, falls er nicht existiert
    # Create the target path if it doesn't exist
    output_dir_fs = f"./fs_examples/{TASK}/{DATASET_NAME}/fs_{N_FEW_SHOT}"
    os.makedirs(output_dir_fs, exist_ok=True)

    # Speichere die resultierenden Beispiele in einer Datei
    # Save the resulting examples to a file
    output_path_fs = os.path.join(output_dir_fs, "examples.txt")
    with open(output_path_fs, "w") as f:
        f.write(fs_examples_txt)
        
    #############################################
    #############################################
    
    ## Load Train Dataset
    #fs_embeddings = get_embeddings_dict([example["text"] for example in few_shot_split_0])

    ## label
    if MODE in ["label", "chain-of-thought", "plan-and-solve"]:
     for idx, example in enumerate(dataset_test):
        prediction = { 
            "task": TASK,
            "dataset_name": DATASET_NAME, 
            "dataset_type": DATASET_TYPE,
            "llm_base_model": LLM_BASE_MODEL,
            "mode": MODE,
            "id": example["id"], 
            "invalid_precitions_label": [],
            "init_seed": SEED,
        }
        

        seed = SEED
        
        #few_shot_split_0 = sort_fs_examples(example, few_shot_split_0, fs_embeddings)

            
        prompt = promptloader.load_prompt(task=TASK,
                                      prediction_type=MODE, 
                                      aspects=unique_aspect_categories, 
                                      examples=few_shot_split_0,
                                      seed_examples=seed,
                                      input_example=example, shuffle_examples=SORT_EXAMPLES==False) # use False if specific order

        correct_output = False   
        while correct_output == False:
            while True:
              try:
                  output, duration = run_with_timeout(llm, prompt, seed)
                  break  # Erfolgreiche Ausführung -> Schleife beenden
              except Exception as e:
                  print(f"Fehler aufgetreten: {e}, versuche es erneut...")

            output_raw = output
            # delete new lines
            output = output.replace("\n", "")
            
            validator_output = validate_label(output, example["text"], unique_aspect_categories, task=TASK, allow_small_variations=True)

            if validator_output[0] != False:
                prediction["pred_raw"] = output_raw
                prediction["pred_label"] = validator_output[0]
                prediction["duration_label"] = duration
                prediction["seed"] = seed
                correct_output = True
            else:
                prediction["invalid_precitions_label"].append({"pred_label_raw": output_raw, "pred_label": validator_output[0], "duration_label": duration, "seed": seed, "regeneration_reason": validator_output[1]})
                seed += 5
                pass
        
            if len(prediction["invalid_precitions_label"]) > 9:
                correct_output = True
                prediction["pred_label"] = []
                prediction["duration_label"] = duration
                prediction["seed"] = seed
    
        print("########## ", idx, "\nText:", example["text"], "\nLabel:",prediction["pred_label"], "\nRegenerations:", prediction["invalid_precitions_label"])
        predictions.append(dict(prediction, **example))

    dir_path = f"generations/llm_test"

    # Create the directories if they don't exist
    os.makedirs(dir_path, exist_ok=True)
    print("Number of predictions to write:", len(predictions))
    with open(f"{dir_path}/{TASK}_{DATASET_NAME}_{DATASET_TYPE}_{LLM_BASE_MODEL.split(':')[0]}_{SEED}_{MODE}_{N_FEW_SHOT}.json", 'w', encoding='utf-8') as json_file:
        json.dump(predictions, json_file, ensure_ascii=False, indent=4)
        
        

##### Zero-Shot

# tasks = ["asqp", "tasd"]
# datasets = ["rest15", "rest16"]
# dataset_types = ["train", "test", "dev"]
# models = ["gemma3:27b", "llama3.1:70b"]
# seeds = [0, 1, 2, 3, 4]
# modes = ["chain-of-thought", "plan-and-solve", "label"] # "label"

seeds = [0]
n_few_shot = [50] # 0 fehlt noch
datasets = ["rest16"]
tasks = ["asqp"]
dataset_types = ["test"]
#Tried "deepseek-r1:latest" but it couldn't fit properly on my GPU
#qwen3:4b does not output the aspectss as expected
models = ["gemma3:4b", "cogito:3b", "gemma3n:e4b"]
modes = ["label"] # "label"
sort_examples = [False]


combinations = itertools.product(n_few_shot, datasets, tasks, dataset_types, models, modes, sort_examples, seeds)

import time
import subprocess

for combination in combinations:
    fs,  dataset_name, task, dataset_type, model, mode, s_ex, seed = combination
    file_path = f"generations/llm_test/{task}_{dataset_name}_{dataset_type}_{model.split(':')[0]}_{seed}_{mode}_{fs}.json"
    # Prüfen, ob die Datei bereits existiert
    if not os.path.exists(file_path):
        time.sleep(2)
        subprocess.run(["ollama", "stop", model])
        zero_shot(task, dataset_name, dataset_type, model, seed, mode, fs, s_ex)
    else:
        print(f"Skipping: {file_path} already exists.")