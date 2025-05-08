from dataloader import DataLoader
from helper import DotDict
import shutil
import os

def train_mvp(**kwargs):
    from classifier.mvp.classifier import train_function_mvp
    # Standardwerte
    default_args = DotDict({
        "data_path": "./data",
        "dataset": "rest15",  # Standardwert für dataset
        "model_name_or_path": "t5-base",
        "output_dir": "../outputs/",
        "num_train_epochs": 20,
        "save_top_k": 0,
        "task": "asqp",  # Standardwert für task
        "top_k": 5,
        "ctrl_token": None,
        "multi_path": True,
        "num_path": None,
        "seed": 42,
        "train_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "lowercase": True,
        "sort_label": True,
        "data_ratio": 1.0,
        "check_val_every_n_epoch": 10,
        "agg_strategy": "vote",
        "eval_batch_size": 64,
        "constrained_decode": True,
        "do_train": True,
        "load_path_cache": False,
        "single_view_type": "rank",
        "load_ckpt_name": None,
        "max_seq_length": 200,
        "ctrl_token": "post",
        "n_gpu": 1,
        "warmup_steps": 0.0,
        "weight_decay": 0.0,
        "adam_epsilon": 1e-8,
        "eval_data_split": "test",
        "do_inference": False,
        "num_path": 1,
        "beam_size": 1,
        "train_ds": None,
        "test_ds": None,
    })

    # Überschreiben der Standardwerte mit den übergebenen Argumenten
    for key, value in kwargs.items():
        if key in default_args:
            default_args[key] = value
        else:
            raise ValueError(f"Unknown argument: {key}")

    # Daten laden
    dataloader = DataLoader()
    
    if default_args["train_ds"] is None:
       train_ds = dataloader.load_data(name=default_args['dataset'], data_type="train", target=default_args["task"])
    else:
       train_ds = default_args["train_ds"]
       
    if default_args["test_ds"] is None:
       test_ds = dataloader.load_data(name=default_args['dataset'], data_type="test", target=default_args["task"])
    else:
       test_ds = default_args["test_ds"]
    
    print("Train Dataset length:", len(train_ds))
    print("Test Dataset length:", len(test_ds))
    
    # Update train_ds und test_ds in den Argumenten
    default_args["train_ds"] = train_ds
    default_args["test_ds"] = test_ds

    # Training ausführen
    scores = train_function_mvp(default_args)

    # Outputs-Verzeichnis löschen
    shutil.rmtree('classifier/outputs')
    os.makedirs('classifier/outputs', exist_ok=True)

    return scores

# Beispielaufruf der Funktion mit überschriebenen Werten
# scores = train_mvp(task="tasd", num_train_epochs=1)
# print("SCORES:", scores)


def train_paraphrase(**kwargs):
    from classifier.paraphrase.classifier import train_function_paraphrase
    # Standardwerte
    default_args = DotDict({
    "task": "asqp",  # Vorgabe, der Wert wird im Skript gesetzt
    "dataset": "rest15",
    "model_name_or_path": "t5-base",
    "do_train": True,
    "do_eval": False,  # Standardwert, wenn nicht angegeben
    "do_direct_eval": True,
    "do_inference": False,  # Standardwert, wenn nicht angegeben
    "max_seq_length": 128,
    "n_gpu": 0,
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "learning_rate": 3e-4,
    "num_train_epochs": 20,  # Dein Wert, Standardwert war 30
    "seed": 42,
    "weight_decay": 0.0,
    "adam_epsilon": 1e-8,
    "warmup_steps": 0.0,
    "output_dir": "classifier/outputs",  # Hinzugefügt, weil es in deinem Skript verwendet wird
    "train_ds": None,
    "test_ds": None,
})

    # Überschreiben der Standardwerte mit den übergebenen Argumenten
    for key, value in kwargs.items():
        if key in default_args:
            default_args[key] = value
        else:
            raise ValueError(f"Unknown argument: {key}")

    dataloader = DataLoader()

    if default_args["train_ds"] is None:
       train_ds = dataloader.load_data(name=default_args['dataset'], data_type="train", target=default_args["task"])
    else:
       train_ds = default_args["train_ds"]
       
    if default_args["test_ds"] is None:
       test_ds = dataloader.load_data(name=default_args['dataset'], data_type="test", target=default_args["task"])
    else:
       test_ds = default_args["test_ds"]
    
    print("Train Dataset length:", len(train_ds))
    print("Test Dataset length:", len(test_ds))

    # Update train_ds und test_ds in den Argumenten
    default_args["train_ds"] = train_ds
    default_args["test_ds"] = test_ds

    # Training ausführen
    scores = train_function_paraphrase(default_args)

    # Outputs-Verzeichnis löschen
    shutil.rmtree(default_args["output_dir"])
    os.makedirs(default_args["output_dir"], exist_ok=True)

    return scores

# scores = train_paraphrase(task="tasd")
# print("SCORES:", scores["f1"])

def train_dlo(**kwargs):
    from classifier.dlo.classifier import train_function_dlo
    # Standardwerte
    default_args = DotDict({
    "task": "asqp",  # Vorgabe, der Wert wird im Skript gesetzt
    "dataset": "rest15",
    "model_name_or_path": "t5-base",
    "do_train": True,
    "do_eval": False,  # Standardwert, wenn nicht angegeben
    "do_direct_eval": True,
    "do_inference": False,  # Standardwert, wenn nicht angegeben
    "max_seq_length": 200,
    "n_gpu": 0,
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-4,
    "num_train_epochs": 20,  # Dein Wert, Standardwert war 30
    "seed": 42,
    "weight_decay": 0.0,
    "adam_epsilon": 1e-8,
    "warmup_steps": 0.0,
    "output_dir": "classifier/outputs",  # Hinzugefügt, weil es in deinem Skript verwendet wird
    "train_ds": None,
    "test_ds": None,
    "top_k": 5,
    })

    # Überschreiben der Standardwerte mit den übergebenen Argumenten
    for key, value in kwargs.items():
        if key in default_args:
            default_args[key] = value
        else:
            raise ValueError(f"Unknown argument: {key}")

    dataloader = DataLoader()

    if default_args["train_ds"] is None:
       train_ds = dataloader.load_data(name=default_args['dataset'], data_type="train", target=default_args["task"])
    else:
       train_ds = default_args["train_ds"]
       
    if default_args["test_ds"] is None:
       test_ds = dataloader.load_data(name=default_args['dataset'], data_type="test", target=default_args["task"])
    else:
       test_ds = default_args["test_ds"]
    
    print("Train Dataset length:", len(train_ds))
    print("Test Dataset length:", len(test_ds))

    # Update train_ds und test_ds in den Argumenten
    default_args["train_ds"] = train_ds
    default_args["test_ds"] = test_ds

    # Training ausführen
    scores = train_function_dlo(default_args)

    # Outputs-Verzeichnis löschen
    shutil.rmtree(default_args["output_dir"])
    os.makedirs(default_args["output_dir"], exist_ok=True)
    

    return scores

# scores = train_dlo(task="tasd", dataset="rest16")
# print("SCORES:", scores)

def train_ilo(**kwargs):
    from classifier.ilo.classifier import train_function_ilo
    # Standardwerte
    default_args = DotDict({
    "task": "asqp",  # Vorgabe, der Wert wird im Skript gesetzt
    "dataset": "rest15",
    "model_name_or_path": "t5-base",
    "do_train": True,
    "do_eval": False,  # Standardwert, wenn nicht angegeben
    "do_direct_eval": True,
    "do_inference": False,  # Standardwert, wenn nicht angegeben
    "max_seq_length": 200,
    "n_gpu": 0,
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-4,
    "num_train_epochs": 20,  # Dein Wert, Standardwert war 30
    "seed": 42,
    "weight_decay": 0.0,
    "adam_epsilon": 1e-8,
    "warmup_steps": 0.0,
    "output_dir": "classifier/outputs",  # Hinzugefügt, weil es in deinem Skript verwendet wird
    "train_ds": None,
    "test_ds": None,
    "top_k": 5,
    })

    # Überschreiben der Standardwerte mit den übergebenen Argumenten
    for key, value in kwargs.items():
        if key in default_args:
            default_args[key] = value
        else:
            raise ValueError(f"Unknown argument: {key}")

    dataloader = DataLoader()

    if default_args["train_ds"] is None:
       train_ds = dataloader.load_data(name=default_args['dataset'], data_type="train", target=default_args["task"])
    else:
       train_ds = default_args["train_ds"]
       
    if default_args["test_ds"] is None:
       test_ds = dataloader.load_data(name=default_args['dataset'], data_type="test", target=default_args["task"])
    else:
       test_ds = default_args["test_ds"]
    
    print("Train Dataset length:", len(train_ds))
    print("Test Dataset length:", len(test_ds))

    # Update train_ds und test_ds in den Argumenten
    default_args["train_ds"] = train_ds
    default_args["test_ds"] = test_ds

    # Training ausführen
    scores = train_function_ilo(default_args)

    # Outputs-Verzeichnis löschen
    shutil.rmtree(default_args["output_dir"])
    os.makedirs(default_args["output_dir"], exist_ok=True)

    return scores

# scores = train_ilo(task="tasd", dataset="rest16")
# print("SCORES:", scores)


def train_llm(**kwargs):
    from classifier.llm.classifier import train_function_llm
    # Standardwerte
    default_args = DotDict({
    "task": "asqp", 
    "dataset": "rest15",
    "model_name_or_path": "gemma-3-4b-it",
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "learning_rate": 2e-4,
    "num_train_epochs": 10, 
    "seed": 42,
    "train_ds": None,
    "test_ds": None,
    "unique_aspect_categories": None,
    "language": "en",
    "max_seq_length": 1024,
    "max_new_tokens": 256,
    "max_num_regenerations_eval": 10,
    "temperature": 0.8,
    "output_dir": "classifier/outputs", 
    })

    # Überschreiben der Standardwerte mit den übergebenen Argumenten
    for key, value in kwargs.items():
        if key in default_args:
            default_args[key] = value
        else:
            raise ValueError(f"Unknown argument: {key}")

    dataloader = DataLoader()

    if default_args["train_ds"] is None:
       train_ds = dataloader.load_data(name=default_args['dataset'], data_type="train", target=default_args["task"])
    else:
       train_ds = default_args["train_ds"]
       
    if default_args["test_ds"] is None:
       test_ds = dataloader.load_data(name=default_args['dataset'], data_type="test", target=default_args["task"])
    else:
       test_ds = default_args["test_ds"]
    
    print("Train Dataset length:", len(train_ds))
    print("Test Dataset length:", len(test_ds))

    # Update train_ds und test_ds in den Argumenten
    default_args["train_ds"] = train_ds
    default_args["test_ds"] = test_ds

    # Training ausführen
    scores = train_function_llm(default_args)

    # Outputs-Verzeichnis löschen
    shutil.rmtree(default_args["output_dir"])
    os.makedirs(default_args["output_dir"], exist_ok=True)

    return scores
