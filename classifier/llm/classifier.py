import os, re, sys, json, random, torch, time


import numpy as np
import torch
from dataloader import DataLoader
from datasets import Dataset

from transformers import TrainingArguments
from unsloth import is_bfloat16_supported, FastModel

from trl import SFTTrainer

from validator import validate_label, to_pred_list


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision != 0 or recall != 0
        else 0
    )
    scores = {"precision": precision, "recall": recall, "f1": f1}

    return scores


def compute_scores(pred_seqs, gold_seqs, task):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):

        gold_list = extract_spans_para(task, gold_seqs[i], "gold")
        pred_list = extract_spans_para(task, pred_seqs[i], "pred")

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    scores = compute_f1_scores(all_preds, all_labels)

    return scores, all_labels, all_preds


dataloader = DataLoader("./datasets")


def get_prompt_header(task, unique_aspect_categories, language):
    if language == "en":
        file_path = f"./prompt/{task}/prompt.txt"
    else:
        file_path = f"./prompt/{task}/prompt_{language}.txt"

    # load txt file
    with open(file_path, "r") as f:
        prompt_header = f.read()

    return prompt_header


with open("./prompt/alpaca_prompt_base.txt", "r") as f:
    alpaca_prompt = f.read()


def get_model_and_tokenizer(max_seq_length, model_name_or_path, seed):
    dtype = None
    load_in_4bit = True
    load_in_8bit = False
    model, tokenizer = FastModel.from_pretrained(
        model_name=f"unsloth/{model_name_or_path}",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Turn off for just text!
        finetune_language_layers=True,  # Should leave on!
        finetune_attention_modules=True,  # Attention good for GRPO
        finetune_mlp_modules=True,  # SHould leave on always!
        r=8,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=8,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=seed,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    return model, tokenizer


def get_trainer(model, tokenizer, dataset, args):
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args["max_seq_length"],
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=args["train_batch_size"],
            warmup_steps=0,
            num_train_epochs=args["num_train_epochs"],
            learning_rate=args["learning_rate"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args["seed"],
            output_dir="outputs",
            report_to="none",  # Use this for WandB etc
        ),
    )



def train_function_llm(args):
    train_ds = args["train_ds"]
    test_ds = args["test_ds"]
    seed = args["seed"]
    model_name_or_path = args["model_name_or_path"]
    ds_name = args["dataset"]
    task = args["task"]
    unique_aspect_categories = args["unique_aspect_categories"]
    lanugage = args["language"]
    max_seq_length = args["max_seq_length"]
    max_new_tokens = args["max_new_tokens"]
    max_num_regenerations_eval = args["max_num_regenerations_eval"]
    temperature = args["temperature"]

    set_seed(seed)

    if unique_aspect_categories == None:
        unique_aspect_categories = sorted(
            {
                aspect["aspect_category"]
                for entry in dataloader.load_data(
                    name=ds_name, data_type="all", target=task
                )
                for aspect in entry["aspects"]
            }
        )
    else:
        unique_aspect_categories = sorted(unique_aspect_categories)

    prompt_header = (
        get_prompt_header(task, unique_aspect_categories, lanugage)
        .replace("[[aspect_category]]", str(unique_aspect_categories)[1:-1])
        .replace("[[examples]]", "")[:-2]
    )

    dataset = []
    for idx, example in enumerate(train_ds):
        dataset.append(
            {
                "input": example["text"],
                "output": str(example["tuple_list"]),
            }
        )
    dataset = Dataset.from_dict({key: [d[key] for d in dataset] for key in dataset[0]})
    model, tokenizer = get_model_and_tokenizer(max_seq_length, model_name_or_path, seed)
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for input, output in zip(inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(prompt_header, input, output) + EOS_TOKEN
            texts.append(text)
        return {
            "text": texts,
        }

    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )
    
    start_total_time = time.time()

    trainer = get_trainer(model, tokenizer, dataset, args)
    trainer.train()
    
    end_total_time = time.time()
    total_time = end_total_time - start_total_time
    

    FastModel.for_inference(model)

    all_preds = []
    all_labels = []
    all_raw_preds = []

    for idx, example in enumerate(test_ds):
        test_text = example["text"]
        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    prompt_header,
                    test_text,
                    "",
                )
            ],
            return_tensors="pt",
        ).to("cuda")

        tuple_list = [list(_tuple) for _tuple in example["tuple_list"]]
        tuple_list = [[_tuple[1], _tuple[0], *_tuple[2:]] for _tuple in tuple_list]
        all_labels.append(tuple_list)

        output_label, output_raw = evaluate_llm(
            tokenizer,
            model,
            inputs,
            max_new_tokens,
            temperature,
            task,
            max_num_regenerations_eval,
            unique_aspect_categories,
            test_text
        )
        
        all_preds.append(output_label)
        all_raw_preds.append(output_raw)
        

        print(f"{idx}/{len(test_ds)} ### Pred:", output_label, "\nGold:", tuple_list)

    predictions = compute_f1_scores(all_preds, all_labels)
    predictions["all_preds"] = (all_preds,)
    predictions["all_labels"] = all_labels
    predictions["all_raw_preds"] = all_raw_preds
    predictions["total_time"] = total_time

    return predictions


def extract_output_as_list(output_raw, task):
    try:
        output_label = output_raw.split("### Label:")[1]
        output_label = output_label.split("]")[0] + "]"

        n_elements = {"asqp": 4, "tasd": 3, "e2e": 2}
        output_label = to_pred_list(output_label, n_elements[task])
        output_label = [
            _tuple for _tuple in output_label if len(_tuple) == n_elements[task]
        ]
    except:
        output_label = []
    return output_label


def evaluate_llm(
    tokenizer,
    model,
    inputs,
    max_new_tokens,
    temperature,
    task,
    max_num_regenerations_eval,
    unique_aspect_categories,
    test_text
):

    n_try = 0
    found_valid_output = False
    while n_try < max_num_regenerations_eval and not found_valid_output:
        output_raw = tokenizer.batch_decode(
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=False,
                do_sample=True,
                temperature=temperature,
            ),
        )[0]
        # filter label from predicted string
        output_label = extract_output_as_list(output_raw, task)
        if (len(output_label) == 0):
            n_try += 1
        else:
           output_validated = validate_label([tuple(t) for t in output_label], test_text, unique_aspect_categories, task=task, is_string=False, allow_small_variations=True)
           if len(output_validated) > 1:
               n_try += 1
           else:
               output_label = output_validated[0]
               output_label = [[_tuple[1], _tuple[0], *_tuple[2:]] for _tuple in output_label]
               found_valid_output = True
               
    if n_try == max_num_regenerations_eval:
        output_label = [[_tuple[1], _tuple[0], *_tuple[2:]] for _tuple in output_label]
    

    return output_label, output_raw
