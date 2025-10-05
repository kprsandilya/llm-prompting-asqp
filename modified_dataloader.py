import os
import ast
import random
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import json
import sys, os
from helper import merge_aspect_lists

class ModifiedDataLoader:
    def __init__(self, base_path="datasets", fs_path="fs_examples"):
        self.base_path = base_path
        self.fs_path = fs_path

    def load_fs_ann(self, name, data_type, target, fs_num, llm_name):
        
        data = []
        for seed in range(5):
          # print current directory
          print("Current working directory:", os.getcwd())
          with open(f"./_out_synthetic_examples/01_llm_annotate_train/{target}_{name}_train_{llm_name}_{seed}_{fs_num}.json", "r", encoding="utf-8") as file:
            examples = json.load(file)
          data.append(examples)
        
        lines = []
        for k in range(len(data[0])):
            labels = []

            for i in range(0, len(data)):
                if len(data[i][k]["pred_label"]) > 0:
                   labels += [data[i][k]["pred_label"]]
                else:
                   labels.append([])
            merged_label = merge_aspect_lists(labels, minimum_appearance=3)
            lines.append(f"{data[0][k]['text']}####{merged_label}")
                 
        return lines
    
    def load_aug_ann(self, name, target, fs_num, aug_method):
        lines = []
        idx_aug = {"eda": 2, "llm_eda": 3, "back_translation": 4}
        with open(f"./_out_synthetic_examples/0{str(idx_aug[aug_method])}_{aug_method}_few_shot_augmenter/{target}_{name}_{fs_num}.txt", "r", encoding="utf-8") as file:
            for line in file:
                lines.append(line.strip())  # Entfernt Zeilenumbrüche
        
        lines_sorted = []
                        
        if aug_method == "eda" or aug_method == "llm_eda":
            num_aug_for_example = int(len(lines) / fs_num)
        if aug_method == "back_translation":
            num_aug_for_example = 5
        for j in range(num_aug_for_example): # anzahl an annotierten beispielen   
            for i in range(fs_num): # anzahl an beispielen die augmentiert wurden
                lines_sorted.append(lines[j+num_aug_for_example*i])

        lines_sorted = lines_sorted * 10
        return lines_sorted

    def load_data(self, name, data_type, cv=False, seed=42, target="asqp", fs_mode=False, fs_num=0, language="en", balance_train=False, balance_test=False, aug_strategy=None, fs_ann_mode=False, llm_name="gemma3:27b", n_ann_examples="full", aug_mode=False, aug_method=None):
        if fs_mode or fs_ann_mode or aug_mode:
            dataset_paths = [os.path.join(self.fs_path, target, name, f"fs_{str(fs_num)}", "examples.txt")] 
        elif aug_strategy=="code-switching":
            dataset_paths = [self.base_path +"/" + target + "_" + name + "_" + language + "_train_cs" + ".txt"]
        elif aug_strategy=="translation":
            if balance_train:
                bal_str = "_b"
            else:
                bal_str = ""
            dataset_paths = [self.base_path +"/" + target + "_" + name + "_" + language + "_train" + bal_str + ".txt"]
        elif name == "multilingual-rest":
             dataset_paths = ["train", "test", "dev"] if data_type == "all" else [data_type]
             if balance_train or balance_test:
                dataset_paths = [os.path.join(self.base_path, target, name, language, f"{d_path}_b.txt") for d_path in dataset_paths]
             else:
                dataset_paths = [os.path.join(self.base_path, target, name, language, f"{d_path}.txt") for d_path in dataset_paths]
        else:
            dataset_paths = ["train", "test", "dev"] if data_type == "all" else [data_type]
            dataset_paths = [os.path.join(self.base_path, target, name, f"{d_path}.txt") for d_path in dataset_paths]

        if fs_ann_mode:
            dataset_paths += [f"./_out_synthetic_examples/01_llm_annotate_train/"]
        
        if aug_mode:
            dataset_paths += [f"./_out_synthetic_examples/02_{aug_method}_few_shot_augmenter/"]

        data = []

        for d_path in dataset_paths:
            if not os.path.exists(d_path):
                raise FileNotFoundError(f"Dataset file {d_path} not found.")
            
            lines = []
            
            if "_out_synthetic_examples/01_llm_annotate_train" in d_path:
                    ann_examples = self.load_fs_ann(name, data_type, target, fs_num, llm_name)
                    lines += ann_examples
                    if n_ann_examples != "full":
                        lines = lines[0:n_ann_examples-fs_num]       
            elif f"_out_synthetic_examples/02_{aug_method}_few_shot_augmenter" in d_path:
                    ann_examples = self.load_aug_ann(name, target, fs_num, aug_method)
                    lines += ann_examples
                    if n_ann_examples != "full":
                        lines = lines[0:fs_num*n_ann_examples]                   
            else:
                    with open(d_path, 'r', encoding='utf-8') as file:
                       lines += file.readlines()
            
            for idx, line in enumerate(lines):
                try:
                    text, aspects_str = line.split("####")
                    aspects = ast.literal_eval(aspects_str.strip())
                    aspect_list = []

                    for aspect in aspects:
                        aspect_dict = {
                            "aspect_term": aspect[0],
                            "aspect_category": aspect[1],
                            "polarity": aspect[2]
                        }
                        # Add 'opinion_term' only if target is 'asqp'
                        if target == "asqp":
                            aspect_dict["opinion_term"] = aspect[3]
                        aspect_list.append(aspect_dict)

                    if len(aspects) > 0:
                        data.append({
                            "id": f"{idx}_{name}_{d_path}",
                            "text": text.strip(),
                            "aspects": aspect_list,
                            "tuple_list": [tuple(aspect) for aspect in aspects]
                        })
                except ValueError as e:
                    aspect_list = []
                    aspect_dict = {
                        "aspect_term": "Placeholder",
                        "aspect_category": "Placeholder",
                        "polarity": "Placeholder"
                    }
                    aspects = []
                    # Add 'opinion_term' only if target is 'asqp'
                    if target == "asqp":
                        aspect_dict["opinion_term"] = "Placeholder"
                    aspect_list.append(aspect_dict)
                    data.append({
                            "id": f"{idx}_{name}_{d_path}",
                            "text": line.strip(),
                            "aspects": aspect_list,
                            "tuple_list": ["Placeholder", "Placeholder", "Placeholder", "Placeholder"]
                        })
                    print(f"No Aspects or Sentiments on Line")
                    continue
        
        if cv:
            return self.random_cross_validation_split(data, seed)
        
        
        return data


    def random_cross_validation_split(self, data, seed=42):
        categories = [[el["aspect_category"] for el in an["aspects"]] for an in data]

        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(categories)

        n_splits = 5
        mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
        splits = []
        for train_index, test_index in mskf.split(np.zeros(len(Y)), Y):
           splits.append([data[i] for i in test_index])
    
    
        return splits