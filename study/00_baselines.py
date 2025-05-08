import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloader import DataLoader
from trainer import train_paraphrase, train_mvp, train_dlo, train_llm
import json
import time

from helper import clean_up, create_output_directory

dataloader = DataLoader("./datasets", "./fs_examples")

model_name_or_path = "gemma-3-4b-it"

for i in range(5):
 for ds_name in ["rest16", "rest15", "hotels", "flightabsa", "coursera"]:
   for task in ["asqp", "tasd"]:
         train_ds = dataloader.load_data(ds_name, "train", cv=False, target=task)
         test_ds = dataloader.load_data(ds_name, "test", cv=False, target=task)
      
         for ml_method in [f"llm_{model_name_or_path}", "paraphrase", "mvp", "dlo"]:
            print(f"Task:", task, "Dataset:", ds_name, "Seed:", i, "ML-Method:", ml_method)
            filename = f"./generations/00_baselines/training_{task}_{ds_name}_seed-{i}_n-train_{ml_method}.json"

            if os.path.exists(filename):
               print(f"File {filename} already exists. Skipping.")
               continue
            else:
            
               clean_up()
               create_output_directory()
              
               if ml_method == f"llm_{model_name_or_path}":
                  scores = train_llm(train_ds=train_ds, test_ds=test_ds, seed=i, dataset=ds_name, task=task, model_name_or_path=model_name_or_path)
               if ml_method == "paraphrase":
                  scores = train_paraphrase(train_ds=train_ds, test_ds=test_ds, seed=i, dataset=ds_name, task=task)
               if ml_method == "mvp":
                  scores = train_mvp(train_ds=train_ds, test_ds=test_ds, seed=i, dataset=ds_name, task=task)
               if ml_method == "dlo":
                  scores = train_dlo(train_ds=train_ds, test_ds=test_ds, seed=i, dataset=ds_name, task=task)
              
    
               with open(filename, 'w', encoding='utf-8') as json_file:
                  json.dump(scores, json_file, ensure_ascii=False, indent=4)
                  
               

for i in range(5):
    for ds_name in ["rest16", "rest15", "hotels", "flightabsa", "coursera"]:
        for task in ["asqp", "tasd"]:
            for fs_num in [10, 20, 30, 40, 50, 800]:
                if fs_num > 16:
                    bs = 16
                else:
                    bs = 8
                train_ds = dataloader.load_data(ds_name, "train", cv=False, target=task, fs_mode=True, fs_num=fs_num)
                test_ds = dataloader.load_data(ds_name, "test", cv=False, target=task)
      
                for ml_method in [f"llm_{model_name_or_path}", "paraphrase", "mvp", "dlo"]:
                    print(f"Task:", task, "Dataset:", ds_name, "Seed:", i, "ML-Method:", ml_method)
                    filename = f"./generations/00_baselines/training_{task}_{ds_name}_seed-{i}_n-train_{ml_method}_{fs_num}.json"

                    # Ensure directory exists
                    os.makedirs(os.path.dirname(filename), exist_ok=True)

                    if os.path.exists(filename):
                        print(f"File {filename} already exists. Skipping.")
                        continue
                    else:
                        clean_up()
                        create_output_directory()
                        
                        if ml_method == f"llm_{model_name_or_path}":
                            scores = train_llm(train_ds=train_ds, test_ds=test_ds, seed=i, dataset=ds_name, task=task, train_batch_size=bs, model_name_or_path=model_name_or_path)
                        if ml_method == "paraphrase":
                            scores = train_paraphrase(train_ds=train_ds, test_ds=test_ds, seed=i, dataset=ds_name, task=task, train_batch_size=bs)
                        if ml_method == "mvp":
                            scores = train_mvp(train_ds=train_ds, test_ds=test_ds, seed=i, dataset=ds_name, task=task, train_batch_size=bs)
                        if ml_method == "dlo":
                            scores = train_dlo(train_ds=train_ds, test_ds=test_ds, seed=i, dataset=ds_name, task=task, train_batch_size=bs)

                        # Write results to the file
                        with open(filename, 'w', encoding='utf-8') as json_file:
                            json.dump(scores, json_file, ensure_ascii=False, indent=4)
                        time.sleep(20)