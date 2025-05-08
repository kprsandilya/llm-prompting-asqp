import os, json, shutil
from send2trash import send2trash
from validator import validate_label
from collections import Counter

class DotDict(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        try:
            del self[attr]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")
        
    

def clean_up(output_directory='./classifier/outputs'):
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
        print(f'Directory {output_directory} has been deleted.')
    else:
        print(f'Directory {output_directory} does not exist.')

    example_file_path = 'example.txt'  # Beispiel-Datei, die existieren muss

    if os.path.exists(example_file_path):
        send2trash(example_file_path)
        print(f'File {example_file_path} has been moved to trash.')
    else:
        print(f'File {example_file_path} does not exist; nothing to move to trash.')



def create_output_directory(output_directory='./classifier/outputs'):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f'Directory {output_directory} has been created.')
    else:
        print(f'Directory {output_directory} already exists.')


#load json file
# with open('./generations/zeroshot/tasd_rest16_test_gemma3:4b_0_label_0.json', 'r') as f:
#     data = json.load(f)
    
# rest_aspect_cate_list = [
#     'location general', 'food prices', 'food quality', 'food general',
#     'ambience general', 'service general', 'restaurant prices',
#     'drinks prices', 'restaurant miscellaneous', 'drinks quality',
#     'drinks style_options', 'restaurant general', 'food style_options'
# ]

def do_additional_checks(data, task, unique_aspect_categories):
    novel_data = []
    for example in data:
        if len(example["invalid_precitions_label"]) > 0:
            invalid_precitions_label_new = []
            for invalid_prediction in example["invalid_precitions_label"]:
                raw_pred = invalid_prediction["pred_label_raw"]
                validation = validate_label(raw_pred, example["text"], unique_aspect_categories, task=task, allow_small_variations=True)
                if len(validation) == 1:
                    example["pred_label"] = validation[0]
                    break
                else:
                    invalid_precitions_label_new.append(invalid_prediction) 
            example["invalid_precitions_label"] = invalid_precitions_label_new          
            novel_data.append(example)
        else:
            novel_data.append(example)
    return novel_data

# do_additional_checks(data, "tasd", rest_aspect_cate_list)


def get_frequency_for_counts(counts, minimum):
    return sorted(counts, reverse=True)[0:minimum][minimum-1]

def get_unique_keys(dict_list):
    unique_keys = set()  # Set für einzigartige Schlüssel

    for d in dict_list:
        unique_keys.update(d.keys())  # Füge die Schlüssel zum Set hinzu

    return list(unique_keys)  # Wandle das Set in eine Liste um und gebe es zurück


def merge_aspect_lists(aspect_lists, minimum_appearance=3):
    
    aspect_lists_counter = []
    for aspect_list in aspect_lists:
        aspect_counter = dict(Counter(["#####".join(aspect) for aspect in aspect_list]))
        aspect_lists_counter.append(aspect_counter)
        
    unique_tuples = get_unique_keys(aspect_lists_counter)

    label = []
    for tuple_str in unique_tuples:

        count_tuple =  get_frequency_for_counts([asp.get(tuple_str, 0) for asp in aspect_lists_counter], minimum_appearance)
        tuple_reverse = tuple(tuple_str.split("#####"))
        
        label += count_tuple * [tuple_reverse]

    return label