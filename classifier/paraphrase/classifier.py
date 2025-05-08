import os
import time
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
import numpy as np
import random


sentiment_word_list = ['positive', 'negative', 'neutral']

opinion2word = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}
opinion2word_german = {'gut': 'positive', 'schlecht': 'negative', 'ok': 'neutral'}


dataset_name = None

def extract_spans_para(task, seq, seq_type):
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
    if task == 'aste':
        for s in sents:
            # It is bad because editing is problem.
            try:
                c, ab = s.split(' because ')
                c = opinion2word.get(c[6:], 'nope')    # 'good' -> 'positive'
                a, b = ab.split(' is ')
            except ValueError:
                # print(f'In {seq_type} seq, cannot decode: {s}')
                a, b, c = '', '', ''
            quads.append((a, b, c))
    elif task == 'tasd':
        for s in sents:
            # food quality is bad because pizza is bad.
            try:
                ac_sp, at_sp = s.split(' because ')
                
                ac, sp = ac_sp.split(' is ')
                at, sp2 = at_sp.split(' is ')
                
                sp = opinion2word.get(sp, 'nope')
                sp2 = opinion2word.get(sp2, 'nope')
                if sp != sp2:
                    print(f'Sentiment polairty of AC({sp}) and AT({sp2}) is inconsistent!')
                
                # if the aspect term is implicit
                if at.lower() == 'it':
                    at = 'NULL'
            except ValueError:
                # print(f'In {seq_type} seq, cannot decode: {s}')
                ac, at, sp = '', '', ''
            
            quads.append((ac, at, sp))
    elif task == 'asqp':
        for s in sents:
            # food quality is bad because pizza is over cooked.
            try:
                ac_sp, at_ot = s.split(' because ')
                ac, sp = ac_sp.split(' is ')
                at, ot = at_ot.split(' is ')

                # if the aspect term is implicit
                if at.lower() == 'it':
                    at = 'NULL'
            except ValueError:
                try:
                    # print(f'In {seq_type} seq, cannot decode: {s}')
                    pass
                except UnicodeEncodeError:
                    # print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
                ac, at, sp, ot = '', '', '', ''

            quads.append((ac, at, sp, ot))
    else:
        raise NotImplementedError
    return quads

def extract_spans_para_german(task, seq, seq_type):
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
    if task == 'aste':
        for s in sents:
            # It is bad because editing is problem.
            try:
                c, ab = s.split(' weil ')
                c = opinion2word_german.get(c[6:], 'nope')    # !!!! BEACHTEN!
                a, b = ab.split(' ist ')
            except ValueError:
                # print(f'In {seq_type} seq, cannot decode: {s}')
                a, b, c = '', '', ''
            quads.append((a, b, c))
    elif task == 'tasd':
        for s in sents:
            # food quality is bad because pizza is bad.
            try:
                ac_sp, at_sp = s.split(' weil ')
                
                ac, sp = ac_sp.split(' ist ')
                at, sp2 = at_sp.split(' ist ')
                
                sp = opinion2word_german.get(sp, 'nope')
                sp2 = opinion2word_german.get(sp2, 'nope')
                if sp != sp2:
                    print(f'Sentiment polairty of AC({sp}) and AT({sp2}) is inconsistent!')
                
                # if the aspect term is implicit
                if at.lower() == 'es':
                    at = 'NULL'
            except ValueError:
                # print(f'In {seq_type} seq, cannot decode: {s}')
                ac, at, sp = '', '', ''
            
            quads.append((ac, at, sp))
    elif task == 'asqp':
        for s in sents:
            # food quality is bad because pizza is over cooked.
            try:
                ac_sp, at_ot = s.split(' weil ')
                ac, sp = ac_sp.split(' ist ')
                at, ot = at_ot.split(' ist ')

                # if the aspect term is implicit
                if at.lower() == 'es':
                    at = 'NULL'
            except ValueError:
                try:
                    # print(f'In {seq_type} seq, cannot decode: {s}')
                    pass
                except UnicodeEncodeError:
                    # print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
                ac, at, sp, ot = '', '', '', ''

            quads.append((ac, at, sp, ot))
    else:
        raise NotImplementedError
    return quads


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
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores


def compute_scores(pred_seqs, gold_seqs, sents):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        if dataset_name in german_datasets:
            gold_list = extract_spans_para_german('asqp', gold_seqs[i], 'gold')
            pred_list = extract_spans_para_german('asqp', pred_seqs[i], 'pred')
        else:
            gold_list = extract_spans_para('asqp', gold_seqs[i], 'gold')
            pred_list = extract_spans_para('asqp', pred_seqs[i], 'pred')

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    print("\nResults:")
    scores = compute_f1_scores(all_preds, all_labels)

    return scores, all_labels, all_preds

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
senttag2opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}

senttag2word_german = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
senttag2opinion_german = {'POS': 'gut', 'NEG': 'schlecht', 'NEU': 'ok'}
sentword2opinion_german = {'positive': 'gut', 'negative': 'schlecht', 'neutral': 'ok'}


def get_para_aste_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        all_tri_sentences = []
        for tri in label:
            # a is an aspect term
            if len(tri[0]) == 1:
                a = sents[i][tri[0][0]]
            else:
                start_idx, end_idx = tri[0][0], tri[0][-1]
                a = ' '.join(sents[i][start_idx:end_idx+1])

            # b is an opinion term
            if len(tri[1]) == 1:
                b = sents[i][tri[1][0]]
            else:
                start_idx, end_idx = tri[1][0], tri[1][-1]
                b = ' '.join(sents[i][start_idx:end_idx+1])

            # c is the sentiment polarity
            if dataset_name in german_datasets:
                c = senttag2opinion_german[tri[2]]
            else:
                c = senttag2opinion[tri[2]]          
                
            one_tri = f"It is {c} because {a} is {b}"
            all_tri_sentences.append(one_tri)
        targets.append(' [SSEP] '.join(all_tri_sentences))
    return targets

german_datasets = ["gerest"]

def get_para_tasd_targets(sents, labels):
    targets = []
    for label in labels:
        all_tri_sentences = []
        for triplet in label:
            at, ac, sp = triplet
            
            if dataset_name in german_datasets:
                man_ot = sentword2opinion_german[sp]  
            else:
                man_ot = sentword2opinion[sp]  

            if at == 'NULL':
                if dataset_name in german_datasets:
                    at = 'es'
                else: 
                    at = 'it'
                
            if dataset_name in german_datasets:
                one_tri = f"{ac} ist {man_ot} weil {at} ist {man_ot}"
            else:
                one_tri = f"{ac} is {man_ot} because {at} is {man_ot}"
            all_tri_sentences.append(one_tri)

        target = ' [SSEP] '.join(all_tri_sentences)
        targets.append(target)
    return targets


def get_para_asqp_targets(sents, labels):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    for label in labels:
        all_quad_sentences = []
        for quad in label:
            at, ac, sp, ot = quad

            if dataset_name in german_datasets:
                man_ot = sentword2opinion_german[sp]  
            else:
                man_ot = sentword2opinion[sp]  

            if at == 'NULL':  # for implicit aspect term
                if dataset_name in german_datasets:
                    at = 'es'
                else:
                    at = 'it'

            if dataset_name in german_datasets:
                one_quad_sentence = f"{ac} ist {man_ot} weil {at} ist {ot}"
            else:
                one_quad_sentence = f"{ac} is {man_ot} because {at} is {ot}"
            all_quad_sentences.append(one_quad_sentence)

        target = ' [SSEP] '.join(all_quad_sentences)
        targets.append(target)
    return targets

def read_line_examples_from_file(data_type, args):
    if data_type == "train":
        sents = [example["text"].split() for example in args["train_ds"]]
        labels = [list(example["tuple_list"]) for example in args["train_ds"]]
    elif data_type == "test":
        sents = [example["text"].split() for example in args["test_ds"]]
        labels = [list(example["tuple_list"]) for example in args["test_ds"]]        

    return sents, labels


def get_transformed_io(data_type, args):
    """
    The main function to transform input & target according to the task
    """
    sents, labels = read_line_examples_from_file(data_type, args)

    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]

    if args["task"] == 'aste':
        targets = get_para_aste_targets(sents, labels)
    elif args["task"] == 'tasd':
        targets = get_para_tasd_targets(sents, labels)
    elif args["task"] == 'asqp':
        targets = get_para_asqp_targets(sents, labels)
    else:
        raise NotImplementedError

    return inputs, targets


class ABSADataset(Dataset):
    def __init__(self, tokenizer, args, data_type, max_len=128):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.args = args
        self.data_type = data_type

        self.inputs = []
        self.targets = []

        self._build_examples()
        print("data type:", data_type)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _build_examples(self):
        inputs, targets = get_transformed_io(self.data_type, self.args)

        for i in range(len(inputs)):
            input = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
                [input], max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)


def get_dataset(tokenizer, type_path, args):
    return ABSADataset(tokenizer, args,
                       type_path, args.max_seq_length)


class T5FineTuner(nn.Module):
    def __init__(self, hparams, model, tokenizer):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )


def train_model(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        lm_labels = batch["target_ids"].to(device)
        lm_labels[lm_labels[:, :] == model.tokenizer.pad_token_id] = -100

        outputs = model(
            input_ids=batch["source_ids"].to(device),
            attention_mask=batch["source_mask"].to(device),
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'].to(device)
        )

        loss = outputs[0]
        loss.backward()

        total_loss += loss.item()
        optimizer.step()
        scheduler.step()

    return total_loss / len(train_loader)


def validate_model(model, val_loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            lm_labels = batch["target_ids"].to(device)
            lm_labels[lm_labels[:, :] == model.tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=batch["source_ids"].to(device),
                attention_mask=batch["source_mask"].to(device),
                labels=lm_labels,
                decoder_attention_mask=batch['target_mask'].to(device)
            )

            loss = outputs[0]
            total_loss += loss.item()

    return total_loss / len(val_loader)


def evaluate(data_loader, model, sents, device):
    model.eval()
    outputs, targets = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            outs = model.model.generate(input_ids=batch['source_ids'].to(device), 
                                        attention_mask=batch['source_mask'].to(device), 
                                        max_length=128)

            dec = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            target = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

            outputs.extend(dec)
            targets.extend(target)

    scores, all_labels, all_preds = compute_scores(outputs, targets, sents)
    print("Scores: ", scores)
    scores["all_labels"] = all_labels
    scores["all_preds"] = all_preds

    return scores


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
    print(f"Random seed set as {seed}")
    
    

def train_function_paraphrase(args):
    set_seed(args.seed)
    
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    global dataset_name
    dataset_name = args.dataset

    if args.do_train:
        print("Enter training phase!")
        model = T5FineTuner(args, T5ForConditionalGeneration.from_pretrained(args.model_name_or_path), tokenizer)
        model = model.to(device)

        train_dataset = get_dataset(tokenizer, "train", args)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                  drop_last=True, shuffle=True, num_workers=4)

        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = (len(train_loader.dataset) // (args.train_batch_size * max(1, args.n_gpu))) // args.gradient_accumulation_steps * float(args.num_train_epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        for epoch in range(int(args.num_train_epochs)):
            train_loss = train_model(model, train_loader, optimizer, scheduler, device)
            print(f"Epoch {epoch+1}/{args.num_train_epochs}, Training Loss: {train_loss:.4f}")
            
            if args.do_eval:
              val_dataset = get_dataset(tokenizer, "test", args)
              val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, num_workers=4)
              val_loss = validate_model(model, val_loader, device)
              print(f"Epoch {epoch+1}/{args.num_train_epochs}, Validation Loss: {val_loss:.4f}")

        #model_path = os.path.join(args.output_dir, "model.bin")
        #torch.save(model.state_dict(), model_path)
        #print("Model saved!")

    if args.do_direct_eval:
        #model = T5FineTuner(args, T5ForConditionalGeneration.from_pretrained(args.model_name_or_path), tokenizer)
        #model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.bin")))
        #model = model.to(device)

        sents, _ = read_line_examples_from_file("test", args)

        test_dataset = ABSADataset(tokenizer, args, 'test', args.max_seq_length)
        test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

        scores = evaluate(test_loader, model, sents, device)

        log_file_path = f"results_log/{args.dataset}.txt"
        local_time = time.asctime(time.localtime(time.time()))

        exp_settings = f"Dataset={args.dataset}; Train bs={args.train_batch_size}, num_epochs={args.num_train_epochs}"
        exp_results = f"F1 = {scores['f1']:.4f}"

        log_str = f'============================================================\n'
        log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

    return scores
