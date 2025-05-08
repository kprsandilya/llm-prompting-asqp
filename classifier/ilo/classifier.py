import argparse
import os
import sys
sys.path.append('classifier/ilo')
import logging
import pickle
import random
import json
import time
import re
from itertools import permutations
from functools import partial
from collections import Counter
from torch.utils.data import Dataset

import numpy as np
import torch

from transformers.models.t5.modeling_t5 import *
from transformers import AdamW, T5Tokenizer
from t5_score import MyT5ForConditionalGenerationScore
from t5 import MyT5ForConditionalGeneration

from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup
import pytorch_lightning as pl

from tqdm import tqdm


sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}
sentword2opinion_german = {'positive': 'gut', 'negative': 'schlecht', 'neutral': 'ok'}



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
    
class ABSADataset(Dataset):

    def __init__(self,
                 tokenizer,
                 task,
                 data_type,
                 top_k,
                 args,
                 dataset,
                 max_len=128,
                ):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.task = task
        self.data_type = data_type
        self.args = args
        self.dataset = dataset

        self.top_k = top_k

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze(
        )  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze(
        )  # might need to squeeze
        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask
        }

    def _build_examples(self):
        

        inputs, targets = get_transformed_io(self.task, self.data_type, self.top_k,
                                                 self.args, self.dataset)
        for i in range(len(inputs)):
            # change input and target to two strings
            input = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
                [input],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt")
            
            # for ACOS Restaurant and Laptop dataset
            # the max target length is much longer than 200
            # we need to set a larger max length for inference
            target_max_length = 1024 if self.data_type == "test" else self.max_len

            tokenized_target = self.tokenizer.batch_encode_plus(
                [target],
                max_length=target_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt")

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)

         

def get_para_tasd_targets(sents, labels, top_k, language="english"):
    """
    Erzeugt Zielsequenzen für den TASD-Task im Paraphrase-Paradigma.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device("cpu")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = MyT5ForConditionalGenerationScore.from_pretrained("t5-base").to(device)

    targets = []
    new_sents = []
    data_count = {}

    for i in range(len(sents)):
        label = labels[i]
        cur_sent = sents[i]

        if len(label) in data_count:
            data_count[len(label)] += 1
        else:
            data_count[len(label)] = 1

        quad_list = []
        for quad in label:
            at, ac, sp = quad

            if language == "german":
                man_sp = sentword2opinion_german[sp]  # Mapping der Sentiment-Polarity
            else:
                man_sp = sentword2opinion[sp]  # z.B. 'POS' -> 'good'

            if at == "NULL":  # Für implizite Aspekte
                at = "es" if language == "german" else "it"

            quad = [f"[AT] {at}", f"[AC] {ac}", f"[SP] {man_sp}"]
            x = permutations(quad)
            permute_object = {}
            for each in x:
                order = []
                content = []
                for e in each:
                    order.append(e[:4])  # Extrahiert das Tag, z.B. "[AT]"
                    content.append(e[4:])  # Extrahiert den Text
                order_name = " ".join(order)
                content = " ".join(content)
                permute_object[order_name] = [content, " ".join(each)]

            quad_list.append(permute_object)

        best_order = choose_best_order_batch(quad_list, cur_sent, model, tokenizer, device, top_k, task="tasd")

        for b_o in best_order:
            targets.append(b_o)
            new_sents.append(cur_sent)

    return new_sents, targets

def get_para_asqp_targets_test(sents, labels):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    for label in labels:
        all_quad_sentences = []
        for quad in label:
            at, ac, sp, ot = quad
            
            if language == 'german':
                man_ot = sentword2opinion_german[sp]  # 'POS' -> 'gut'
            else:
                man_ot = sentword2opinion[sp]  # 'POS' -> 'good'
                

            if at == 'NULL':  # for implicit aspect term
                if language == 'german':
                    at = 'es'
                else:
                    at = 'it'

            quad_list = [f"[AT] {at}", f"[OT] {ot}", f"[AC] {ac}", f"[SP] {man_ot}"]
            one_quad_sentence = " ".join(quad_list)
            all_quad_sentences.append(one_quad_sentence)

        target = ' [SSEP] '.join(all_quad_sentences)
        targets.append(target)
    return targets

def get_para_tasd_targets_test(sents, labels):
    """
    Erzeugt die Ziel-Sätze für den Testmodus im TASD-Task.
    """
    targets = []
    for label in labels:
        all_triplet_sentences = []
        for triplet in label:
            at, ac, sp = triplet

            if language == 'german':
                man_ot = sentword2opinion_german[sp]  # 'POS' -> 'gut'
            else:
                man_ot = sentword2opinion[sp]  # 'POS' -> 'good'

            if at == 'NULL':  # Für implizite Aspekte
                if language == 'german':
                    at = 'es'
                else:
                    at = 'it'

            # Triplet ohne OT (Opinion Term)
            triplet_list = [f"[AT] {at}", f"[AC] {ac}", f"[SP] {man_ot}"]
            one_triplet_sentence = " ".join(triplet_list)
            all_triplet_sentences.append(one_triplet_sentence)

        target = ' [SSEP] '.join(all_triplet_sentences)
        targets.append(target)
    return targets


def order_scores_function(quad_list, cur_sent, model, tokenizer, device, task):
    """
    Berechnet die Scores für verschiedene Reihenfolgen von Quadruples basierend auf dem Task (tasd oder asqp).
    
    Args:
        quad_list: Liste von Quadruple-Daten mit möglichen Reihenfolgen.
        cur_sent: Der aktuelle Satz (Eingabe).
        model: Das Modell zur Berechnung der Scores.
        tokenizer: Tokenizer zum Verarbeiten der Eingaben und Ziele.
        device: Zielgerät (z. B. GPU oder CPU).
        task: Der spezifische Task, entweder 'tasd' oder 'asqp'.
    
    Returns:
        results: Ein Dictionary mit Scores für jede Reihenfolge.
    """
    # Definiere mögliche Reihenfolgen basierend auf dem Task
    if task == "asqp":
        q = ["[AT]", "[OT]", "[AC]", "[SP]"]
    elif task == "tasd":
        q = ["[AT]", "[AC]", "[SP]"]  # Keine OT-Komponente in TASD
    else:
        raise ValueError("Task muss entweder 'tasd' oder 'asqp' sein.")
    
    all_orders = permutations(q)
    all_orders_list = []

    all_targets = []
    all_inputs = []
    cur_sent = " ".join(cur_sent)  # Satz als String zusammenfügen

    for each_order in all_orders:
        cur_order = " ".join(each_order)
        all_orders_list.append(cur_order)
        cur_target = []
        for each_q in quad_list:
            cur_target.append(each_q[cur_order][0])  # Extrahiere die entsprechende Zielsequenz

        all_inputs.append(cur_sent)
        all_targets.append(" ".join(cur_target))

    # Tokenisiere Eingaben und Ziele
    tokenized_input = tokenizer.batch_encode_plus(
        all_inputs, max_length=200, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    tokenized_target = tokenizer.batch_encode_plus(
        all_targets, max_length=200, padding="max_length",
        truncation=True, return_tensors="pt"
    )

    target_ids = tokenized_target["input_ids"].to(device)

    # Maskiere Padding-Tokens
    target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100

    # Modell-Ausgabe
    outputs = model(
        input_ids=tokenized_input["input_ids"].to(device),
        attention_mask=tokenized_input["attention_mask"].to(device),
        labels=target_ids,
        decoder_attention_mask=tokenized_target["attention_mask"].to(device)
    )

    loss, entropy = outputs[0]
    results = {}

    # Ergebnisse speichern
    for i, _ in enumerate(all_orders_list):
        cur_order = all_orders_list[i]
        results[cur_order] = {"loss": loss[i], "entropy": entropy[i]}

    return results


def choose_best_order_batch(quad_list, cur_sent, model, tokenizer, device, top_k, task):
    """
    Wählt die beste Reihenfolge der Quads basierend auf dem Task (tasd oder asqp).
    """
    if task == "tasd":
        q = ["[AT]", "[AC]", "[SP]"]
    elif task == "asqp":
        q = ["[AT]", "[OT]", "[AC]", "[SP]"]
    else:
        raise ValueError("Task must be either 'tasd' or 'asqp'")
    
    all_orders = permutations(q)
    all_orders_list = []

    all_targets = []
    all_inputs = []
    cur_sent = " ".join(cur_sent)
    for each_order in all_orders:
        cur_order = " ".join(each_order)
        all_orders_list.append(cur_order)
        cur_target = []
        for each_q in quad_list:
            cur_target.append(each_q[cur_order][0])

        all_inputs.append(cur_sent)
        all_targets.append(" ".join(cur_target))

    tokenized_input = tokenizer.batch_encode_plus(
        all_inputs, max_length=200, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    tokenized_target = tokenizer.batch_encode_plus(
        all_targets, max_length=200, padding="max_length",
        truncation=True, return_tensors="pt"
    )

    target_ids = tokenized_target["input_ids"].to(device)
    target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100
    outputs = model(
        input_ids=tokenized_input["input_ids"].to(device),
        attention_mask=tokenized_input["attention_mask"].to(device),
        labels=target_ids,
        decoder_attention_mask=tokenized_target["attention_mask"].to(device)
    )

    loss, entropy = outputs[0]

    indexes = np.argsort(np.array(entropy))  # Sortierung nach Entropie
    all_chosen = []
    choosen_indexes = indexes[0: top_k]
    for e in choosen_indexes:
        cur_order = all_orders_list[e]
        best_quad = []
        for each_q in quad_list:
            best_quad.append(each_q[cur_order][1])
        best_quad = " [SSEP] ".join(best_quad)
        all_chosen.append(best_quad)
    
    return all_chosen


def get_para_asqp_targets(sents, labels, top_k):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device("cpu")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")#.to(device)
    #model_org = T5ForConditionalGeneration.from_pretrained("t5-base")
    #torch.save(model_org, "save.pt")
    #model = MyT5ForConditionalGenerationScore.load_state_dict(state_dict=torch.load("save.pt"), strict=True).to(device)
    model = MyT5ForConditionalGenerationScore.from_pretrained("t5-base").to(device)
    targets = []
    new_sents = []
    data_count = {}
    max_length_input = 0
    max_length_target = 0
    

    for i in range(len(sents)):
        label = labels[i]
        cur_sent = sents[i]

        if len(label) in data_count:
            data_count[len(label)] += 1
        else:
            data_count[len(label)] = 1

        all_quad_sentences = []

        already_output = ""

        quad_list = []
        for quad in label:
            at, ac, sp, ot = quad

            if language == 'german':
                man_ot = sentword2opinion_german[sp]  # 'POS
            else:
                man_ot = sentword2opinion[sp]  # 'POS' -> 'good'

            if at == 'NULL':  # for implicit aspect term
                if language == 'german':
                    at = 'es'
                else:
                    at = 'it'

            quad = [f"[AT] {at}",
                    f"[OT] {ot}",
                    f"[AC] {ac}",
                    f"[SP] {man_ot}"]
            x = permutations(quad)
            permute_object = {}
            for each in x:
                order = []
                content = []
                for e in each:
                    order.append(e[0:4])
                    content.append(e[4:])
                order_name = " ".join(order)
                content = " ".join(content)
                permute_object[order_name] = [content, " ".join(each)]

            quad_list.append(permute_object)

        best_order = choose_best_order_batch(quad_list, cur_sent, model, tokenizer, device, top_k, task="asqp")

        for b_o in best_order:
            targets.append(b_o)
            new_sents.append(cur_sent)

    return new_sents, targets



def get_transformed_io(data_name, data_type, top_k, args, dataset):
    """
    Hauptfunktion, um Eingaben und Ziele gemäß der Aufgabe zu transformieren.
    """
    # Extrahiere Texte und Labels aus dem Dataset
    sents = [example["text"].split() for example in dataset]
    labels = [list(example["tuple_list"]) for example in dataset]

    # Eingaben sind die unveränderten Rohsätze
    inputs = [s.copy() for s in sents]

    task = args.task  # Setze den Task explizit auf TASD
    if task == 'tasd':
        if data_type == "test":
            targets = get_para_tasd_targets_test(inputs, labels)  # Nutze die angepasste TASD-Funktion
            return inputs, targets
        else:
            if data_type == "train":
                new_inputs, targets = get_para_tasd_targets(inputs, labels, top_k)
            else:
                targets = get_para_tasd_targets_test(inputs, labels)
                return inputs, targets
    elif task == 'asqp':
        if data_type == "test":
            targets = get_para_asqp_targets_test(inputs, labels)
            return inputs, targets
        else:
            if data_type == "train":
                new_inputs, targets = get_para_asqp_targets(inputs, labels, top_k)
            else:
                targets = get_para_asqp_targets_test(inputs, labels)
                return inputs, targets
            
    else:
        raise NotImplementedError

    return new_inputs, targets


class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """
    def __init__(self, hparams, tfm_model, tokenizer, train_dataset):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.model = tfm_model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, batch_size=self.hparams.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

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

def extract_spans_para(task, seq, seq_type):
    """
    Extrahiert die Komponenten aus der Zielsequenz basierend auf dem Task (tasd oder asqp).

    Args:
        task: Der spezifische Task, entweder 'tasd' oder 'asqp'.
        seq: Die Sequenz, die analysiert werden soll.
        seq_type: Typ der Sequenz (z. B. train, test).

    Returns:
        quads: Eine Liste von Triplets oder Quadruples abhängig vom Task.
    """
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]

    if task == 'tasd':
        for s in sents:
            try:
                # Indexe für die Komponenten finden
                index_ac = s.index("[AC]")
                index_sp = s.index("[SP]")
                index_at = s.index("[AT]")

                combined_list = [index_ac, index_sp, index_at]
                arg_index_list = list(np.argsort(combined_list))  # Reihenfolge sortieren

                result = []
                for i in range(len(combined_list)):
                    start = combined_list[i] + 4
                    sort_index = arg_index_list.index(i)
                    if sort_index < 2:  # Nur die nächsten zwei Elemente überprüfen
                        next_ = arg_index_list[sort_index + 1]
                        re = s[start: combined_list[next_]]
                    else:
                        re = s[start:]
                    result.append(re.strip())

                ac, sp, at = result

                # Wenn der Aspekt-Text implizit ist
                if at.lower() == 'it':
                    at = 'NULL'
            except ValueError:
                try:
                    # Fehlerhafte Sequenzen ignorieren
                    pass
                except UnicodeEncodeError:
                    pass
                ac, at, sp = '', '', ''

            quads.append((ac, at, sp))  # Triplet für TASD speichern
    elif task == 'asqp':
        for s in sents:
            try:
                index_ac = s.index("[AC]")
                index_sp = s.index("[SP]")
                index_at = s.index("[AT]")
                index_ot = s.index("[OT]")

                combined_list = [index_ac, index_sp, index_at, index_ot]
                arg_index_list = list(np.argsort(combined_list))

                result = []
                for i in range(len(combined_list)):
                    start = combined_list[i] + 4
                    sort_index = arg_index_list.index(i)
                    if sort_index < 3:
                        next_ = arg_index_list[sort_index + 1]
                        re = s[start: combined_list[next_]]
                    else:
                        re = s[start:]
                    result.append(re.strip())

                ac, sp, at, ot = result

                if at.lower() == 'it':
                    at = 'NULL'
            except ValueError:
                try:
                    pass
                except UnicodeEncodeError:
                    pass
                ac, at, sp, ot = '', '', '', ''

            quads.append((ac, at, sp, ot))  # Quadruple für ASQP speichern
    else:
        raise NotImplementedError
    return quads


def compute_scores(pred_seqs, gold_seqs, task):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):

        gold_list = extract_spans_para(task, gold_seqs[i], 'gold')
        pred_list = extract_spans_para(task, pred_seqs[i], 'pred')

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    scores = compute_f1_scores(all_preds, all_labels)

    return scores, all_labels, all_preds


def evaluate(data_loader, model, tokenizer, args):
    """
    Compute scores given the predictions and gold labels
    """
    device = torch.device('cuda:0')
    model.model.to(device)

    model.model.eval()

    outputs, targets = [], []

    for batch in tqdm(data_loader):
        # need to push the data to device
        outs = model.model.generate(input_ids=batch['source_ids'].to(device),
                                    attention_mask=batch['source_mask'].to(device),
                                    max_length=args.max_seq_length,
                                    num_beams=5)  # num_beams=5 num_beams=8, early_stopping=True)

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

        outputs.extend(dec)
        targets.extend(target)


    scores, all_labels, all_preds = compute_scores(outputs, targets, args.task)
    results = {'scores': scores}
    results.update({'all_labels': all_labels, 'all_preds': all_preds})

    return results


def train_function_ilo(args):
    global language
    if args.dataset == 'gerest':
        language = 'german'
    else:
        language = 'english'
    set_seed(args.seed)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    
        # sanity check
    # show one sample to check the code and the expected output
    
    

    train_dataset = ABSADataset(tokenizer=tokenizer,
                              task=args.task,
                              data_type='train',
                              top_k=args.top_k,
                              args=args,
                              dataset=args.train_ds,
                              max_len=args.max_seq_length)
    
    test_dataset = ABSADataset(tokenizer=tokenizer,
                              task=args.task,
                              data_type='test',
                              top_k=args.top_k,
                              args=args,
                              dataset=args.test_ds,
                              max_len=args.max_seq_length)
    
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=4)
    
    
    # training process
    if args.do_train:

        # initialize the T5 model
        tfm_model = MyT5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        model = T5FineTuner(args, tfm_model, tokenizer, train_dataset)

        # prepare for trainer
        if torch.cuda.is_available():
            gpus = 1
        else:
            gpus = None
        train_params = dict(
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=gpus,  # args.n_gpu,
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            logger=False,
            checkpoint_callback=True,
            callbacks=[],
        )

        trainer = pl.Trainer(**train_params)
        trainer.fit(model)

        # save the final model
        # model.model.save_pretrained(args.output_dir)
        # tokenizer.save_pretrained(args.output_dir)
        
        results = evaluate(test_loader, model, tokenizer, args)
        return results