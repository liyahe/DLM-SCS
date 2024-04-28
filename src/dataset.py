"""Dataset utils for different data settings for GLUE."""

import logging
import torch
import json
from src.processors import processors_mapping
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union
import pandas as pd
from tqdm import tqdm
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass(frozen=False)
class ModelInput:  # similar to InputFeatures. We only use ModelInput for the model, the model output is still tuple
    """
    Inherit from Transformers' InputFeatuers.
    """

    input_ids: Union[List[int], List[List[int]]] = None
    attention_mask: Union[List[int], List[List[int]]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    mask_pos: Optional[List[int]] = None  # Position of the mask token
    label_pos: Optional[List[int]] = None  # Position of the labels (for electra) ## lyh
    sep_pos: Optional[List[int]] = None  # Position of the sep ## lyh
    label_word_list: Optional[List[int]] = None  # Label word mapping (dynamic)
    tf_idf: Optional[List[float]] = None
    decoder_labels: Optional[List[int]] = None  # T5 decoder input ids, only for T5

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


def get_all_index(lst=None, item=""):  # lyh
    return [index for (index, value) in enumerate(lst) if value == item]


def input_example_to_string(example, sep_token):
    if example.text_b is None:
        return example.text_a
    else:
        # Warning: very simple hack here
        return example.text_a + " " + sep_token + " " + example.text_b


def input_example_to_tuple(example):
    if example.text_b is None:
        if pd.isna(example.text_a) or example.text_a is None:
            logger.warning("Empty input")
            return [""]
        else:
            return [example.text_a]
    else:
        return [example.text_a, example.text_b]


def tokenize_multipart_input(
    input_text_list,
    max_length,
    tokenizer,
    task_name=None,
    prompt=False,
    model_type=None,
    template=None,
    label_word_list=None,
    first_sent_limit=None,
    other_sent_limit=None,
    truncate_head=False,
    support_labels=None,
    idf_dict=False,
    label=None,  # classification label
):
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    def process_one_template(template_):
        input_ids = []
        attention_mask = []
        token_type_ids = []  # Only for BERT
        label_pos = []  # Position of the labels (for electra) ## lyh

        if prompt:
            """
            Concatenate all sentences and prompts based on the provided template.
            Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
            *xx* represent variables:
                *cls*: cls_token
                *mask*: mask_token
                *sep*: sep_token
                *sep+*: sep_token, also means +1 for segment id
                *sent_i*: sentence i (input_text_list[i])
                *sent-_i*: same as above, but delete the last token
                *sentl_i*: same as above, but use lower case for the first word
                *sentl-_i*: same as above, but use lower case for the first word and delete the last token
                *+sent_i*: same as above, but add a space before the sentence
                *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
                *label_i*: label_word_list[i]
                *label_x*: label depends on the example id (support_labels needed). this is only used in GPT-3's in-context learning
            Use "_" to replace space.
            PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
            """
            assert template_ is not None and isinstance(template_, str) and len(template_) > 0

            special_token_mapping = {
                "cls": tokenizer.cls_token_id,
                "mask": tokenizer.mask_token_id,
                "sep": tokenizer.sep_token_id,
                "sep+": tokenizer.sep_token_id,
            }
            if "t5" in model_type:
                # T5 does not have mask token, use <extra_id_0> instead
                # logger.info("T5 does not have mask token, use <extra_id_0> instead")
                special_token_mapping["<extra_id_0>"] = tokenizer._convert_token_to_id("<extra_id_0>")
                special_token_mapping["</s>"] = tokenizer._convert_token_to_id("</s>")
            template_components = template_.split("*")  # Get variable list in the template
            segment_id = 0  # Current segment id. Segment id +1 if encountering sep+.

            for part_id, part in enumerate(template_components):
                new_tokens = []
                segment_plus_1_flag = False
                if part in special_token_mapping:
                    if part == "cls" and "t5" in model_type.lower():
                        # T5 does not have cls token
                        continue
                    new_tokens.append(special_token_mapping[part])
                    if part == "sep+":
                        segment_plus_1_flag = True
                elif part[:6] == "label_":
                    # Note that label_word_list already has extra space, so do not add more space ahead of it.
                    label_id = int(part.split("_")[1])
                    label_word = label_word_list[label_id]
                    new_tokens.append(label_word)
                    label_pos.append(len(input_ids) + len(new_tokens) - 1)  # lyh
                elif part[:7] == "labelx_":
                    instance_id = int(part.split("_")[1])
                    label_id = support_labels[instance_id]
                    label_word = label_word_list[label_id]
                    new_tokens.append(label_word)
                elif part[:8] == "virtual_":  # lyh
                    # Virtual tokens for soft prompt
                    virtual_i = int(part.split("_")[1])
                    new_tokens.append(virtual_token_ids[virtual_i])
                elif part[:5] == "sent_":
                    sent_id = int(part.split("_")[1])
                    new_tokens += enc(input_text_list[sent_id])
                elif part[:6] == "+sent_":
                    # Add space
                    sent_id = int(part.split("_")[1])
                    new_tokens += enc(" " + input_text_list[sent_id])
                elif part[:6] == "sent-_":
                    # Delete the last token
                    sent_id = int(part.split("_")[1])
                    new_tokens += enc(input_text_list[sent_id][:-1])
                elif part[:6] == "sentl_":
                    # Lower case the first token
                    sent_id = int(part.split("_")[1])
                    text = input_text_list[sent_id]
                    text = text[:1].lower() + text[1:]
                    new_tokens += enc(text)
                elif part[:7] == "+sentl_":
                    # Lower case the first token and add space
                    sent_id = int(part.split("_")[1])
                    text = input_text_list[sent_id]
                    text = text[:1].lower() + text[1:]
                    new_tokens += enc(" " + text)
                elif part[:7] == "sentl-_":
                    # Lower case the first token and discard the last token
                    sent_id = int(part.split("_")[1])
                    text = input_text_list[sent_id]
                    text = text[:1].lower() + text[1:]
                    new_tokens += enc(text[:-1])
                elif part[:6] == "sentu_":
                    # Upper case the first token
                    sent_id = int(part.split("_")[1])
                    text = input_text_list[sent_id]
                    text = text[:1].upper() + text[1:]
                    new_tokens += enc(text)
                elif part[:7] == "+sentu_":
                    # Upper case the first token and add space
                    sent_id = int(part.split("_")[1])
                    text = input_text_list[sent_id]
                    text = text[:1].upper() + text[1:]
                    new_tokens += enc(" " + text)
                else:
                    # Just natural language prompt
                    part = part.replace("_", " ")
                    # handle special case when T5 tokenizer might add an extra space
                    if len(part) == 1:  # if the part is a single word
                        new_tokens.append(tokenizer._convert_token_to_id(part))
                    else:
                        new_tokens += enc(part)

                if part[:4] == "sent" or part[1:5] == "sent":
                    # If this part is the sentence, limit the sentence length
                    sent_id = int(part.split("_")[1])
                    if sent_id == 0:
                        if first_sent_limit is not None:
                            new_tokens = new_tokens[:first_sent_limit]
                    else:
                        if other_sent_limit is not None:
                            new_tokens = new_tokens[:other_sent_limit]

                input_ids += new_tokens
                attention_mask += [1 for _ in range(len(new_tokens))]
                token_type_ids += [segment_id for _ in range(len(new_tokens))]

                if segment_plus_1_flag:
                    segment_id += 1
        else:
            input_ids = [tokenizer.cls_token_id]
            attention_mask = [1]
            token_type_ids = [0]

            for sent_id, input_text in enumerate(input_text_list):
                if input_text is None:
                    # Do not have text_b
                    continue
                if pd.isna(input_text) or input_text is None:
                    # Empty input
                    input_text = ""
                input_tokens = enc(input_text) + [tokenizer.sep_token_id]
                input_ids += input_tokens
                attention_mask += [1 for _ in range(len(input_tokens))]
                token_type_ids += [sent_id for _ in range(len(input_tokens))]

            if "t5" in model_type or "gpt2" in model_type:  # T5 and gpt2 does not have CLS token
                input_ids = input_ids[1:]
                attention_mask = attention_mask[1:]
                token_type_ids = token_type_ids[1:]

            if "gpt2" in model_type:  # gpt2 does not have SEP token
                input_ids = input_ids[:-1]
                attention_mask = attention_mask[:-1]
                token_type_ids = token_type_ids[:-1]

        # Padding
        if first_sent_limit is not None and len(input_ids) > max_length:
            # If using sentence limit, the total length still exceeds the maximum limit, report a warning
            logger.warning("Input exceeds max_length limit: {}".format(tokenizer.decode(input_ids)))

        while len(input_ids) < max_length:
            input_ids.append(tokenizer.pad_token_id)
            attention_mask.append(0)
            token_type_ids.append(0)

        # Truncate
        if len(input_ids) > max_length:
            if truncate_head:
                input_ids = input_ids[-max_length:]
                attention_mask = attention_mask[-max_length:]
                token_type_ids = token_type_ids[-max_length:]
            else:
                # Default is to truncate the tail
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                token_type_ids = token_type_ids[:max_length]

        result = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "BERT" in type(tokenizer).__name__:
            # Only provide token type ids for BERT
            result["token_type_ids"] = token_type_ids

        # Fina label word token position
        if "label" in template_ and prompt:  # and 'mask' not in template_:
            assert "mask" not in template_
            # Make sure that the masked position is inside the max_length
            for lp in label_pos:
                assert lp < max_length and input_ids[lp] in label_word_list  # check
            # assert max(label_pos) < max_length
            result["label_pos"] = label_pos

        # Find mask token
        if (
            "mask" in template_ and prompt and "t5" not in model_type
        ):  # prompt and not different_template_for_labels:  # lyh
            # mask_pos = [input_ids.index(tokenizer.mask_token_id)]
            mask_pos = get_all_index(input_ids, tokenizer.mask_token_id)
            # Make sure that the masked position is inside the max_length
            assert mask_pos[-1] < max_length
            result["mask_pos"] = mask_pos

        # for our method
        if idf_dict is not None:
            from src.tfidf import tfidf

            valid_len = sum(attention_mask)
            input_ids_count = Counter(input_ids[:valid_len])
            scores = {word: tfidf(word, input_ids_count, idf_dict=idf_dict) for word in input_ids_count}
            tf_idf = [scores[word] for word in input_ids[:valid_len]] + [0] * max(0, (len(input_ids) - valid_len))
            result["tf_idf"] = tf_idf

        if "t5" in model_type and label is not None and prompt:
            # T5 decoder_labels
            # result['decoder_labels'] = enc(f'<extra_id_0> {tokenizer.convert_ids_to_tokens(label_word_list[label])} <extra_id_1> </s>')
            result["decoder_labels"] = enc(f"<extra_id_0> {tokenizer.convert_ids_to_tokens(label_word_list[label])}")
            assert len(result["decoder_labels"]) == 2

        return result

    virtual_token_ids = tokenizer.additional_special_tokens_ids
    all_results, all_template_list = [], []

    # for electra, input prompt with different label words
    if "electra" in model_type:
        for idx, label_word in enumerate(label_word_list):
            all_template_list.append(template.replace("mask", f"label_{idx}"))
    elif "bert" in model_type or "roberta" in model_type or "gpt2" in model_type or "t5" in model_type:
        # for bert and roberta, only input prompt with mask
        all_template_list = [template]
    else:
        raise ValueError(f"Not support model type {model_type}")

    for template in all_template_list:
        all_results.append(process_one_template(template))

    if len(all_results) > 1:  # for electra
        final_result = {}
        keys = all_results[0].keys()
        for key in keys:
            ll = []
            for res in all_results:
                ll.append(res[key])
            final_result[key] = ll
    else:
        final_result = all_results[0]

    return final_result


class FewShotDataset(torch.utils.data.Dataset):
    """Few-shot dataset."""

    def __init__(self, args, tokenizer, cache_dir=None, mode="train", idf_dict=None):
        self.args = args
        self.task_name = args.task_name
        self.processor = processors_mapping[args.task_name]
        self.tokenizer = tokenizer
        self.mode = mode
        self.idf_dict = idf_dict

        assert mode in ["train", "dev", "test"]

        # Get label list and (for prompt) label word list
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        if args.prompt:
            assert args.mapping is not None
            self.label_to_word = eval(args.mapping)

            for key in self.label_to_word:
                # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
                if self.label_to_word[key][0] not in ["<", "[", ".", ","]:
                    # Make sure space+word is in the vocabulary
                    assert len(tokenizer.tokenize(" " + self.label_to_word[key])) == 1
                    self.label_to_word[key] = tokenizer._convert_token_to_id(
                        tokenizer.tokenize(" " + self.label_to_word[key])[0]
                    )
                else:
                    self.label_to_word[key] = tokenizer._convert_token_to_id(self.label_to_word[key])
                logger.info(
                    "Label {} to word {} ({})".format(
                        key, tokenizer._convert_id_to_token(self.label_to_word[key]), self.label_to_word[key]
                    )
                )

            if len(self.label_list) > 1:
                self.label_word_list = [self.label_to_word[label] for label in self.label_list]
            else:
                # Regression task
                # '0' represents low polarity and '1' represents high polarity.
                self.label_word_list = [self.label_to_word[label] for label in ["0", "1"]]
        else:
            self.label_to_word = None
            self.label_word_list = None

        self.num_sample = 1

        # If we use multiple templates, we also need to do multiple sampling during inference.
        if args.prompt and args.template_list is not None:
            logger.info(
                "There are %d templates. Multiply num_sample by %d" % (len(args.template_list), len(args.template_list))
            )
            self.num_sample *= len(args.template_list)

        logger.info("Total num_sample for mode %s: %d" % (mode, self.num_sample))

        # support_examples: using for demonstration sampling
        # query_examples: original examples
        logger.info(f"Creating features from dataset file at {args.data_dir}")

        # The support examples are sourced from the training set.
        self.support_examples = self.processor.get_train_examples(args.data_dir)

        if mode == "dev":
            self.query_examples = self.processor.get_dev_examples(args.data_dir)
        elif mode == "test":
            self.query_examples = self.processor.get_test_examples(args.data_dir)
        else:
            self.query_examples = self.support_examples

        # Size is expanded by num_sample
        self.size = len(self.query_examples) * self.num_sample

        # Prepare examples
        # support_indices = list(range(len(self.support_examples)))
        self.example_idx = []
        for sample_idx in range(self.num_sample):
            for query_idx in range(len(self.query_examples)):
                context_indices = None
                # We'll subsample context_indices further later.
                self.example_idx.append((query_idx, context_indices, sample_idx))

        # If it is not training, we pre-process the data; otherwise, we process the data online.
        cnt = 0
        self.features = []
        for query_idx, context_indices, bootstrap_idx in tqdm(self.example_idx, desc=f"processing {mode} data"):
            # The input (query) example
            example = self.query_examples[query_idx]

            supports = None

            if args.template_list is not None:
                raise NotImplementedError
                # template = args.template_list[bootstrap_idx % len(args.template_list)]  # Use template in order
            else:
                template = args.template

            self.features.append(
                self.convert_fn(
                    example=example,
                    supports=supports,  # We don't use supports for now. it is for demonstration sampling.
                    label_list=self.label_list,
                    prompt=args.prompt,
                    template=template,
                    label_word_list=self.label_word_list,
                    verbose=True if cnt == 0 else False,
                )
            )
            cnt += 1

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return self.features[i]

    def get_labels(self):
        return self.label_list

    def convert_fn(
        self, example, supports, label_list=None, prompt=False, template=None, label_word_list=None, verbose=False
    ):
        """
        Returns a list of processed "ModelInput".
        """
        max_length = self.args.max_seq_length

        # Prepare labels
        label_map = {label: i for i, label in enumerate(label_list)}  # Mapping the label names to label ids
        if len(label_list) == 1:
            # Regression
            label_map = {"0": 0, "1": 1}

        # Get example's label id (for training/inference)
        if example.label is None:
            example_label = None
        elif len(label_list) == 1:
            # Regerssion
            example_label = float(example.label)
        else:
            example_label = label_map[example.label]

        # Prepare other features
        # No using demonstrations
        inputs = tokenize_multipart_input(
            input_text_list=input_example_to_tuple(example),
            max_length=max_length,
            tokenizer=self.tokenizer,
            task_name=self.args.task_name,
            prompt=prompt,
            model_type=self.args.model_type,
            template=template,
            label_word_list=label_word_list,
            first_sent_limit=self.args.first_sent_limit,
            other_sent_limit=self.args.other_sent_limit,
            idf_dict=self.idf_dict,
            label=example_label,
        )
        features = ModelInput(**inputs, label=example_label)

        if verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))

            # logger.info("features: %s" % features)

            def show(input_ids, attention_mask, tf_idf, decoder_labels):
                valid_len = sum(attention_mask) + 3
                logger.info(f"text: {self.tokenizer.decode(input_ids[:valid_len])}")
                names = ["input_ids", "tokens"]
                all_res = [input_ids[:valid_len], self.tokenizer.convert_ids_to_tokens(input_ids[:valid_len])]
                if tf_idf is not None:
                    all_res.append(tf_idf[:valid_len])
                    names.append("tf_idf")
                logger.info(f"{tuple(names)}:{[_ for _ in zip(*all_res)]}")
                if decoder_labels is not None:
                    logger.info(f"decoder_labels text: {self.tokenizer.decode(decoder_labels)}")
                    logger.info(
                        f"decoder_labels: {[_ for _ in zip(decoder_labels, self.tokenizer.convert_ids_to_tokens(decoder_labels))]}"
                    )

            if isinstance(features.input_ids[0], list):
                for i in range(len(features.input_ids)):
                    show(
                        features.input_ids[i],
                        features.attention_mask[i],
                        features.tf_idf[i] if features.tf_idf is not None else None,
                        features.decoder_labels[i] if features.decoder_labels is not None else None,
                    )
            else:
                show(features.input_ids, features.attention_mask, features.tf_idf, features.decoder_labels)

        return features
