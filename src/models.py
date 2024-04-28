"""Custom models for few-shot learning specific operations."""

import copy

import torch
import torch.nn as nn
from transformers.modeling_bert import (
    BertPreTrainedModel,
    BertModel,
    BertOnlyMLMHead,
)
from transformers.modeling_roberta import (
    RobertaModel,
    RobertaLMHead,
    RobertaClassificationHead,
    RobertaPreTrainedModel,
)

from transformers.modeling_electra import (
    ElectraPreTrainedModel,
    ElectraModel,
    ElectraDiscriminatorPredictions,
)
from transformers.modeling_gpt2 import (
    GPT2PreTrainedModel,
    GPT2Model,
)
from transformers.modeling_t5 import (
    T5PreTrainedModel,
    T5Stack,
)
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from torch.nn import CrossEntropyLoss

import logging

logger = logging.getLogger(__name__)


def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    """
    Resize the segment (token type) embeddings for BERT
    """
    if hasattr(model, "bert"):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(new_num_types, old_token_type_embeddings.weight.size(1))
    if not random_segment:
        new_token_type_embeddings.weight.data[: old_token_type_embeddings.weight.size(0)] = (
            old_token_type_embeddings.weight.data
        )

    model.config.type_vocab_size = new_num_types
    if hasattr(model, "bert"):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError


class BertForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.classifier = SimpleClassificationHead(config.hidden_size, config.hidden_dropout_prob, config.num_labels)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)
        assert mask_pos is not None and mask_pos.size(1) == 1  # Only support one mask position
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]

        if mask_pos is None:
            logits = self.classifier(sequence_output[:, 0, :])
        else:
            sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

            # Logits over vocabulary tokens
            prediction_mask_scores = self.cls(sequence_mask_output)

            # Return logits for each label
            logits = []
            for label_id in range(len(self.label_word_list)):
                logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
            logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack(
                    [
                        1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                        (labels.view(-1) - self.lb) / (self.ub - self.lb),
                    ],
                    -1,
                )
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output


class RobertaForPromptFinetuning(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.tokenizer = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None

        self.output_attentions = False

    def forward(
        self,
        input_ids,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        sep_pos=None,
    ):
        batch_size = input_ids.size(0)
        assert mask_pos is not None and mask_pos.size(1) == 1  # Only support one mask position
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # output_attentions=self.output_attentions,
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]  # sequence_output:[batch_size, seq_len. embed_size]

        if mask_pos is None:
            raise NotImplementedError
        else:
            sequence_mask_output = sequence_output[
                torch.arange(sequence_output.size(0)), mask_pos
            ]  # [batch_size, embed_size]

            prediction_mask_scores = self.lm_head(sequence_mask_output)  # [batch_size, vocab_size]

            # Return logits for each label
            logits = []
            for label_id in range(len(self.label_word_list)):
                logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
            logits = torch.cat(logits, -1)  # [batch_size, label_nums]

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack(
                    [
                        1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                        (labels.view(-1) - self.lb) / (self.ub - self.lb),
                    ],
                    -1,
                )
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output


class ElectraForPromptFinetuning_maskOnlyDiscrimination(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraModel(config)
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.init_weights()

    def forward(
        self,
        input_ids=None,  # should be [batch_size, num_classes, seq_len]
        attention_mask=None,  # should be [batch_size, num_classes, seq_len], and same for all classes
        mask_pos=None,  # should be None
        label_pos=None,  # should be [batch_size, num_classes, 1] and should be the same for all classes
        labels=None,
        tf_idf=None,
    ):
        batch_size = input_ids.size(0)
        attention_mask = attention_mask[:, 0, :]  # [batch_size, seq_len]
        label_pos = label_pos[:, 0]  # [batch_size, 1]
        assert label_pos.size(1) == 1

        mask_logits_list = []
        # for (k, new_input_ids) in enumerate(new_input_ids_list):
        for k in range(self.num_labels):
            discriminator_hidden_states = self.electra(
                input_ids=input_ids[:, k, :],
                attention_mask=attention_mask,
            )
            discriminator_sequence_output = discriminator_hidden_states[0]  # [batch_size, seq_len, hidden_size]
            logits = -self.discriminator_predictions(discriminator_sequence_output).view(
                batch_size, -1
            )  # [batch_size, seq_len]
            mask_logits = torch.gather(logits, dim=1, index=label_pos).view(batch_size)  # [batch_size]
            mask_logits_list.append(mask_logits)
        logits = torch.stack(mask_logits_list, dim=1)  # [batch_size, num_labels]
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            target = torch.zeros(batch_size, self.num_labels).to(labels.device)
            target.index_put_(
                (torch.arange(batch_size).to(labels.device), labels.view(batch_size)),
                torch.tensor(1.0).to(labels.device),
            )
            loss = loss_fct(logits.view(batch_size, self.num_labels), target)
            # loss_fct = nn.CrossEntropyLoss()
            # loss = loss_fct(mask_label_logits.view(batch_size, self.num_labels), labels)
        output = (logits,)
        return ((loss,) + output) if loss is not None else output


class ElectraForPromptFinetuning_multiTokenDiscrimination(ElectraPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraModel(config)
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)
        self.loss_weight = nn.Parameter(torch.zeros(3), requires_grad=False)

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.init_weights()
        # logger.info('Model hyper parameters:')

    def forward(
        self,
        input_ids=None,  # should be [batch_size, num_classes, seq_len]
        attention_mask=None,  # should be [batch_size, num_classes, seq_len], and same for all classes
        mask_pos=None,  # should be None
        label_pos=None,  # should be [batch_size, num_classes, 1] and should be the same for all classes
        labels=None,
        tf_idf=None,  # if not None, should be [batch_size, num_classes, seq_len]
    ):
        # assert tf_idf is not None
        batch_size = input_ids.size(0)
        device = input_ids.device
        assert mask_pos is None
        assert self.label_word_list is not None
        assert len(self.label_word_list) == self.num_labels
        assert label_pos is not None
        label_pos = label_pos[:, 0, 0]  # [batch_size]
        if tf_idf is None:
            tf_idf = torch.ones_like(input_ids)
        tf_idf = tf_idf[:, 0, :]  # [batch_size, seq_len]

        attention_mask = attention_mask[:, 0, :]  # [batch_size, seq_len]
        real_length = attention_mask.sum(-1)  # [batch_size]
        if self.data_args.task_name in ["snli", "qnli", "mnli", "rte", "qqp", "mrpc"]:
            sent1_mask, sent2_mask = torch.zeros_like(attention_mask), torch.zeros_like(attention_mask)
            for b in range(batch_size):
                sent1_mask[b, 1 : label_pos[b] - 1] = 1
                sent2_mask[b, label_pos[b] + 2 : real_length[b] - 1] = 1
        elif self.data_args.task_name in ["sst-2", "sst-5", "mr", "cr"]:
            sent1_mask, sent2_mask = torch.zeros_like(attention_mask), None
            for b in range(batch_size):
                sent1_mask[b, 1 : label_pos[b] - 2] = 1
        else:
            raise NotImplementedError

        for k, label_word_id in enumerate(self.label_word_list):  # check label_pos
            assert (input_ids[:, k, :].gather(dim=1, index=label_pos.view(-1, 1)).view(-1) - label_word_id).sum() == 0

        sent1_logits_list, sent2_logits_list, mask_logits_list = [], [], []
        for k in range(input_ids.size(1)):
            class_input_ids = input_ids[:, k, :]  # [batch_size, seq_len]
            discriminator_hidden_states = self.electra(
                input_ids=class_input_ids,
                attention_mask=attention_mask,
            )
            discriminator_sequence_output = discriminator_hidden_states[0]  # [batch_size, seq_len, hidden_size]
            # 这里要取‘-’，因为ELECTRA是将replaced预测为1
            logits = -self.discriminator_predictions(discriminator_sequence_output).view(
                batch_size, -1
            )  # [batch_size, seq_len]
            sent1_logits_weight = (tf_idf * sent1_mask) / ((tf_idf * sent1_mask).sum(dim=-1, keepdims=True) + 1e-7)
            sent1_logits = (logits * sent1_logits_weight).sum(-1)  # [batch_size]
            sent1_logits_list.append(sent1_logits)
            if sent2_mask is not None:
                sent2_logits_weight = (tf_idf * sent2_mask) / ((tf_idf * sent2_mask).sum(dim=-1, keepdims=True) + 1e-7)
                sent2_logits = (logits * sent2_logits_weight).sum(-1)  # [batch_size]
                sent2_logits_list.append(sent2_logits)
            mask_logits = torch.gather(logits, dim=1, index=label_pos.view(-1, 1)).view(batch_size)  # [batch_size]
            mask_logits_list.append(mask_logits)
        sent1_label_logits = torch.stack(sent1_logits_list, dim=1)  # [batch_size, num_labels]
        if sent2_mask is not None:
            sent2_label_logits = torch.stack(sent2_logits_list, dim=1)  # [batch_size, num_labels]
        mask_label_logits = torch.stack(mask_logits_list, dim=1)  # [batch_size, num_labels]
        if sent2_mask is None:
            weight = self.loss_weight[:2]  # [2]
            logits = sent1_label_logits.softmax(-1) * weight[0] + mask_label_logits.softmax(-1) * weight[1]
        else:
            weight = self.loss_weight  # [3]
            logits = (
                sent1_label_logits.softmax(-1) * weight[0]
                + mask_label_logits.softmax(-1) * weight[1]
                + sent2_label_logits.softmax(-1) * weight[2]
            )
        if self.model_args.output_each_part_logits:
            if sent2_mask is None:
                logits = torch.cat(
                    (logits, sent1_label_logits, mask_label_logits), dim=-1
                )  # [batch_size, num_labels*3]
            else:
                logits = torch.cat(
                    (logits, sent1_label_logits, mask_label_logits, sent2_label_logits), dim=-1
                )  # [batch_size, num_labels*4]

        loss = None
        if labels is not None:
            # loss_fct = nn.BCEWithLogitsLoss()
            # target = torch.zeros(batch_size, self.num_labels).to(device)
            # target.index_put_((torch.arange(batch_size).to(device), labels.view(batch_size)),torch.tensor(1.0).to(device))
            # loss = loss_fct(logits.view(batch_size, self.num_labels), target)
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = torch.tensor(0.0, device=device)
            loss += loss_fct(sent1_label_logits.view(batch_size, self.num_labels), labels).mean(-1) * weight[0]
            loss += loss_fct(mask_label_logits.view(batch_size, self.num_labels), labels).mean(-1) * weight[1]
            if sent2_mask is not None:
                loss += loss_fct(sent2_label_logits.view(batch_size, self.num_labels), labels).mean(-1) * weight[2]
            # loss_fct = nn.NLLLoss()
            # log_p = logits.log()
            # loss = loss_fct(log_p.view(batch_size, self.num_labels), labels)
        output = (logits,)
        return ((loss,) + output) if loss is not None else output


# modified from transformers.modeling_gpt2.GPT2LMHeadModel
class GPT2ForPromptFinetuning(GPT2PreTrainedModel):
    authorized_missing_keys = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.ori_vocab_size, bias=False)
        self.num_labels = config.num_labels

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        mask_pos=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        """
        if "past" in kwargs:
            raise RuntimeError("past is deprecated, use `past_key_values` instead.")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        assert not self.num_labels == 1, "Regression task is not supported."
        assert len(self.label_word_list) == self.num_labels
        label_word_ids = self.label_word_list.long().reshape(-1).to(input_ids.device)

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs[0]  # [batch_size, seq_len, hidden_size]

        lm_logits = self.lm_head(hidden_states)  # [batch_size, seq_len, vocab_size]

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    f"unexpected if using padding tokens in conjuction with `inputs_embeds.`"
                )

        assert mask_pos is not None
        assert mask_pos.reshape(-1).eq(sequence_lengths).all(), "mask_pos should be the last token of each sequence"

        cls_logits = lm_logits[range(batch_size), sequence_lengths - 1, :].contiguous()  # [batch_size, vocab_size]
        cls_logits = cls_logits[:, label_word_ids].contiguous()  # [batch_size, num_labels]

        loss = None
        if labels is not None:  # here, labels is classification labels
            # Shift so that tokens < n predict n
            # shift_logits = lm_logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()

            # only calc loss on the last token, we assume the label words of prompt is the last token
            shift_logits = lm_logits[
                range(batch_size), sequence_lengths - 1, :
            ].contiguous()  # [batch_size, vocab_size]
            shift_labels = label_word_ids[labels.view(-1)]  # [batch_size]

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        output = (cls_logits,)
        return ((loss,) + output) if loss is not None else output


# modified from transformers.modeling_t5.T5ForConditionalGeneration
class T5ForPromptFinetuning(T5PreTrainedModel):
    authorized_missing_keys = [r"encoder\.embed_tokens\.weight", r"decoder\.embed_tokens\.weight", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model
        self.num_labels = config.num_labels

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        head_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        decoder_labels=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)

        # assert encoder_outputs is None
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]  # [batch_size, seq_len, hidden_size]

        # here, labels is classification labels
        # if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
        if decoder_labels is not None:  # [batch_size, dec_seq_len] [<extra_id_0>, label_word_token]
            assert decoder_input_ids is None
            decoder_input_ids = self._shift_right(decoder_labels)  # add <s> token for decoder input

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]  # [batch_size, dec_seq_len, hidden_size]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim**-0.5)
        lm_logits = self.lm_head(sequence_output)  # [batch_size, dec_seq_len, vocab_size]

        # assert lm_logits.size(1) == 2  # [<s>, <extra_id_0>, label_word_token, <extra_id_1>]
        label_word_ids = self.label_word_list.long().reshape(-1).to(lm_logits.device)
        cls_logits = lm_logits[:, 1, label_word_ids].contiguous()  # [batch_size, num_labels]

        loss = None
        if labels is not None and decoder_labels is not None:
            # here labels is classification labels
            assert (label_word_ids[labels].view(-1) == decoder_labels[:, 1]).all()  # 1 is label word token
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), decoder_labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

            # loss = loss_fct(cls_logits.view(-1, cls_logits.size(-1)), labels.view(-1))

        # lm_logits: -1 is label_word_ids, predict <\s>, -2 is <extra_id_0>, predict label word token, so we use -2
        if not return_dict:
            output = (cls_logits,)  # + decoder_outputs[1:] + encoder_outputs  # to save memory
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=cls_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


class SimpleClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, hidden_dropout_prob, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
