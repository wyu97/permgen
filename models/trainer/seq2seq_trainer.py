import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DistributedSampler, RandomSampler

from trainer.model_trainer import Trainer
from transformers.configuration_fsmt import FSMTConfig
from transformers.optimization import Adafactor, AdamW, get_linear_schedule_with_warmup

try:
    from transformers.utils import label_smoothed_nll_loss
except ImportError:
    from trainer.model_utils import label_smoothed_nll_loss
logger = logging.getLogger(__name__)


class Seq2SeqTrainer(Trainer):
    def __init__(self, config, data_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.data_args = data_args
        self.max_gen_length = data_args.val_max_target_length
        self.vocab_size = self.config.tgt_vocab_size if isinstance(self.config, FSMTConfig) else self.config.vocab_size

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.adafactor:
                self.optimizer = Adafactor(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    scale_parameter=False,
                    relative_step=False,
                )

            else:
                self.optimizer = AdamW(
                    optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon
                )

        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return None
        if self.args.sortish_sampler:
            self.train_dataset.make_sortish_sampler(
                self.args.per_device_train_batch_size, distributed=self.args.n_gpu > 1
            )

        return (
            RandomSampler(self.train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(self.train_dataset)
        )

    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, use_cache=False)
        logits = outputs[0]
        return self._compute_loss(logits, labels)

    def _compute_loss(self, logits, labels):
        if self.args.label_smoothing == 0:
            # Same behavior as modeling_bart.py
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            assert logits.shape[-1] == self.vocab_size
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.args.label_smoothing, ignore_index=self.config.pad_token_id
            )
        return loss

    def prediction_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if self.args.predict_with_generate and not self.args.prediction_loss_only:
                generated_tokens = model.generate(
                    input_ids=inputs["input_ids"],
                    input_segment=inputs["input_segment"],
                    input_position=inputs["input_position"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=True,
                    num_beams=self.data_args.eval_beams,
                    max_length=self.max_gen_length,
                    do_sample=self.data_args.do_sample,
                    top_k=self.data_args.top_k,
                    top_p=self.data_args.top_p,
                    n_sample=self.data_args.n_sample,
                    k_out=self.data_args.k_out,
                    n_special_tokens=self.data_args.n_special_tokens
                )

                # in case the batch is shorter than max length, the output should be padded
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, self.max_gen_length)

            labels_out = inputs.get("labels")
            # Call forward again to get loss # TODO: avoidable?
            outputs = model(**inputs, use_cache=False)
            loss = self._compute_loss(outputs[1], labels_out)
            loss = loss.mean().item()

            if self.args.prediction_loss_only:
                return (loss, None, None, None)

            logits = generated_tokens if self.args.predict_with_generate else outputs[1]

        labels_out = labels_out.detach()
        labels = self._pad_tensors_to_max_len(labels_out, self.max_gen_length)
        return (loss, logits.detach(), labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        padded_tensor = self.config.pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
