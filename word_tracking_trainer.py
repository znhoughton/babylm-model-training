"""
WordTrackingTrainer — a Trainer subclass that records cumulative per-step
token frequencies for the entire vocabulary during training.

At each training step, torch.bincount over the flattened batch input_ids
gives per-token counts for all vocab_size tokens at once.  These are
accumulated and periodically flushed to an NPZ file.

To find when any word was first seen (or its cumulative frequency at any
checkpoint), tokenize the word and look up its token ID(s) in the matrix.

Output NPZ keys
---------------
  steps        : int32  (total_steps,)             — global step numbers (1-indexed)
  token_counts : int32  (total_steps, vocab_size)  — CUMULATIVE token counts

Multiple epochs
---------------
global_step increments continuously across epochs, so the NPZ covers all
epochs seamlessly (steps 1-1144 epoch 1, 1145-2288 epoch 2, etc.).

Usage in train_autoreg.py
--------------------------
    from word_tracking_trainer import WordTrackingTrainer

    trainer = WordTrackingTrainer(model=model, args=training_args, ...)

    if data_args.word_tracking_output:
        trainer.set_word_tracking(
            vocab_size=model.config.vocab_size,
            output_path=data_args.word_tracking_output,
        )
"""

import logging
from pathlib import Path

import numpy as np
import torch
from transformers import Trainer

logger = logging.getLogger(__name__)


class WordTrackingTrainer(Trainer):
    """
    Trainer that tracks cumulative per-step token frequencies for the
    full vocabulary.  Call set_word_tracking() before trainer.train().
    """

    def set_word_tracking(
        self,
        vocab_size: int,
        output_path: str,
        save_every: int = 50,
    ):
        """
        Parameters
        ----------
        vocab_size   : model vocabulary size (e.g. 8192)
        output_path  : where to write the NPZ
        save_every   : flush to disk every this many steps (default 50)
        """
        self._wt_enabled    = True
        self._wt_vocab_size = vocab_size
        self._wt_output     = Path(output_path)
        self._wt_save_every = save_every
        self._wt_output.parent.mkdir(parents=True, exist_ok=True)

        self._wt_step_list:   list = []
        self._wt_counts_list: list = []
        self._wt_cumsum = np.zeros(vocab_size, dtype=np.int64)

        logger.info(
            f"[WordTracking] Tracking all {vocab_size} tokens → {self._wt_output}"
        )

    def training_step(self, model, inputs, *args, **kwargs):
        # transformers 4.46 added num_items_in_batch as a third positional
        # argument. Accept and forward whatever the installed Trainer passes so
        # this subclass does not pin the library version.
        result = super().training_step(model, inputs, *args, **kwargs)

        if not getattr(self, "_wt_enabled", False):
            return result

        # global_step = completed steps; current step is +1 (1-indexed)
        current_step = self.state.global_step + 1

        flat = inputs["input_ids"].reshape(-1).cpu().to(torch.long)
        batch_counts = torch.bincount(flat, minlength=self._wt_vocab_size).numpy()

        self._wt_cumsum += batch_counts
        self._wt_step_list.append(current_step)
        self._wt_counts_list.append(self._wt_cumsum.copy().astype(np.int32))

        if len(self._wt_step_list) % self._wt_save_every == 0:
            self._wt_flush()

        return result

    def _wt_flush(self):
        if not self._wt_step_list:
            return
        np.savez_compressed(
            self._wt_output,
            steps        = np.array(self._wt_step_list, dtype=np.int32),
            token_counts = np.stack(self._wt_counts_list, axis=0).astype(np.int32),
        )
        logger.info(
            f"[WordTracking] Flushed {len(self._wt_step_list)} steps "
            f"→ {self._wt_output}"
        )

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        if getattr(self, "_wt_enabled", False):
            self._wt_flush()

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics=metrics)
        if getattr(self, "_wt_enabled", False):
            self._wt_flush()