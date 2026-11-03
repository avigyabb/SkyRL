#!/bin/bash
# 256k-context RL sim: opt2 config on ~256k-token prompts (book excerpt + GSM8K question), short responses.
cd "$(dirname "$0")"
D=/shared/hackskyrl/data/rl_longprompt/longprompt_262144.parquet
T=trainer.policy.megatron_config.transformer_config_kwargs
NAME=${NAME:-long256} TBS=${TBS:-4} MBS=${MBS:-4} NSAMP=${NSAMP:-4} MAX_PROMPT=${MAX_PROMPT:-266240} MAX_RESP=${MAX_RESP:-1024} \
  MTPM=${MTPM:-262144} MLP_CHUNKS=${MLP_CHUNKS:-16} RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-full} RECOMPUTE_MODULES=${RECOMPUTE_MODULES:-null} bash opt2.sh \
  "data.train_data=['$D']" "data.val_data=['$D']" environment.env_class=gsm8k \
  generator.apply_overlong_filtering=false trainer.algorithm.overlong_buffer_len=256 $T.recompute_method=uniform $T.recompute_num_layers=1 "$@"
