export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"
export LIBTPU_INIT_ARGS="--xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

export absolute_path= # absolute path to the project directory
export llama_tokenizer_path="$absolute_path/lapa_checkpoints/tokenizer.model"
export output_dir="$absolute_path/outputs"

export project_id='lapa'
export experiment_note='lapa_finetune'

export dataset_path="$absolute_path/data/simpler.jsonl"
export experiment_id='finetune_simpler'

python3 -u -m latent_pretraining.train \
    --modality='vision,action,delta' \
    --mesh_dim='!-1,4,1,1' \
    --dtype='bf16' \
    --total_steps=500 \
    --log_freq=1 \
    --eval_steps=0 \
    --save_model_freq=0 \
    --eval_log_freq=100 \
    --save_milestone_freq=500 \
    --load_llama_config='7b' \
    --load_checkpoint="params::$absolute_path/lapa_checkpoints/params" \
    --update_llama_config="dict(action_vocab_size=245,delta_vocab_size=8,theta=50000000,max_sequence_length=2048,use_flash_attention=True,scan_attention=True,scan_query_chunk_size=512,scan_key_chunk_size=1024,remat_attention='nothing_saveable',scan_mlp=True,scan_mlp_chunk_size=8192,remat_mlp='nothing_saveable',remat_block='nothing_saveable',scan_layers=True)" \
    --tokenizer.vocab_file="$llama_tokenizer_path" \
    --optimizer.type='adamw' \
    --llama.action_vocab_size=245 \
    --llama.delta_vocab_size=8 \
    --optimizer.accumulate_gradient_steps=1 \
    --optimizer.adamw_optimizer.weight_decay=0 \
    --optimizer.adamw_optimizer.lr=2e-5 \
    --optimizer.adamw_optimizer.end_lr=2e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=0 \
    --optimizer.adamw_optimizer.lr_decay_steps=100 \
    --use_data_sharded_loader=True \
    --train_dataset.type='json_vision_delta_action' \
    --train_dataset.delta_vision_action_processor.fields_from_example='fields' \
    --train_dataset.delta_vision_action_processor.n_tokens_per_action=7 \
    --train_dataset.delta_vision_action_processor.n_tokens_per_delta=4 \
    --train_dataset.delta_vision_action_processor.img_aug=False \
    --train_dataset.delta_vision_action_processor.max_n_frames=1 \
    --train_dataset.json_delta_action_dataset.mode="pad" \
    --train_dataset.json_delta_action_dataset.path="$dataset_path" \
    --train_dataset.json_delta_action_dataset.seq_length=384 \
    --train_dataset.json_delta_action_dataset.batch_size=128 \
    --train_dataset.json_delta_action_dataset.tokenizer_processes=1 \
    --train_dataset.json_delta_action_dataset.tokenizer_parallel_chunk_size=128 \
    --train_dataset.json_delta_action_dataset.tokenizer_parallel_batch_size=128 \
    --train_dataset.json_delta_action_dataset.use_data_sharded_loader=True \
    --checkpointer.save_optimizer_state=False \
    --autoresume=False \
    --logger.append_uuid=False \
    --logger.online=True \
    --logger.project_id="$project_id" \
    --logger.experiment_id="$experiment_id" \
    --logger.experiment_note="$experiment_note" \
    --logger.output_dir="$output_dir" \
    --logger.wandb_dir="$HOME/experiment_output/$project_id"



