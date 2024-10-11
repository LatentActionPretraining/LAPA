import pprint
import os

from tqdm import tqdm, trange
import numpy as np
from absl.app import run
import absl.logging as logging
import tux

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState

from latent_pretraining.data import DatasetFactory
from tux import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, global_norm, get_float_dtype_by_name,
    set_random_seed, average_metrics, get_mask,
    make_shard_and_gather_fns, with_sharding_constraint, define_flags_with_default,
    OptimizerFactory, StreamingCheckpointer
)
from latent_pretraining.llama import LLaMAConfig, FlaxLLaMAForCausalLMModule
from latent_pretraining.vision_llama import VideoLLaMAConfig, FlaxVideoLLaMAForCausalLMModule
from latent_pretraining.delta_llama import VideoLLaMAConfig, FlaxDeltaLaMAForCausalLMModule
from latent_pretraining.llama_action import VideoLLaMAConfig, FlaxActionLaMAForCausalLMModule
from latent_pretraining.delta_llama_action import VideoLLaMAConfig, FlaxDeltaActionLaMAForCausalLMModule
import random

import flax
import jax
import jax.numpy as jnp
import msgpack
import numpy as np
from flax.serialization import (from_bytes, from_state_dict, to_state_dict)
from flax.traverse_util import empty_node, flatten_dict, unflatten_dict
from tux.utils import open_file
import tensorflow as tf
tf.config.optimizer.set_jit(True)
import time

random.seed(time.time())




def l1_loss(predicted_logits, true_tokens, valid=None):
    # Get the predicted tokens by taking the argmax over logits
    predicted_tokens = jnp.argmax(predicted_logits, axis=-1)
    
    # Calculate the L1 loss as the sum of absolute differences between predicted and true tokens
    loss = jnp.abs(predicted_tokens - true_tokens)
    
    # Mask the loss with the valid mask if provided
    if valid is not None:
        loss = loss * valid
    loss = jnp.mean(jnp.sum(loss, axis=-1))
    
    return loss

FLAGS, FLAGS_DEF = define_flags_with_default(
    modality='text',
    use_data_sharded_loader=True,
    seed=42,
    mesh_dim='1,-1,1,1',
    dtype='fp32',
    total_steps=10000,
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    load_dataset_state='',
    log_freq=10,
    eval_log_freq = 10,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    tokenizer=VideoLLaMAConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    unseen_eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=VideoLLaMAConfig.get_default_config(),
    logger=tux.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
    autoresume=False,
    delta_tokens=0,
    freeze=0,
    mse_loss=1,
) 



def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    variant = tux.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = tux.user_flags_to_config_dict(FLAGS, FLAGS_DEF)

    logger = tux.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    if jax.process_index() == 0:
        output_dir = logger.output_dir
    else:
        output_dir = os.path.join(logger.output_dir, logger.experiment_id)

    if FLAGS.modality == 'text':
        config_cls = LLaMAConfig
        llama_cls = FlaxLLaMAForCausalLMModule
    elif FLAGS.modality == 'vision,text':
        config_cls = VideoLLaMAConfig
        llama_cls = FlaxVideoLLaMAForCausalLMModule
    elif FLAGS.modality == 'vision,text,delta':
        config_cls = VideoLLaMAConfig
        llama_cls = FlaxDeltaLaMAForCausalLMModule
    elif FLAGS.modality == 'vision,action':
        config_cls = VideoLLaMAConfig
        llama_cls = FlaxActionLaMAForCausalLMModule
    elif FLAGS.modality == 'vision,action,delta':
        config_cls = VideoLLaMAConfig
        llama_cls = FlaxDeltaActionLaMAForCausalLMModule
    else:
        raise ValueError(f"Unsupported modality: {FLAGS.modality}")

    mesh = config_cls.get_jax_mesh(FLAGS.mesh_dim)
    node_info = config_cls.get_ranks_and_size(mesh)

    tokenizer = config_cls.get_tokenizer(FLAGS.tokenizer)
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer, node_info=node_info)
    if FLAGS.autoresume and tux.check_exists(output_dir):
        logging.info('Found existing output. Resuming dataset from latest checkpoint...')
        resume_path = f"{output_dir}/dataset.pkl"
        dataset.load_state_dict(tux.load_pickle(resume_path))
    elif FLAGS.load_dataset_state != '':
        dataset.load_state_dict(tux.load_pickle(FLAGS.load_dataset_state))

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, dataset.tokenizer, node_info=node_info)
        eval_iterator = iter(eval_dataset)
        unseen_eval_dataset = DatasetFactory.load_dataset(
            FLAGS.unseen_eval_dataset, dataset.tokenizer, node_info=node_info)
        unseen_eval_iterator = iter(unseen_eval_dataset)

    seq_length = dataset.seq_length

    if FLAGS.load_llama_config != '':
        llama_config = config_cls.load_config(FLAGS.load_llama_config)
        updates = config_cls(**FLAGS.llama)
        llama_config.update(dict(
            remat_block=updates.remat_block,
            remat_attention=updates.remat_attention,
            remat_mlp=updates.remat_mlp,
            scan_attention=updates.scan_attention,
            scan_mlp=updates.scan_mlp,
            scan_query_chunk_size=updates.scan_query_chunk_size,
            scan_key_chunk_size=updates.scan_key_chunk_size,
            scan_mlp_chunk_size=updates.scan_mlp_chunk_size,
            scan_layers=updates.scan_layers,
            param_scan_axis=updates.param_scan_axis,
        ))
    else:
        llama_config = config_cls(**FLAGS.llama)

    if FLAGS.update_llama_config != '':
        llama_config.update(dict(eval(FLAGS.update_llama_config)))

    llama_config.update(dict(
        bos_token_id=dataset.tokenizer.bos_token_id,
        eos_token_id=dataset.tokenizer.eos_token_id,
    ))
    if llama_config.vocab_size < dataset.vocab_size:
        llama_config.update(dict(vocab_size=dataset.vocab_size))
    llama_config.update(dict(mesh_dim=FLAGS.mesh_dim))

    model = llama_cls(
        llama_config, dtype=get_float_dtype_by_name(FLAGS.dtype)
    )

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_mask(config_cls.get_weight_decay_exclusions()),
        None,
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        batch = 512
        if FLAGS.modality == 'text':
            params = model.init(
                input_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                position_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                attention_mask=jnp.ones((batch, seq_length), dtype=jnp.int32),
                rngs=rng_generator(llama_config.rng_keys()),
            )
        elif FLAGS.modality == 'vision,text':
            params = model.init(
                input_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                vision_masks=jnp.zeros((batch, seq_length), dtype=bool),
                position_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                attention_mask=jnp.ones((batch, seq_length), dtype=jnp.int32),
                rngs=rng_generator(llama_config.rng_keys()),
            )
        elif FLAGS.modality == 'vision,text,delta':
            params = model.init(
                input_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                vision_masks=jnp.zeros((batch, seq_length), dtype=bool),
                delta_masks=jnp.zeros((batch, seq_length), dtype=bool),
                position_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                attention_mask=jnp.ones((batch, seq_length), dtype=jnp.int32),
                rngs=rng_generator(llama_config.rng_keys()),
            )
        elif FLAGS.modality == 'vision,action':
            params = model.init(
                input_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                vision_masks=jnp.zeros((batch, seq_length), dtype=bool),
                action_masks=jnp.zeros((batch, seq_length), dtype=bool),
                position_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                attention_mask=jnp.ones((batch, seq_length), dtype=jnp.int32),
                rngs=rng_generator(llama_config.rng_keys()),
            )
        elif FLAGS.modality == 'vision,action,delta':
            params = model.init(
                input_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                vision_masks=jnp.zeros((batch, seq_length), dtype=bool),
                delta_masks=jnp.zeros((batch, seq_length), dtype=bool),
                action_masks=jnp.zeros((batch, seq_length), dtype=bool),
                position_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                attention_mask=jnp.ones((batch, seq_length), dtype=jnp.int32),
                rngs=rng_generator(llama_config.rng_keys()),
            )
        else:
            raise ValueError(f"Unsupported modality: {FLAGS.modality}")
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp'), 'sp'))
        def loss_and_accuracy(params):
            if FLAGS.modality == 'text':
                logits = model.apply(
                    params, 
                    batch['input_tokens'], 
                    deterministic=False,
                    rngs=rng_generator(llama_config.rng_keys()),
                ).logits
                loss, acc = cross_entropy_loss_and_accuracy(
                    logits, 
                    batch['target_tokens'],
                    batch['loss_masks']
                )
                metrics = dict(acc=acc)
                return loss, metrics
            elif FLAGS.modality == 'vision,text':
                vision_logits, text_logits = model.apply(
                    params, 
                    batch['input_tokens'], 
                    batch['input_vision_masks'],
                    deterministic=False,
                    rngs=rng_generator(llama_config.rng_keys()),
                ).logits
                vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                    vision_logits, 
                    jnp.where(batch['target_vision_masks'], batch['target_tokens'], 0),
                    batch['loss_masks'] * batch['target_vision_masks']
                )
                text_loss, text_acc = cross_entropy_loss_and_accuracy(
                    text_logits, 
                    jnp.where(batch['target_vision_masks'], 0, batch['target_tokens']),
                    batch['loss_masks'] * (1.0 - batch['target_vision_masks'])
                )
                loss = text_loss
                
                metrics = dict(
                    vision_loss=vision_loss,
                    vision_acc=vision_acc,
                    text_loss=text_loss,
                    text_acc=text_acc,
                )
            elif FLAGS.modality == 'vision,text,delta':
                vision_logits, text_logits, delta_logits = model.apply(
                    params, 
                    batch['input_tokens'], 
                    batch['input_vision_masks'],
                    batch['input_delta_masks'],
                    deterministic=False,
                    rngs=rng_generator(llama_config.rng_keys()),
                ).logits
                delta_loss, delta_acc = cross_entropy_loss_and_accuracy(
                    delta_logits, 
                    jnp.where(batch['target_delta_masks'], batch['target_tokens'], 0),
                    batch['loss_masks'] * batch['target_delta_masks']
                )
                vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                    vision_logits, 
                    jnp.where(batch['target_vision_masks'], batch['target_tokens'], 0),
                    batch['loss_masks'] * batch['target_vision_masks']
                )
                text_loss, text_acc = cross_entropy_loss_and_accuracy(
                    text_logits, 
                    jnp.where((1.0 - batch["target_vision_masks"]) * (1.0 - batch['target_delta_masks']), batch['target_tokens'], 0),
                    batch['loss_masks'] * (1.0 - batch['target_vision_masks'] * (1.0 - batch['target_delta_masks']))
                )
                loss = 0.99 * delta_loss + 0.01 * text_loss 
                
                metrics = dict(
                    vision_loss=vision_loss,
                    vision_acc=vision_acc,
                    text_loss=text_loss,
                    text_acc=text_acc,
                    delta_loss=delta_loss,
                    delta_acc=delta_acc,
                )
            elif FLAGS.modality == 'vision,action':
                vision_logits, text_logits, action_logits = model.apply(
                    params, 
                    batch['input_tokens'], 
                    batch['input_vision_masks'],
                    batch['input_action_masks'],
                    deterministic=False,
                    rngs=rng_generator(llama_config.rng_keys()),
                ).logits
                action_loss, action_acc = cross_entropy_loss_and_accuracy(
                    action_logits, 
                    jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                    batch['loss_masks'] * batch['target_action_masks']
                )
                vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                    vision_logits, 
                    jnp.where(batch['target_vision_masks'], batch['target_tokens'], 0),
                    batch['loss_masks'] * batch['target_vision_masks']
                )
                text_loss, text_acc = cross_entropy_loss_and_accuracy(
                    text_logits, 
                    jnp.where((1.0 - batch["target_vision_masks"]) * (1.0 - batch['target_action_masks']), batch['target_tokens'], 0),
                    batch['loss_masks'] * (1.0 - batch['target_vision_masks'] * (1.0 - batch['target_action_masks']))
                )
                loss = action_loss
                metrics = dict(
                    vision_loss=vision_loss,
                    vision_acc=vision_acc,
                    action_loss=action_loss,
                    action_acc=action_acc,
                    text_loss=text_loss,
                    text_acc=text_acc,
                )
            elif FLAGS.modality == 'vision,action,delta':
                vision_logits, text_logits, delta_logits, action_logits = model.apply(
                    params, 
                    batch['input_tokens'], 
                    batch['input_vision_masks'],
                    batch['input_delta_masks'],
                    batch['input_action_masks'],
                    deterministic=False,
                    rngs=rng_generator(llama_config.rng_keys()),
                ).logits
                delta_loss, delta_acc = cross_entropy_loss_and_accuracy(
                    delta_logits, 
                    jnp.where(batch['target_delta_masks'], batch['target_tokens'], 0),
                    batch['loss_masks'] * batch['target_delta_masks']
                )
                action_loss, action_acc = cross_entropy_loss_and_accuracy(
                    action_logits, 
                    jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                    batch['loss_masks'] * batch['target_action_masks']
                )
                vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                    vision_logits, 
                    jnp.where(batch['target_vision_masks'], batch['target_tokens'], 0),
                    batch['loss_masks'] * batch['target_vision_masks']
                )
                text_loss, text_acc = cross_entropy_loss_and_accuracy(
                    text_logits, 
                    jnp.where((1.0 - batch["target_vision_masks"]) * (1.0 - batch['target_delta_masks']) * (1.0 - batch['target_action_masks']), batch['target_tokens'], 0),
                    batch['loss_masks'] * (1.0 - batch['target_vision_masks']) * (1.0 - batch['target_delta_masks']) * (1.0 - batch['target_action_masks']),
                )
                loss = action_loss
                
                metrics = dict(
                    vision_loss=vision_loss,
                    vision_acc=vision_acc,
                    text_loss=text_loss,
                    text_acc=text_acc,
                    delta_loss=delta_loss,
                    delta_acc=delta_acc,
                    action_loss=action_loss,
                    action_acc=action_acc,
                )
            else:
                raise ValueError(f"Unsupported modality: {FLAGS.modality}")
            return loss, metrics 
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, loss_metrics), grads = grad_fn(train_state.params)


        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step),
            param_norm=global_norm(train_state.params),
            gradient_norm=global_norm(grads),
            **loss_metrics
        )
        return train_state, rng_generator(), metrics

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp'), 'sp'))
        if FLAGS.modality == 'text':
            logits = model.apply(
                train_state.params, 
                batch['input_tokens'], 
                deterministic=True,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            loss, acc = cross_entropy_loss_and_accuracy(
                logits, 
                batch['target_tokens'],
                batch['loss_masks']
            )
            metrics = dict(
                eval_loss=loss,
                eval_acc=acc,
            )
        elif FLAGS.modality == 'vision,text':
            vision_logits, text_logits = model.apply(
                train_state.params, 
                batch['input_tokens'], 
                batch['input_vision_masks'],
                deterministic=True,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                vision_logits, 
                jnp.where(batch['target_vision_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_vision_masks']
            )
            text_loss, text_acc = cross_entropy_loss_and_accuracy(
                text_logits, 
                jnp.where(batch['target_vision_masks'], 0, batch['target_tokens']),
                batch['loss_masks'] * (1.0 - batch['target_vision_masks'])
            )
            loss = text_loss
            metrics = dict(
                eval_loss=loss,
                eval_vision_accuracy=vision_acc,
                eval_vision_loss=vision_loss,
                eval_text_accuracy=text_acc,
                eval_text_loss=text_loss,
            )
        elif FLAGS.modality == 'vision,text,delta':
            vision_logits, text_logits, delta_logits = model.apply(
                train_state.params, 
                batch['input_tokens'], 
                batch['input_vision_masks'],
                batch['input_delta_masks'],
                deterministic=True,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            delta_loss, delta_acc = cross_entropy_loss_and_accuracy(
                delta_logits, 
                jnp.where(batch['target_delta_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_delta_masks']
            )
            vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                vision_logits, 
                jnp.where(batch['target_vision_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_vision_masks']
            )
            text_loss, text_acc = cross_entropy_loss_and_accuracy(
                text_logits, 
                jnp.where((1.0 - batch["target_vision_masks"]) * (1.0 - batch['target_delta_masks']), batch['target_tokens'], 0),
                batch['loss_masks'] * (1.0 - batch['target_vision_masks'] * (1.0 - batch['target_delta_masks']))
            )
            loss = 0.99 * delta_loss + 0.01 * text_loss 
            # loss = delta_loss
            # TODO: add pheanki reconstruction result for validation
            metrics = dict(
                eval_loss=loss,
                eval_vision_accuracy=vision_acc,
                eval_vision_loss=vision_loss,
                eval_text_accuracy=text_acc,
                eval_text_loss=text_loss,
                eval_delta_accuracy=delta_acc,
                eval_delta_loss=delta_loss,
            )
        elif FLAGS.modality == 'vision,action':
            vision_logits, text_logits, action_logits = model.apply(
                train_state.params, 
                batch['input_tokens'], 
                batch['input_vision_masks'],
                batch['input_action_masks'],
                deterministic=True,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            action_loss, action_acc = cross_entropy_loss_and_accuracy(
                action_logits, 
                jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_action_masks']
            )
            vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                vision_logits, 
                jnp.where(batch['target_vision_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_vision_masks']
            )
            text_loss, text_acc = cross_entropy_loss_and_accuracy(
                text_logits, 
                jnp.where((1.0 - batch["target_vision_masks"]) * (1.0 - batch['target_action_masks']), batch['target_tokens'], 0),
                batch['loss_masks'] * (1.0 - batch['target_vision_masks'] * (1.0 - batch['target_action_masks']))
            )
            loss = action_loss
            action_l1_loss = l1_loss(
                action_logits, 
                jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_action_masks']
            )
            metrics = dict(
                eval_loss=loss,
                eval_vision_accuracy=vision_acc,
                eval_vision_loss=vision_loss,
                eval_action_accuracy=action_acc,
                eval_action_loss=action_loss,
                eval_text_accuracy=text_acc,
                eval_text_loss=text_loss,
                eval_action_l1_loss=action_l1_loss,
            )
        elif FLAGS.modality == 'vision,action,delta':
            vision_logits, text_logits, delta_logits, action_logits = model.apply(
                train_state.params, 
                batch['input_tokens'], 
                batch['input_vision_masks'],
                batch['input_delta_masks'],
                batch['input_action_masks'],
                deterministic=True,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            delta_loss, delta_acc = cross_entropy_loss_and_accuracy(
                delta_logits, 
                jnp.where(batch['target_delta_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_delta_masks']
            )
            action_loss, action_acc = cross_entropy_loss_and_accuracy(
                action_logits, 
                jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_action_masks']
            )
            vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                vision_logits, 
                jnp.where(batch['target_vision_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_vision_masks']
            )
            text_loss, text_acc = cross_entropy_loss_and_accuracy(
                text_logits, 
                jnp.where((1.0 - batch["target_vision_masks"]) * (1.0 - batch['target_delta_masks']) * (1.0 - batch["target_action_masks"]), batch['target_tokens'], 0),
                batch['loss_masks'] * (1.0 - batch['target_vision_masks']) * (1.0 - batch['target_delta_masks']) * (1.0 - batch['target_action_masks']),
            )
            loss = action_loss

            action_l1_loss = l1_loss(
                action_logits, 
                jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_action_masks']
            )


            # TODO: add pheanki reconstruction result for validation
            metrics = dict(
                eval_loss=loss,
                eval_vision_accuracy=vision_acc,
                eval_vision_loss=vision_loss,
                eval_text_accuracy=text_acc,
                eval_text_loss=text_loss,
                eval_delta_accuracy=delta_acc,
                eval_delta_loss=delta_loss,
                eval_action_accuracy=action_acc,
                eval_action_loss=action_loss,
                eval_action_l1_loss=action_l1_loss,
            )
        return rng_generator(), metrics
    
    def unseen_eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp'), 'sp'))
        if FLAGS.modality == 'text':
            logits = model.apply(
                train_state.params, 
                batch['input_tokens'], 
                deterministic=True,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            loss, acc = cross_entropy_loss_and_accuracy(
                logits, 
                batch['target_tokens'],
                batch['loss_masks']
            )
            metrics = dict(
                eval_loss=loss,
                eval_acc=acc,
            )
        elif FLAGS.modality == 'vision,action':
            vision_logits, text_logits, action_logits = model.apply(
                train_state.params, 
                batch['input_tokens'], 
                batch['input_vision_masks'],
                batch['input_action_masks'],
                deterministic=True,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            action_loss, action_acc = cross_entropy_loss_and_accuracy(
                action_logits, 
                jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_action_masks']
            )
            loss = action_loss

            action_l1_loss = l1_loss(
                action_logits, 
                jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_action_masks']
            )

            metrics = dict(
                unseen_eval_loss=loss,
                unseen_eval_action_accuracy=action_acc,
                unseen_eval_action_loss=action_loss,
                unseen_eval_action_l1_loss=action_l1_loss,
            )
        elif FLAGS.modality == 'vision,action,delta':
            vision_logits, text_logits, delta_logits, action_logits = model.apply(
                train_state.params, 
                batch['input_tokens'], 
                batch['input_vision_masks'],
                batch['input_delta_masks'],
                batch['input_action_masks'],
                deterministic=True,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            action_loss, action_acc = cross_entropy_loss_and_accuracy(
                action_logits, 
                jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_action_masks']
            )
            loss = action_loss

            action_l1_loss = l1_loss(
                action_logits, 
                jnp.where(batch['target_action_masks'], batch['target_tokens'], 0),
                batch['loss_masks'] * batch['target_action_masks']
            )

            metrics = dict(
                unseen_eval_loss=loss,
                unseen_eval_action_accuracy=action_acc,
                unseen_eval_action_loss=action_loss,
                unseen_eval_action_l1_loss=action_l1_loss,
            )
        return rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        config_cls.get_partition_rules(llama_config.scan_layers, llama_config.param_scan_axis), train_state_shapes
    )

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    if FLAGS.use_data_sharded_loader:
        batch_spec = PS(('dp', 'fsdp'), 'sp')
    else:
        batch_spec = PS()
    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), batch_spec),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), batch_spec),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    sharded_unseen_eval_step = pjit(
        unseen_eval_step,
        in_shardings=(train_state_partition, PS(), batch_spec),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def load_checkpoint(path, target=None, shard_fns=None, remove_dict_prefix=None, max_buffer_size=0):
        if shard_fns is not None:
            shard_fns = flatten_dict(
                to_state_dict(shard_fns)
            )
        if remove_dict_prefix is not None:
            remove_dict_prefix = tuple(remove_dict_prefix)
        flattend_train_state = {}
        with open_file(path) as fin:
            # 83886080 bytes = 80 MB, which is 16 blocks on GCS
            unpacker = msgpack.Unpacker(fin, read_size=83886080, max_buffer_size=max_buffer_size)
            for key, value in unpacker:
                key = tuple(key)
                if remove_dict_prefix is not None:
                    if key[:len(remove_dict_prefix)] == remove_dict_prefix:
                        key = key[len(remove_dict_prefix):]
                    else:
                        continue
                tensor = from_bytes(None, value)
                if shard_fns is not None:
                    tensor = shard_fns[key](tensor)
                flattend_train_state[key] = tensor

        if target is not None:
            flattened_target = flatten_dict(
                to_state_dict(target), keep_empty_nodes=True
            )
            for key, value in flattened_target.items():
                if key not in flattend_train_state and value == empty_node:
                    flattend_train_state[key] = value
                elif key not in flattend_train_state:
                    initializer = jax.nn.initializers.lecun_normal()  # Example initializer
               
                    tensor = initializer(jax.random.PRNGKey(0), value.shape, dtype=value.dtype)
                    flattend_train_state[key] = tensor
                 


        train_state = unflatten_dict(flattend_train_state)
        if target is None:
            return train_state

        return from_state_dict(target, train_state)

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
        )

    with mesh:
        train_state, restored_params = None, None

        if FLAGS.autoresume and tux.check_exists(output_dir):
            logging.info('Found existing output. Resuming model from latest checkpoint...')
            resume_path = f"trainstate::{output_dir}/streaming_train_state"
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                resume_path, train_state_shapes, shard_fns, max_buffer_size=32 * 2 ** 30
            )
        elif FLAGS.load_checkpoint != '':
            params_target = train_state_shapes.params['params']
            params_shard_fns = shard_fns.params['params']
            load_type, load_path = FLAGS.load_checkpoint.split('::', 1)
            train_state = None
            restored_params = None
            if load_type == 'trainstate':
                train_state = load_checkpoint(
                    path=load_path,
                    target=train_state_shapes,
                    shard_fns=shard_fns,
                    max_buffer_size=32 * 2 ** 30,
                )
            elif load_type == 'trainstate_params':
                # Load the params part of the train state in the streaming format
                restored_params = load_checkpoint(
                    path=load_path,
                    target=params_target,
                    shard_fns=params_shard_fns,
                    remove_dict_prefix=('params', 'params'),
                    max_buffer_size=32 * 2 ** 30,
                )
                restored_params = flax.core.frozen_dict.freeze(
                    {'params': restored_params}
                )
            elif load_type == 'params':
                # Load the params in the streaming format
                restored_params = load_checkpoint(
                    path=load_path,
                    target=params_target,
                    shard_fns=params_shard_fns,
                    max_buffer_size=32 * 2 ** 30
                )
                restored_params = flax.core.frozen_dict.freeze(
                    {'params': restored_params}
                )

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        start_step = int(jax.device_get(train_state.step))

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)
        for step, (batch, dataset_metrics) in zip(step_counter, dataset):
            train_state, sharded_rng, metrics = sharded_train_step(
                train_state, sharded_rng, batch 
            )
            if step % FLAGS.log_freq == 0:
                if FLAGS.eval_steps > 0 and step % FLAGS.eval_log_freq == 0:
                    eval_metric_list = []
                    for _ in range(FLAGS.eval_steps):
                        eval_batch, _ = next(eval_iterator)
                        sharded_rng, eval_metrics = sharded_eval_step(
                            train_state, sharded_rng, eval_batch
                        )
                        eval_metrics = jax.device_get(eval_metrics)
                        eval_batch, _ = next(unseen_eval_iterator)
                        sharded_rng, eval_metrics2 = sharded_unseen_eval_step(
                            train_state, sharded_rng, eval_batch
                        )
                        eval_metrics2 = jax.device_get(eval_metrics2)
                        # concat two dict 
                        eval_metrics.update(eval_metrics2)
                        eval_metric_list.append(eval_metrics)
                    metrics.update(average_metrics(eval_metric_list))

                log_metrics = {"step": step}
                log_metrics.update(metrics)
                log_metrics.update(dataset_metrics)
                log_metrics = jax.device_get(log_metrics)
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                save_checkpoint(train_state)

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)


if __name__ == "__main__":
    run(main)
