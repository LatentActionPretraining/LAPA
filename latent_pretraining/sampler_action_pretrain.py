from functools import cached_property
import numpy as np
import jax
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from transformers import GenerationConfig
from tux import (
    define_flags_with_default, StreamingCheckpointer, JaxDistributedConfig,
    set_random_seed, get_float_dtype_by_name, JaxRNG, next_rng,
    match_partition_rules, make_shard_and_gather_fns,
    with_sharding_constraint, tree_apply, open_file
)
from latent_pretraining.llama_action import VideoLLaMAConfig, FlaxVideoLLaMAForCausalLM 
from latent_pretraining.vqgan import VQGAN
import albumentations
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2

class CenterCropThenResize(ImageOnlyTransform):
    def __init__(self, height, width, scale, ratio, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0):
        super(CenterCropThenResize, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def apply(self, img, **params):
        # Calculate the size of the crop
        crop_height = int(self.height * self.scale[0])
        crop_width = int(self.width * self.ratio[0])

        # Calculate center crop coordinates
        y1 = (img.shape[0] - crop_height) // 2
        x1 = (img.shape[1] - crop_width) // 2
        y2 = y1 + crop_height
        x2 = x1 + crop_width

        # Crop and resize
        cropped_img = img[y1:y2, x1:x2]
        return A.Resize(self.height, self.width).apply(cropped_img, interpolation=self.interpolation)


class ActionSampler:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.mesh = VideoLLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
        self.vqgan = VQGAN(FLAGS.vqgan_checkpoint, replicate=False)
        self.prefix_tokenizer = VideoLLaMAConfig.get_tokenizer(
            FLAGS.tokenizer, truncation_side='left', padding_side='left'
        )
        self.tokenizer = VideoLLaMAConfig.get_tokenizer(FLAGS.tokenizer)
        self.min_buffer_size = 256
        self.sharded_rng = next_rng()
        self._load_model()


    @property
    def block_size(self):
        return max(self.config.scan_query_chunk_size, self.config.scan_key_chunk_size) * self.mesh.shape['sp']
    
    @property
    def data_dim(self):
        return self.mesh.shape['dp'] * self.mesh.shape['fsdp']

    def _process_frame(self, images, image_aug=False):
        if image_aug:
            preprocessor_finetune = albumentations.Compose([
                    albumentations.LongestMaxSize(max_size=256),  # Resize the longest side to 256
                    albumentations.Resize(256, 256), 
                    CenterCropThenResize(height=256, width=256, scale=[0.9, 0.9], ratio=[1.0, 1.0])
                ])
        else:
            preprocessor_finetune = albumentations.Compose([
                    albumentations.LongestMaxSize(max_size=256),  # Resize the longest side to 256
                    albumentations.Resize(256, 256),
                ])
        image_vqgan_list = []
        for image in images:
            img_array = np.array(image).astype(np.uint8)
                
            image_vqgan = preprocessor_finetune(image=img_array)["image"]
            image_vqgan = (image_vqgan/127.5 - 1.0).astype(np.float32)
            image_vqgan_list.append(image_vqgan[None])
        image_vqgan_list = np.concatenate(image_vqgan_list, axis=0)
        return image_vqgan_list


    def _read_process_vision(self, images, image_aug=False):

        vision = self._process_frame(images, image_aug)
        
        B = 1
        encodings = []
        for i in range(0, len(vision), 1):
            v = vision[i:i + B]
            if len(v) % B == 0:
                n_pad = 0
            else:
                n_pad = B - len(v) % B
            v = np.pad(v, ((n_pad, 0), (0, 0), (0, 0), (0, 0)))
            enc = jax.device_get(self.vqgan.encode(v))[1].astype(int)
            enc = enc[n_pad:]
            for t in range(len(enc)):
                encodings.extend(enc[t].reshape(-1).tolist())
        return encodings



    def construct_input(self, prompts):
        for i, prompt in enumerate(prompts):
            vision = self._read_process_vision(prompt['image'], image_aug=self.FLAGS.image_aug)
            tokens, vm = [], []
            tokens.extend(vision)
            vm.extend([True] * len(vision))
            tokens.extend([8193])
            vm.extend([True] * len([8193]))
        return {
            'input_ids': np.expand_dims(tokens, axis=0),
        }
             

    def _load_model(self):
        if self.FLAGS.load_llama_config != '':
            llama_config = VideoLLaMAConfig.load_config(self.FLAGS.load_llama_config)
            updates = VideoLLaMAConfig(**self.FLAGS.llama)
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
            llama_config = VideoLLaMAConfig(**self.FLAGS.llama)




        if self.FLAGS.update_llama_config != '':
            llama_config.update(dict(eval(self.FLAGS.update_llama_config)))

        llama_config.update(dict(
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        ))
        llama_config.update(dict(mesh_dim=self.FLAGS.mesh_dim))
        self.config = llama_config

        with jax.default_device(jax.devices("cpu")[0]):
            _, self.params = StreamingCheckpointer.load_trainstate_checkpoint(
                    self.FLAGS.load_checkpoint, disallow_trainstate=True, max_buffer_size=32 * 2 ** 30
            )
            self.model = FlaxVideoLLaMAForCausalLM(
                llama_config, 
                input_shape=(512, 8192), 
                seed=self.FLAGS.seed, 
                _do_init=False,
                dtype=get_float_dtype_by_name(self.FLAGS.dtype),
            )

        
            self.model_ps = match_partition_rules(
                VideoLLaMAConfig.get_partition_rules(llama_config.scan_layers, llama_config.param_scan_axis), self.params
            )
            shard_fns, _ = make_shard_and_gather_fns(
                self.model_ps, get_float_dtype_by_name(self.FLAGS.dtype)
            )

            with self.mesh:
                self.params = tree_apply(shard_fns, self.params)

    @cached_property
    def _forward_generate(self):
        def fn(params, rng, batch, n_tokens):
            batch = with_sharding_constraint(batch, PS(('dp', 'fsdp'), 'sp'))
            rng_generator = JaxRNG(rng)


            self.model.config.sample_mode='action'
            action_output = self.model.generate_vision(
                batch['input_ids'],
                vision_masks=batch['vision_masks'],
                attention_mask=batch['attention_mask'],
                action_masks=batch['action_masks'],
                params=params['params'],
                prng_key=rng_generator(),
                generation_config=GenerationConfig(
                    max_new_tokens=n_tokens,
                    min_new_tokens=n_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            ).sequences
            action_output= action_output[:,batch['input_ids'].shape[1]:]
            return action_output, rng_generator()
        return pjit(
            fn,
            in_shardings=(self.model_ps, PS(), PS()),
            out_shardings=(PS(), PS()),
            static_argnums=(3,)
        )

    
    def generate_video_pred(self, prompts, images, max_input_length):
        
        sharded_rng = next_rng()
        inputs = self.prefix_tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=max_input_length,
            return_tensors='np'
        )
        prefix_for_gen = ["</vision> <action>"] * len(prompts)
        inputs_for_gen = self.prefix_tokenizer(
            prefix_for_gen,
            return_tensors='np'
        )

        batch = dict(
            input_ids=np.concatenate([inputs.input_ids, images, inputs_for_gen.input_ids], axis=1),
            attention_mask=np.concatenate([inputs.attention_mask, np.ones(images.shape, dtype=inputs.attention_mask.dtype), inputs_for_gen.attention_mask], axis=1),
            vision_masks=np.concatenate([
                np.zeros(inputs.input_ids.shape, dtype=bool),
                np.ones(images.shape, dtype=bool),
                np.zeros(inputs_for_gen.input_ids.shape, dtype=bool)
            ], axis=1),
            action_masks=np.concatenate([
                np.zeros(inputs.input_ids.shape, dtype=bool),
                np.zeros(images.shape, dtype=bool),
                np.zeros(inputs_for_gen.input_ids.shape, dtype=bool),
            ], axis=1),
        )

        with self.mesh:
            action_output, sharded_rng = self._forward_generate(
                self.params, sharded_rng, batch, 
                self.FLAGS.tokens_per_action
            )
            action_output = jax.device_get(action_output)
            
        return action_output

    def __call__(self, prompts):
        batch = self.construct_input(prompts)
        text_prompt = f"<s> <s> You are a helpful assistant. USER: What action should the robot take to `{prompts[0]['question']}` ASSISTANT: <vision>"
        action_output = self.generate_video_pred(prompts=[text_prompt], images=batch['input_ids'], max_input_length=128)
        return action_output
        