import argparse
from typing import Optional
import numpy as np
from PIL import Image
from latent_pretraining.sampler_latent_pretrain import DeltaSampler
from latent_pretraining.delta_llama import VideoLLaMAConfig

from tux import JaxDistributedConfig, set_random_seed



class FLAGSClass:
    def __init__(self, flag_dict):
        for key, value in flag_dict.items():
            setattr(self, key, value)

class LAPAInference:
    def __init__(
        self,
        image_size: int = 256,
        **kwargs,
    ) -> None:
        flags = FLAGSClass(kwargs)

        self.model = DeltaSampler(FLAGS=flags)
        self.image_size = image_size
        self.tokens_per_delta = kwargs['tokens_per_delta']
        self.task_description = None

    def inference(self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        assert image.dtype == np.uint8
        image = Image.fromarray(image)
        prompts = [{'image': [image], 'question': task_description}]
        
        latent_output = self.model(prompts)
        latent_action = latent_output[0]

        return latent_action


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAPA Inference script")
    parser.add_argument('--tokens_per_delta', type=int, default=4, help='Tokens per delta')
    parser.add_argument('--vqgan_checkpoint', type=str, default="lapa_checkpoints/vqgan")
    parser.add_argument('--vocab_file', type=str, default='lapa_checkpoints/tokenizer.model')
    parser.add_argument('--multi_image', type=int, default=1)
    parser.add_argument('--jax_distributed', type=dict, default=JaxDistributedConfig.get_default_config())
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--mesh_dim', type=str, default="1,-1,1,1")
    parser.add_argument('--dtype', type=str, default="bf16")
    parser.add_argument('--load_llama_config', type=str, default="7b")
    parser.add_argument('--update_llama_config', type=str, default="dict(delta_vocab_size=8,sample_mode='text',theta=50000000,max_sequence_length=32768,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)")
    parser.add_argument('--load_checkpoint', type=str, default="params::lapa_checkpoints/params")
    parser.add_argument('--codebook_size', type=int, default=8)


    # Add more arguments as needed
    generated_images=[]
    
    args = parser.parse_args()

    args.tokenizer = VideoLLaMAConfig.get_tokenizer_config()
    args.llama = VideoLLaMAConfig.get_default_config()
    args.tokenizer.vocab_file = args.vocab_file

    
    
    JaxDistributedConfig.initialize(args.jax_distributed)
    set_random_seed(args.seed)

    lapa = LAPAInference(
        image_size=256,
        tokens_per_delta=args.tokens_per_delta,
        vqgan_checkpoint=args.vqgan_checkpoint,
        vocab_file=args.vocab_file,
        multi_image=args.multi_image,
        jax_distributed=args.jax_distributed,
        seed=args.seed,
        mesh_dim=args.mesh_dim,
        dtype=args.dtype,
        load_llama_config=args.load_llama_config,
        update_llama_config=args.update_llama_config,
        load_checkpoint=args.load_checkpoint,
        tokenizer=args.tokenizer,
        llama=args.llama
    )


    image = 'imgs/bridge_inference.jpg'
    instruction ="take the broccoli out of the pot"
    
    image = Image.open(image)
    image = np.array(image)

    
    latent_action = lapa.inference(image, instruction)
    print("latent action is", latent_action)
