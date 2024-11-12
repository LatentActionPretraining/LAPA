gpu_id=0
declare -a policy_models=(
  "lapa"
)

scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028



ckpt_path="params::"
  
for policy_model in "${policy_models[@]}"; do
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_lapa.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
    --robot ${robot} --policy-setup widowx_bridge --action-scale-file ../data/simpler.csv\
    --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
    --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name ${scene_name} \
    --rgb-overlay-path ${rgb_overlay_path} \
    --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24\
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --load-llama-config 7b --update-llama-config "dict(action_vocab_size=245,sample_mode='text',theta=50000000,max_sequence_length=32768,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)" \
    --vocab-file ../lapa_checkpoints/tokenizer.model \
    --tokens-per-action 7 --tokens-per-delta 4

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_lapa.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
    --robot ${robot} --policy-setup widowx_bridge --action-scale-file ../data/simpler.csv\
    --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
    --env-name PutCarrotOnPlateInScene-v0 --scene-name ${scene_name} \
    --rgb-overlay-path ${rgb_overlay_path} \
    --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24\
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --load-llama-config 7b --update-llama-config "dict(action_vocab_size=245,sample_mode='text',theta=50000000,max_sequence_length=32768,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)" \
    --vocab-file ../lapa_checkpoints/tokenizer.model \
    --tokens-per-action 7 --tokens-per-delta 4

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_lapa.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
    --robot ${robot} --policy-setup widowx_bridge --action-scale-file ../data/simpler.csv\
    --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
    --env-name PutSpoonOnTableClothInScene-v0 --scene-name ${scene_name} \
    --rgb-overlay-path ${rgb_overlay_path} \
    --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24\
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --load-llama-config 7b --update-llama-config "dict(action_vocab_size=245,sample_mode='text',theta=50000000,max_sequence_length=32768,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)" \
    --vocab-file ../lapa_checkpoints/tokenizer.model \
    --tokens-per-action 7 --tokens-per-delta 4
    
done



scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path=ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
robot_init_x=0.127
robot_init_y=0.06


ckpt_path="params::"
  
  for policy_model in "${policy_models[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_lapa.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
      --robot ${robot} --policy-setup widowx_bridge --action-scale-file ../data/simpler.csv\
      --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
      --env-name PutEggplantInBasketScene-v0 --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24\
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --load-llama-config 7b --update-llama-config "dict(action_vocab_size=245,sample_mode='text',theta=50000000,max_sequence_length=32768,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)" \
      --vocab-file ../lapa_checkpoints/tokenizer.model \
      --tokens-per-action 7 --tokens-per-delta 4
  done
done