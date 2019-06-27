#!/bin/bash
declare -A MODEL_PATHS
MODEL_PATHS["0.0"]="/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.0_Run0/checkpoint/torcs_remi/torcs_remi_1367"
MODEL_PATHS["0.1"]="/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.1_Run2/checkpoint/torcs_remi/torcs_remi_2828"
MODEL_PATHS["0.2"]="/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.2_Run2/checkpoint/torcs_remi/torcs_remi_5463"
MODEL_PATHS["0.3"]="/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.3_Run2/checkpoint/torcs_remi/torcs_remi_3248"
MODEL_PATHS["0.4"]="/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.4_Run1/checkpoint/torcs_remi/torcs_remi_1256"
MODEL_PATHS["0.6"]="/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.6_Run2/checkpoint/torcs_remi/torcs_remi_2461"
MODEL_PATHS["0.7"]="/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.7_Run2/checkpoint/torcs_remi/torcs_remi_2747"
MODEL_PATHS["0.8"]="/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.8_Run2/checkpoint/torcs_remi/torcs_remi_3144"
MODEL_PATHS["0.9"]="/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.9_Run2/checkpoint/torcs_remi/torcs_remi_3981"
MODEL_PATHS["1.0"]="/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_1.0_Run0/checkpoint/torcs_remi/torcs_remi_4339"

declare -a ALPHA_VALS=( "0.0" "0.1" "0.2" "0.3" "0.4" "0.6" "0.7" "0.8" "0.9" "1.0")
# declare -a ALPHA_VALS=( "0.0" "0.1")
# declare -p ALPHA_VALS

# declare -p MODEL_PATHS

for ALPHA_VAL in ${ALPHA_VALS[@]}; do
  # First RUn the record scripts
  # echo ${MODEL_PATHS[$ALPHA_VAL]}
  xvfb-run -a -s "-screen $DISPLAY 640x480x24" python -m baselines.remi.play --load_model_path=${MODEL_PATHS[$ALPHA_VAL]}
  # Move the records raws somewhere
  ALPHA_VAL_DIR="/home/z3r0/torcs_data/""${ALPHA_VAL}"
  if [ ! -d $ALPHA_VAL_DIR ]; then
    mkdir ${ALPHA_VAL_DIR}
  fi
  mv /home/z3r0/torcs_data/{obs,acs,rews}.csv -t "${ALPHA_VAL_DIR}""/."
done
