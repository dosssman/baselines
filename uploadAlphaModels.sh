#!/bin/bash
declare -A MODEL_PATHS
MODEL_PATHS["0.0"]="Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.0_Run0"
MODEL_PATHS["0.1"]="Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.1_Run2"
MODEL_PATHS["0.2"]="Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.2_Run2"
MODEL_PATHS["0.3"]="Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.3_Run2"
MODEL_PATHS["0.4"]="Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.4_Run1"
MODEL_PATHS["0.6"]="Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.6_Run2"
MODEL_PATHS["0.7"]="Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.7_Run2"
MODEL_PATHS["0.8"]="Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.8_Run2"
MODEL_PATHS["0.9"]="Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.9_Run2"
MODEL_PATHS["1.0"]="Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_1.0_Run0"

declare -a ALPHA_VALS=( "0.0" "0.1" "0.2" "0.3" "0.4" "0.6" "0.7" "0.8" "0.9" "1.0")
# declare -a ALPHA_VALS=( "0.0" "0.1")
# declare -p ALPHA_VALS

# declare -p MODEL_PATHS

declare -a SERVER_NAMES=( "cairo" "yeager" "defiant")

for SERVER_NAME in ${SERVER_NAMES[@]}; do
  for ALPHA_VAL in ${ALPHA_VALS[@]}; do
    # First RUn the record scripts
    # echo ${MODEL_PATHS[$ALPHA_VAL]}
    # xvfb-run -a python -m baselines.remi.play --load_model_path=${MODEL_PATHS[$ALPHA_VAL]}
    rsync -ravu "/home/z3r0/random/rl/openai_logs/openai-remi/""${MODEL_PATHS[$ALPHA_VAL]}" $SERVER_NAME:~/random/rl/openai_logs/openai-remi/.
    # Move the records raws somewhere
    # ALPHA_VAL_DIR="/home/z3r0/torcs_data/""${ALPHA_VAL}"
    # if [ ! -d $ALPHA_VAL_DIR ]; then
    #   mkdir ${ALPHA_VAL_DIR}
    # fi
    # mv /home/z3r0/torcs_data/{obs,acs,rews}.csv -t "${ALPHA_VAL_DIR}""/."
  done
done
