DATA="xxxx.json"
TASK_NAME="test"
DO_TRAIN=1
CONTINUE_TRAIN=0

ORIGINAL_MODEL_FOLDER="xxxxx"
OUTPUT_MODEL_FOLDER="xxxxx"
EPOCH_NUM=2
CUTOFF_LEN=256
TRAIN_STAGE="sft"
FINETUNING_TYPE="lora"

TRAIN_NUM=200
EVAL_NUM=200
EVAL_BATCH_SIZE=64
TRAIN_TEMPLATE="empty"
EVAL_TEMPLATE="empty"
EVAL_OUTPUT_LENGTH=1024

LF_PATH="~/LLaMA-Factory"
LF_DATA_DIR="train_data"
BASE_URL="your_base_url"
API_KEY="your_api_key"

PER_DEVICE_TRAIN_BATCH_SIZE=4
NPROC_PER_NODE=1
LEARNING_RATE=1e-4

python -u src/main.py \
    --data $DATA \
    --task_name $TASK_NAME \
    --do_train $DO_TRAIN \
    --continue_train $CONTINUE_TRAIN \
    --train_stage $TRAIN_STAGE \
    --original_model_folder $ORIGINAL_MODEL_FOLDER \
    --output_model_folder $OUTPUT_MODEL_FOLDER \
    --epoch_num $EPOCH_NUM \
    --cutoff_len $CUTOFF_LEN \
    --train_stage $TRAIN_STAGE \
    --finetuning_type $FINETUNING_TYPE \
    --train_num $TRAIN_NUM \
    --eval_num $EVAL_NUM \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --train_template $TRAIN_TEMPLATE \
    --eval_template $EVAL_TEMPLATE \
    --eval_output_length $EVAL_OUTPUT_LENGTH \
    --lf_path $LF_PATH \
    --lf_data_dir $LF_DATA_DIR \
    --base_url $BASE_URL \
    --api_key $API_KEY \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --learning_rate $LEARNING_RATE  2>&1 | tee logs/$TASK_NAME.log
