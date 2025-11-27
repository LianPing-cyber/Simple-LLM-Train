LOG=000

DATA="mmlu.json"
DATA_TYPE="content"
TASK_NAME="iclr"

DO_TRAIN=1
DATA_BALANCE=0
RANDOM_SAMPLE=1

EPOCH_NUM=2
DATA_NUM=10
EVAL_NUM=10

ORIGINAL_MODEL_FOLDER=""
OUTPUT_MODEL_FOLDER=""

LF_PATH="~/llama_factory"
LF_DATA_DIR="train_data"

PER_DEVICE_TRAIN_BATCH_SIZE=4
NPROC_PER_NODE=1
FINETUNING_TYPE="lora"
LEARNING_RATE=1e-4

python -u src/main.py \
    --data $DATA \
    --data_type $DATA_TYPE \
    --task_name $TASK_NAME \
    --do_train $DO_TRAIN \
    --data_balance $DATA_BALANCE \
    --random_sample $RANDOM_SAMPLE \
    --epoch_num $EPOCH_NUM \
    --data_num $DATA_NUM \
    --eval_num $EVAL_NUM \
    --original_model_folder $ORIGINAL_MODEL_FOLDER \
    --output_model_folder $OUTPUT_MODEL_FOLDER \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --finetuning_type $FINETUNING_TYPE \
    --nproc_per_node $NPROC_PER_NODE \
    --lf_path $LF_PATH \
    --lf_data_dir $LF_DATA_DIR 2>&1 | tee logs/$LOG.log


