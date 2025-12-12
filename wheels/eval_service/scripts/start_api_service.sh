#!/bin/bash
set -x
# 1. begin ray server
host=0.0.0.0
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port --tool_type "search_retrieval" --workers_per_tool 8 --done_if_invalid True --slient True &
server_pid=$!
echo "Server (pid=$server_pid) started at $tool_server_url"

# 2. start api service
model_path=/root/aliyunpan-v0.3.7-linux-amd64/Downloads/9e25f2886bc841b7a99b478648e2393e/117-2/huggingface
max_turns=5 # defines the maximum number of interaction turns between the model and the tool server, if set to 0, it will not limit the number of turns
min_turns=0 # defines the minimum number of action turns between the model and the tool server, will force the model to call tools at least this many times, if set to 0, it will not force the model to call tools
api_host="0.0.0.0"
api_port=5001
action_stop_tokens="</search>,</answer>" # stop at this token, then send the output to the tool server, this is a special token that we use to indicate the end of the action, you can change it to any other token that your model will produce when it is asking for a tool calling round
tensor_parallel_size=1
num_models=1 # number of vllm instances; num_models * tensor_parallel_size should be equal to the number of GPUs
enable_mtrl=False # whether to evaluatoin in multi-chat-turn setting (taken each observation as a new chat turn)
# temp file for action tokens as verl cannot pass special strs as params
action_stop_tokens_file=$(mktemp)
echo "$action_stop_tokens" > $action_stop_tokens_file
echo "action_stop_tokens_file=$action_stop_tokens_file"

python eval_service/app.py \
    --host $api_host \
    --port $api_port \
    --tool_server_url $tool_server_url \
    --model $model_path \
    --max_turns $max_turns \
    --min_turns $min_turns \
    --action_stop_tokens $action_stop_tokens_file \
    --tensor_parallel_size $tensor_parallel_size \
    --num_models $num_models \
    --enable_mtrl $enable_mtrl 

api_server_pid=$!
echo "API started at $api_host:$api_port"

# 3. kill all server
#pkill -9 -P $server_pid
#kill -9 $kill $server_pid
#pkill -9 -P $api_server_pid
#kill -9 $kill $api_server_pid
