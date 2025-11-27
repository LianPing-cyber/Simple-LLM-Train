#!/bin/bash
# git change the source
git config --global url."https://mirrors.aliyun.com/git/".insteadOf https://github.com/
git clone https://github.com/hiyouga/LLaMA-Factory.git

# pip change the source
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install -r LLaMA-Factory/requirements.txt
pip install openai