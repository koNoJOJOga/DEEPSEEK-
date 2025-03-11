# 在单个T4上对Qwen2.5 0.5B进行完整的GRPO微调

原版代码为了让这些代码能在colab上运行做了很多调整，使GRPO完整微调Qwen2.5-0.5-Instruct能够在单个T4 GPU上运行，因此可以在免费的Google Colab中运行。使用VLLm进行快速推理，并且在批处理和完成组大小上不做妥协。

通过这种设置，可以在单个T4 GPU上仅用约150步（约30分钟）就将Qwen2.5-0.5B-Instruct的gsm8k评估结果从22.4%提高到48.6%。

但是我的实现`train-checkpoint-900.ipynb`是没有使用vllm的
```python
training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=16,
    max_prompt_length=256,
    max_completion_length=200,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=False,
    vllm_gpu_memory_utilization=.3,
    vllm_device="cuda:0",
    report_to="none"
)
```

以下是使用的一些重要优化：

* 由andyl98创建的TRL仓库分支，引入了批量logprobs计算。我进一步分叉并优化了logprobs计算函数以减少VRAM使用。
* 8位AdamW优化器
* 通过`PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128'`设置明确的内存分配限制

如果使用Ampere或更新架构的NVIDIA GPU，可以通过以下方式进一步减少VRAM使用：

* 在模型加载期间启用`attn_implementation="flash_attention_2"`
* 使用Liger-Kernel包装器加载模型：

```python
from liger_kernel.transformers import AutoLigerKernelForCausalLM
model = AutoLigerKernelForCausalLM.from_pretrained("path/to/some/model")
```

```
%%capture
!pip install uv
!uv pip install --system git+https://github.com/qunash/trl-1.git@grpo-vram-optimization
!uv pip install --system triton==2.2.0
!uv pip install --system vllm
!uv pip install --system bitsandbytes
```

## 参考资料

https://gist.github.com/qunash/820c86d1d267ec8051d9f68b4f4bb656

https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb?permalink_comment_id=5417630