# borrowed from https://medium.com/data-science-in-your-pocket/tested-nvidia-h200-vs-h100-gpus-for-ai-the-winner-will-surprise-you-4bbf0b3cd62e

import os
import platform
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, modeling_utils

# Configure parallel processing styles
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

# Model configuration
model_name = "Qwen/Qwen3-0.6B"

# Try to import TPU support
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.debug.metrics as met
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

# Try to import Ascend NPU support
try:
    import torch_npu
    NPU_AVAILABLE = torch_npu.npu.is_available()
except ImportError:
    NPU_AVAILABLE = False

# Check for Intel GPU support
try:
    import intel_extension_for_pytorch as ipex
    if platform.system() == "Windows":
        # Windows specific check for Intel GPU
        import ctypes
        try:
            ctypes.WinDLL("igdrcl64.dll")
            INTEL_GPU_AVAILABLE = True
        except OSError:
            INTEL_GPU_AVAILABLE = False
    else:
        # Linux check for Intel GPU
        INTEL_GPU_AVAILABLE = os.path.exists("/dev/dri/renderD128")
except ImportError:
    INTEL_GPU_AVAILABLE = False

# Determine the device (Discrete GPU > TPU > NPU > Intel GPU > CPU)
if torch.cuda.is_available():
    device = "cuda"
    print(f"Using NVIDIA GPU device")
elif TPU_AVAILABLE:
    device = "xla"
    print(f"Using Google Cloud TPU device")
elif NPU_AVAILABLE:
    device = "npu"
    print(f"Using Huawei Ascend NPU device")
elif INTEL_GPU_AVAILABLE:
    device = "xpu"
    print(f"Using Intel GPU device")
else:
    device = "cpu"
    print(f"Using CPU device")

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Handle model loading based on device type
if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
elif device == "xla":
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    )
    # Move model to TPU and compile for TPU execution
    device = xm.xla_device()
    model = model.to(device)
    model = torch.compile(model)  # Compile model for TPU
elif device == "npu":
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    )
    # Move model to NPU
    model = model.to("npu")
elif device == "xpu":
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    )
    # Move model to Intel GPU and optimize
    model = model.to("xpu")
    model = ipex.optimize(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=None
    )
    model = model.to(device)

# prepare the model input
prompt = "Give me a short introduction to large language model.keep it very short, 1 line only"
messages = [
    {"role": "user", "content": prompt}
]
print(f"messages: {messages}")
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)

# Handle input tensors based on device type
if device == "xla":
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
elif device == "npu":
    model_inputs = tokenizer([text], return_tensors="pt").to("npu")
elif device == "xpu":
    model_inputs = tokenizer([text], return_tensors="pt").to("xpu")
else:
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

# conduct text completion
start = time.time()
n_iter = 10
generated_ids = None

if device == "xla":
    # TPU-specific execution
    for x in range(n_iter):
        # Clear TPU cache if needed
        if x > 0:
            torch_xla._XLAC._xla_gc_executor(True)
        
        # Generate with TPU optimization
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100
        )
        # Ensure TPU operations are completed efficiently
        xm.mark_step()
        if x < n_iter - 1:  # Don't wait on the last iteration
            xm.wait_device_ops()
else:
    # Non-TPU execution
    for x in range(n_iter):
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100
        )
        if device == "npu":
            # Ensure NPU operations are completed
            torch.npu.synchronize()
        elif device == "xpu":
            # Ensure Intel GPU operations are completed
            torch.xpu.synchronize()

end = time.time()

print(f"Time taken: {end - start:.2f} seconds with {n_iter} iterations on {device}")

# Print device-specific information
if device == "cuda":
    print(f"GPU type: {torch.cuda.get_device_name(0)}")
elif device == "xla":
    print(f"TPU type: {xm.xla_device()}")
    # Print TPU metrics for debugging
    print("\nTPU Metrics:")
    print(met.metrics_report())
elif device == "npu":
    print(f"NPU type: {torch_npu.npu.get_device_name()}")
elif device == "xpu":
    print(f"Intel GPU type: {torch.xpu.get_device_name()}")
else:
    print(f"CPU threads: {torch.get_num_threads()}")

# Print the prompt and the final response
print("\nPrompt:")
print(prompt)
print("\nResponse:")
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)


