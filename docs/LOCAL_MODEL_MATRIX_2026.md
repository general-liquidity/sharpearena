# Local frontier-model matrix (26 August 2026)

This note separates two questions that are easy to conflate:

1. Which downloadable open-weight models are at the frontier, regardless of hardware?
2. Which models can produce a reproducible SharpeArena/SharpeBench field on this workstation?

The host is an NVIDIA RTX A4000 (16,376 MiB, Ampere compute capability 8.6), a 32-core/64-thread Threadripper PRO 5975WX, and 256 GiB of RAM. Model storage can live on `D:`, but storage does not change the memory limit.

All benchmark numbers below are model-publisher results, not a new common-harness comparison. They are useful for identifying candidates; they are not directly rank-comparable across model cards.

## Bottom line

- The strongest downloadable systems are now trillion-parameter sparse models. Kimi K3 and Qwen3.8-2.4T-A95B are the largest of the requested current flagships. Neither can load on this machine, even at an idealized four bits per parameter.
- Thinking Machines' **Inkling is a real open-weight multimodal model**, not a dataset, RL environment, or training framework. It is Apache-2.0, 975B parameters in total and 41B active per token. It is also far too large for 256 GiB RAM.
- The best high-end near-local candidate is **Qwen3.8-27B**. It is Apache-2.0, multimodal and agent-oriented, but the official Ollama Q4 artifact is about 18 GB, so it requires partial CPU offload on a 16 GB A4000.
- The cleanest full-GPU field should use several 8B–14B-class Q4 models. **Ornith-1.5-9B**, **Gemma 4 12B** and **Ministral 3 14B** add useful diversity. **gpt-oss-20b** is a valuable agent/tool comparator and is advertised for 16 GB systems, but is tight enough that context and KV-cache settings must be measured rather than assumed.
- Use **llama.cpp as the native-Windows/offload baseline** and **vLLM under WSL2 as the throughput baseline** for models that fit completely in VRAM. Use SGLang as a cross-engine sensitivity arm. Ollama and LM Studio are convenient wrappers, not invisible experimental infrastructure.

## Memory accounting

Active parameters determine approximate compute per generated token; **total parameters determine resident weight memory**. A 1.6T MoE with 49B active parameters still needs the 1.6T-parameter checkpoint loaded or distributed.

Approximate weight-only floors are:

| Representation | Ideal weight bytes/parameter | Caveat |
|---|---:|---|
| BF16/FP16 | 2.0 | excludes KV cache and runtime workspaces |
| FP8/INT8 | 1.0 | hardware/kernel support varies |
| ideal 4-bit | 0.5 | real GGUF Q4_K_M is commonly about 0.60–0.67 after block metadata |

For this host, a robust full-GPU target is at most about 13–14 GB of weights after leaving room for the runtime and KV cache. A conservative CPU/offload ceiling is about 220 GB, leaving RAM for the OS, mappings, KV cache, and the harness. Native 256K or 1M context claims are not practical local operating points; SharpeArena observations are small enough that an initial 8K–16K cap is preferable.

Ampere supports useful W4A16/AWQ/GPTQ/Marlin paths. It does not provide the native Blackwell NVFP4/MXFP4 path used by several frontier checkpoints, nor the FP8 Tensor Core path available on later NVIDIA generations.

## Absolute frontier: downloadable, but not local

| Model | Weights and license | Scale and context | Agent-relevant properties | A4000 + 256 GB verdict |
|---|---|---|---|---|
| **Kimi K3** | Native QAT MXFP4 weights; custom [Kimi K3 License](https://huggingface.co/moonshotai/Kimi-K3/raw/main/LICENSE), with separate-agreement and branding conditions for specified large commercial services | 2.8T total / 104B active; 1,048,576 context; multimodal text, image, and video | Preserved thinking, selectable reasoning effort, tools and long-horizon agent work; official serving recipes for vLLM, SGLang and TokenSpeed | Ideal Q4 alone is about 1.4 TB: impossible locally |
| **Qwen3.8-2.4T-A95B** | Publisher checkpoint plus official FP8 artifact; custom [Qwen3.8-Max License](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/raw/main/LICENSE), not Apache-2.0 | 2.4T / 95B active; 262,144 native context, documented extension to about 1.01M; text-only open-weight artifact | Reasoning-effort controls, preserved thinking and agent execution; official vLLM/SGLang/TokenSpeed recipes | Ideal Q4 is about 1.2 TB: impossible locally |
| **DeepSeek-V4-Pro-0813** | MIT; official mixed low-precision release and serving instructions | V4-Pro core: 1.6T / 49B active; 1M context. The `0813` repository reports about 1.7T because the attached speculative DSpark module adds parameters | Low/high/max reasoning, tool use and terminal/coding focus; official vLLM/SGLang deployment is multi-GB300 | Ideal Q4 core is about 800 GB: impossible locally |
| **Inkling** | BF16 and official NVFP4 checkpoints; Apache-2.0 | 975B / 41B active; 1,048,576 configured context; multimodal text/image/audio input and text output (architecture also describes a video encoder) | Agent/tool use; official SGLang, vLLM, TokenSpeed, Unsloth and Transformers paths | Ideal Q4 is about 488 GB. NVFP4 does not make it fit and lacks native acceleration on Ampere |
| **GLM-5.2** | MIT; base and official FP8 artifacts | 753B total; approximately 40B active for the GLM-5 core (the 5.2 card does not restate an active total); 1M context | Long-horizon agent and coding focus, flexible effort; vLLM, SGLang, Transformers, KTransformers and Unsloth recipes | Ideal Q4 is about 377 GB: impossible locally |
| **NVIDIA Nemotron 3 Ultra** | Open Model/Data License 1.1; BF16 and lower-precision NVIDIA releases | 550B / 55B active; 1M context; hybrid Mamba-2, attention and MoE | Configurable thinking and agent/tool operation | Ideal Q4 is about 275 GB before overhead; official minimum configurations are multi-datacenter-GPU |
| **Ornith-1.5-397B** | MIT; official BF16, FP8, NVFP4, GGUF and MLX family artifacts | 397B total MoE; active total is not disclosed on the model card; 262K native context, publisher-validated YaRN extension to about 1M | Reasoning and OpenAI-style tool calls; publisher recipes for vLLM, SGLang, llama.cpp and Ollama. Publisher reports Terminal-Bench 2.1 86.1, SWE-bench Verified 86.0 and MCP-Atlas 80.0 | Official Q4_K_M GGUF is 244.3 GB plus runtime/KV/vision projection: beyond the safe local RAM ceiling |
| **DeepSeek-V4-Flash-0731** | MIT; official release | V4-Flash core: 284B / 13B active; repository reports about 304B with DSpark; 1M context | Lower-active-parameter V4 agent/coding model; official examples still use 4×GB300 | An ideal Q4 core is about 142 GB, but the official format, new architecture and kernel path make this experimental CPU/offload work—not a dependable field backend |
| **Kimi K2.6** | Modified MIT-style Kimi license; native INT4 | 1T / 32B active; 256K context; multimodal | Optional/preserved reasoning and multi-step tools; vLLM/SGLang recipes | Roughly 500 GB at INT4: impossible locally and superseded by K3 for frontier comparison |

Primary model cards: [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3), [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B), [DeepSeek-V4-Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813), [DeepSeek-V4-Pro core](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro), [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), [Inkling](https://huggingface.co/thinkingmachines/Inkling), [Inkling-NVFP4](https://huggingface.co/thinkingmachines/Inkling-NVFP4), [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2), [Nemotron 3 Ultra](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16), [Ornith-1.5-397B](https://huggingface.co/ornith-ai/Ornith-1.5-397B), and [Kimi K2.6](https://huggingface.co/moonshotai/Kimi-K2.6).

Selected publisher-reported anchors explain why these models belong in the frontier set; they do not form a common leaderboard:

| Model | Selected model-card results |
|---|---|
| Kimi K3 | GPQA Diamond 93.5, Terminal-Bench 2.1 88.3, BrowseComp 91.2, Finance Agent v2 54.4 |
| Qwen3.8-2.4T-A95B | Terminal-Bench 2.1 86.6, SWE-bench Pro 67.7, PaperBench 93.0 |
| DeepSeek-V4-Pro-0813 | Terminal-Bench 2.1 87.9, HLE 42.7 without tools / 60.0 with tools, Toolathlon-Verified 74.1 |
| DeepSeek-V4-Flash-0731 | Terminal-Bench 2.1 82.7 |
| GLM-5.2 | Terminal-Bench 2.1 81.0 in the matched setting / 82.7 best reported, SWE-bench Pro 62.1, MCP-Atlas 76.8 |
| Inkling | SWE-bench Verified 77.6, best reported Terminal-Bench 2.1 63.8, HLE 29.7 without tools / 46.0 with tools, MCP-Atlas 74.1 |
| Ornith-1.5-397B | Terminal-Bench 2.1 86.1, SWE-bench Verified 86.0, DeepSWE 56.0, MCP-Atlas 80.0 |

### The requested Hugging Face links are not equivalent objects

- [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) and [`thinkingmachines/Inkling`](https://huggingface.co/thinkingmachines/Inkling) are model repositories.
- [`Qwen/qwen38`](https://huggingface.co/collections/Qwen/qwen38) is a four-item collection: 2.4T-A95B, its FP8 artifact, 27B, and its FP8 artifact.
- [`zai-org/glm-52`](https://huggingface.co/collections/zai-org/glm-52) is a two-item collection: GLM-5.2 and its FP8 artifact.
- [`deepseek-ai/deepseek-v4`](https://huggingface.co/collections/deepseek-ai/deepseek-v4) is a changing release collection containing preview, base, DSpark and dated production artifacts. A benchmark record must name the exact repository and revision; “DeepSeek V4” is insufficient.
- [`ornith-ai/ornith-15`](https://huggingface.co/collections/ornith-ai/ornith-15) is a release collection, not one model. It contains 9B dense, 35B-A3B MoE and 397B MoE base artifacts plus publisher GGUF, FP8, NVFP4 and MLX variants where applicable.
- [`google/gemma-4`](https://huggingface.co/collections/google/gemma-4) is a family collection. Its strongest instruction checkpoint is the 30.7B dense `gemma-4-31B-it`; 12B and 26B-A4B are deployment alternatives, not the family frontier.

### What Inkling is

Inkling's [model card](https://huggingface.co/thinkingmachines/Inkling) and [configuration](https://huggingface.co/thinkingmachines/Inkling/blob/main/config.json) identify a decoder MoE model with 66 layers, 256 routed experts, six selected routed experts plus two shared experts, multimodal encoders, and a 1,048,576-token configured maximum. Its [Apache-2.0 license](https://huggingface.co/thinkingmachines/Inkling/raw/main/LICENSE) permits research and commercial use under that license. The associated repository contains model weights and inference artifacts. It is not the environment or RL training system that produces SharpeArena episodes.

## Strongest candidates for this workstation

| Candidate | Official identity | Approximate local artifact | Expected placement | Recommended role |
|---|---|---:|---|---|
| **Ornith-1.5-9B** | MIT; 9B dense Qwen3.5-derived multimodal reasoning model; 262K native context; OpenAI-style tool calls | Publisher Q4_K_M GGUF is 5.78 GB, plus 0.92 GB vision projection if used | Comfortable full GPU | Strong fast agentic field entry. Publisher reports Terminal-Bench 2.1 46.2, SWE-bench Verified 70.6 and MCP-Atlas 54.2 |
| **Ornith-1.5-35B-A3B** | MIT; about 35B total / about 3B active MoE; 262K native context; reasoning and tools | Publisher Q4_K_M GGUF is 21.71 GB, plus 0.90 GB vision projection | Partial GPU offload | Higher-capability Ornith arm. Publisher reports Terminal-Bench 2.1 67.8, SWE-bench Verified 79.0 and MCP-Atlas 70.2 |
| **Qwen3.8-27B** | Apache-2.0; dense 27B, multimodal, 262K native context, thinking on/off, reasoning effort and agent execution | Official [Ollama Q4_K_M](https://ollama.com/library/qwen3.8) is about 18 GB | Partial GPU offload; cap context initially. Official FP8 is larger | Highest-capability near-local arm; publisher reports Terminal-Bench 2.1 73.0, SWE-bench Pro 61.7, GPQA Diamond 89.2 and OSWorld Verified 84.3 |
| **gpt-oss-20b** | Apache-2.0; 21B total / 3.6B active, 128K, Harmony tool/structured-output protocol | Official [Ollama artifact](https://ollama.com/library/gpt-oss/tags) is about 14 GB | Marginal full-GPU at short context, otherwise tiny offload | Strong agent/tool comparator; preserve the mandatory Harmony template |
| **Gemma 4 31B** | Apache-2.0; strongest Gemma 4 checkpoint, dense 30.7B; text/image, 256K, configurable thinking and native function calling | Publisher QAT Q4_0 GGUF is 17.65 GB, plus 1.20 GB vision projection | Partial GPU offload | Gemma family frontier. Publisher reports AIME 2026 89.2, LiveCodeBench v6 80.0 and Tau2 76.9 |
| **Gemma 4 26B-A4B** | Apache-2.0; 25.2B total / 3.8B active MoE, text/image, 256K, thinking and function calling | Publisher QAT Q4_0 GGUF is 14.44 GB, plus 1.19 GB vision projection | Borderline even without the vision projection; expect a small CPU offload once runtime/KV memory is included | Faster Gemma MoE arm; publisher reports AIME 2026 88.3, LiveCodeBench v6 77.1 and Tau2 68.2 |
| **Gemma 4 12B** | Apache-2.0; dense 11.95B unified text/image/audio model, 256K, thinking and native function calling | Publisher QAT Q4_0 GGUF is 6.98 GB, plus 0.18 GB multimodal data | Comfortable full GPU | Primary fast Gemma entry; publisher reports AIME 2026 77.5, LiveCodeBench v6 72.0 and Tau2 69.0 |
| **Ministral 3 14B Instruct** | Apache-2.0; multimodal, long context, function calling/JSON; publisher supplies additional quantized checkpoints | Q4 is expected around 8–10 GB depending on format | Full GPU | Vendor/architecture diversity; validate the exact official quant revision before field use |
| **Qwen3.6-35B-A3B** | Apache-2.0; 35B / 3B active, multimodal, 262K native context and tools | Q4 commonly exceeds 20 GB | Partial GPU offload | MoE diversity and continuity with already-installed Qwen-family models; not the first throughput arm |
| **gpt-oss-120b** | Apache-2.0; 117B / 5.1B active, 128K, native MXFP4 | Official Ollama artifact about 65 GB | Heavy CPU/RAM offload | Large-model calibration arm at low seed count only; not a full field |

Primary cards and quant collections: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b), [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B), [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), [Ornith publisher GGUF collection](https://huggingface.co/collections/ornith-ai/ornith-15), [Gemma 4 model card](https://huggingface.co/google/gemma-4-31B-it), [Gemma 4 publisher QAT/GGUF collection](https://huggingface.co/collections/google/gemma-4-qat-q4-0), and [Ministral 3 14B](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-BF16).

### Current local inventory

The older local tags (`qwen3.5:*`, `qwen3.6:*`, `ornith:*`, `gemma4:12b` and `ministral-3:8b`) were removed before field design so an obsolete checkpoint cannot enter by convenience. Those weights remain recoverable from their registries; no project data was deleted. The Ollama model root remains on `D:\\ollama\\models`.

The sole installed model at this snapshot is the current `qwen3.8:27b` Ollama artifact:

| Field | Recorded value |
|---|---|
| Registry digest | `22130167c4c20e20c7b71454612966ca8e8171e9b3cc8ab6ce8aa6cbfec79643` |
| Local size | 17,741,872,154 bytes |
| Family / parameters | Ollama family `qwen35`; 27.3B parameters |
| Format / quantization | GGUF / Q4_K_M |
| Capabilities | completion, vision, tools, thinking |
| Installed | 2026-08-26 01:45:30 +03:00 |

The internal Ollama family label is not the benchmark identity. A field row must identify the publisher repository as Qwen3.8-27B, retain the registry and blob digests, and record Ollama plus its version and serving configuration. Installation is preparation, not a completed inference smoke test or performance result.

The removed `ornith:9b` and `ornith:35b` tags were Ornith 1.0, not the current 1.5 family. If Ornith enters the field, use the distinct `ornith-1.5:9b` or `ornith-1.5:35b` tags and capture their exact local manifests; the publisher/Ollama [1.5 tag page](https://ollama.com/library/ornith-1.5/tags) exposes the current artifacts.

Do not fill a field with Qwen-derived models merely because they are available. At least three model families and two inference backends are needed to distinguish model behavior from family- or engine-specific behavior.

## Inference engines versus desktop wrappers

| System | Category | A4000/Windows fit | Structured output and tools | Appropriate use here |
|---|---|---|---|---|
| **llama.cpp** | Inference engine and server | Native Windows or WSL2; strongest GGUF CPU/GPU layer-offload path | OpenAI-compatible endpoints, JSON-schema-to-grammar constraints, tools, continuous batching and parallel decoding | **Primary offload baseline**, especially for 18–65 GB GGUF models. Pin a commit and still post-validate every response: grammar/schema conversion has had fail-open edge cases |
| **vLLM** | High-throughput inference engine | Linux; use WSL2. A4000 CC 8.6 is supported | OpenAI server, structured outputs, named/required tools, reasoning/tool parsers, continuous batching and prefix caching | **Primary throughput baseline** for models that fit fully in VRAM. Prefer native AWQ/GPTQ/FP8 formats supported by the exact architecture over experimental GGUF/offload paths |
| **SGLang** | High-throughput inference engine | Linux/WSL2 or Docker; Ampere supported | XGrammar JSON/regex/EBNF constraints, tool and reasoning parsers, radix/prefix cache | Cross-engine sensitivity and high-throughput agent arm. On Ampere, NVFP4 falls back to a W4A16/Marlin-style path rather than native FP4 |
| **TensorRT-LLM** | NVIDIA-optimized compiler/runtime | Linux/WSL2; Ampere supports FP16/BF16/INT8/INT4, not native FP8/NVFP4 weight execution | Guided decoding and tool-serving support | Optional optimization arm only after correctness on another engine; model conversion and engine building add substantial provenance and maintenance cost |
| **Transformers / Transformers Serve** | Reference library and general server | Broad newest-architecture support; `device_map` can spread work across GPU/CPU | Model-template-dependent tools/structured generation | Compatibility and one-cell reference backend, not the initial high-throughput field server |
| **TGI** | Former Hugging Face production server | No reason to start a new deployment | Supported constrained generation historically | **Do not adopt**: the [official repository](https://github.com/huggingface/text-generation-inference) was archived in March 2026 and recommends vLLM, SGLang, llama.cpp or MLX |
| **Ollama** | Packager/runtime and local API wrapper | Excellent Windows convenience; manages GGUF-like quantized artifacts and GPU/CPU placement | OpenAI-compatible API, model-dependent tools/thinking and schema formats | Smoke tests and convenient acquisition. Record its version, model digest, context and concurrency; do not treat it as invisible |
| **LM Studio** | Desktop wrapper/server | Good Windows interactive UX over local engines | [Structured output](https://lmstudio.ai/docs/developer/openai-compat/structured-output) and OpenAI-compatible tools; GGUF constraints inherit the underlying llama.cpp grammar path | Manual inspection and debugging. For published runs, invoke a pinned engine directly or record LM Studio plus its embedded runtime/version |
| **Jan / LocalAI** | Desktop wrapper / multi-backend gateway | Viable convenience layers | Backend- and model-dependent | No unique advantage for the first reproducible field; introducing another gateway expands the provenance surface |

Engine sources: [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [llama.cpp grammars](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md), [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/), [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/), [vLLM tool calling](https://docs.vllm.ai/en/latest/features/tool_calling/), [SGLang installation](https://docs.sglang.io/docs/get-started/install), [SGLang structured outputs](https://docs.sglang.io/docs/advanced_features/structured_outputs), [SGLang quantization](https://docs.sglang.io/docs/advanced_features/quantization), [TensorRT-LLM support matrix](https://nvidia.github.io/TensorRT-LLM/1.0.0/reference/support-matrix.html), and [TensorRT-LLM quantization](https://nvidia.github.io/TensorRT-LLM/1.1.0/features/quantization.html).

## Recommended local sequence

The research pass did not run an experiment. Qwen3.8-27B was installed only after the matrix was frozen, and remains unscored. The installation policy is frontier-family-only: older but convenient comparators such as Qwen3.6, Ornith 1.0, Ministral 3 and gpt-oss are not in the initial download queue.

1. **Correctness baseline:** a current-family full-GPU arm, Gemma 4 12B or Ornith-1.5-9B, through a pinned llama.cpp server at 8K context with schema-constrained output and strict host-side validation. A schema-valid but semantically invalid trade remains an agent fault.
2. **Near-local frontier:** smoke-test the installed Qwen3.8-27B with measured partial offload before making it a field arm. Keep seed count reductions in a separately labelled calibration arm until throughput is known.
3. **Current-family diversity:** add Gemma 4 26B-A4B or 31B and Ornith-1.5-35B-A3B only if their measured offload throughput supports the predeclared replication budget. Do not count Ornith as independent of its Qwen-derived base architecture when interpreting family effects.
4. **Throughput baseline:** serve the same exact checkpoint and quantization through vLLM under WSL2 where supported. Run a frozen one-cell cross-backend check before expanding the grid.
5. **Cross-engine sensitivity:** repeat at least one frozen model/cell through SGLang. If actions, parse-fault rates or scores change materially, the backend must remain a reported experimental factor.
6. **Non-local frontier:** retain Kimi K3, Qwen3.8-2.4T, DeepSeek-V4-Pro, Inkling, GLM-5.2 and Ornith-1.5-397B in the comparison and future multi-GPU plan, not the local download queue. Their active-parameter counts do not make their total checkpoints fit.

Installation does not admit a model to the benchmark. A checkpoint enters only after its exact artifact and license are pinned, a one-cell structured-output test passes, placement and throughput are measured, and the full field plan is frozen before inference.

## Required provenance record

Every evidence row should bind the model and inference stack, not just a display name:

- publisher/repository, exact revision and local artifact digest;
- license identifier and any applicable use restriction;
- quantization method, quantizer/converter version, file digest and calibration provenance;
- engine name, semantic version and commit; wrapper name/version if present;
- chat template, reasoning parser, tool parser and constrained-decoding backend;
- GPU-layer/offload map, tensor parallelism, KV-cache dtype, context cap, batch size, parallel slots and prefix/speculative settings;
- CUDA, NVIDIA driver and GPU identity;
- sampling parameters, model seed, thinking/reasoning setting and token budgets;
- raw completion, schema-validation outcome, semantic-validation outcome, timeout/fault class and retry count.

Structured decoding prevents many syntax failures; it does not prove that a symbol was observed, a weight is finite or within the mandate, or the decision is economically meaningful. The harness must reject and record faults distinctly. It must never turn a parser, timeout or server failure into an unflagged hold.

Before a field run, compare the same prompt, schema, model revision, quantization and sampling seed across the chosen backends. Record action agreement, invalid-output rate, timeout rate, prompt/completion tokens, latency and downstream score. That small sensitivity study is the evidence needed to decide whether an inference engine is merely infrastructure or part of the treatment.
