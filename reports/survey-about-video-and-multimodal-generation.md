# Video and multimodal generation: from short clips to interactive systems

## TL;DR

- Video generation has moved from 3D convolutional diffusion toward compressed spatiotemporal latents and Transformer denoisers, which make variable-size and larger-scale training more practical. The available quality comparisons are still specific to a model and benchmark. [1][2][3][4]
- Interactive systems add causal streaming, control inputs, and memory, but their speed depends heavily on resolution, hardware, and inference setup. Current demonstrations do not establish general physical reasoning. [5][6][7][8][9]
- Joint audio-video models can exchange information across modalities while generating, and recent systems also accept image-plus-text conditions. Evidence for better synchronization is encouraging for particular systems, but cross-paper comparisons remain limited. [10][11][12][13]
- Benchmarks have become more diagnostic, covering temporal consistency, alignment, trustworthiness, and human preference. Automated evaluators and efficiency techniques help, yet they do not remove evaluation gaps or deployment costs. [14][15][16][17][18]

## Background

This survey reflects sources retrieved through 2026-10-09. I treat work since 2024-10-09 as recent, while retaining earlier papers to explain the technical foundations.

Early video diffusion extended image generation by replacing two-dimensional operations with a 3D U-Net and training across images and videos. Ho and colleagues reported strong results on video-prediction and unconditional-generation benchmarks, and showed that conditional sampling could extend fixed-length generation autoregressively. This established a useful recipe, but its results are not directly comparable to current text-to-video systems, which differ in scale, data, resolution, and evaluation. [1]

A later architectural step was to compress video in space and time, then perform diffusion in that latent representation. CogVideoX combines a 3D VAE with a Transformer and an expert adaptive normalization design intended to improve text-video fusion. Its authors report progressive training, multi-resolution frame packing, and generation of ten-second videos at 16 frames per second and 768 by 1360 resolution. The abstract also reports strong benchmark and human-evaluation performance, though the retrieved summary does not give scores or enough protocol detail to compare it numerically with later models. [2]

OpenAI's Sora report describes a related patch-based approach: compress video into a latent representation, divide it into spacetime patches, and train a Transformer across images and videos of different durations and sizes. This supports a shared representation for varied aspect ratios and clip lengths. The report shows qualitative improvement in fixed-seed samples as training compute grows, while withholding implementation details and acknowledging errors in physical interactions and longer samples. I read this as evidence that scaling and representation help broaden generation, not proof that generated scenes behave like a reliable world simulator. [3]

## Latent representations and scale

Recent systems make the representation and data pipeline central parts of video generation. LTX-Video's paper summary describes a latent diffusion Transformer that integrates the Video-VAE and denoiser, aiming to reduce compute for high-resolution generation. The summary does not provide quantitative benchmark results, so efficiency and temporal consistency should be treated as the paper's stated aims rather than a result that can be ranked against other systems. [19]

Wan illustrates the scale of this design direction. Its technical report describes 1.3-billion- and 14-billion-parameter models, a spatiotemporal causal VAE, and pretraining on billions of images and videos. The authors' Wan-Bench gives their 14B model a weighted score of 0.724 versus 0.700 for Sora. The aggregate uses human-preference-derived weights and hides differences: Wan scores higher on its physical-plausibility measure while scoring lower on smoothness and large-motion generation. Those measurements support the authors' comparison on that benchmark, not a general ranking across products or tasks. [4]

Taken together, these systems suggest a practical pattern: learned compression lowers the sequence length that the denoiser sees, while Transformer designs offer a way to scale across space, time, and conditioning. But compression can discard fine motion or detail, and increased capacity does not by itself guarantee stable objects across frames. The reports provide varying levels of evidence, from qualitative samples to benchmark tables, so a fair comparison should keep model size, clip duration, prompt set, and metric visible instead of collapsing them into a single quality label. [2][3][4]

## Longer horizons, control, and interactive generation

A short prompt-to-clip system generates once and stops; an interactive system must continue after user input while retaining enough past context to avoid abrupt changes. Matrix-Game 2.0 uses keyboard and mouse actions, autoregressive diffusion, and a rolling attention cache. Its authors report 25 frames per second on one H100 and minute-level generation, with training on annotated game footage. This is a relevant demonstration of interactive streaming, but the retrieved evidence does not establish causal behavior beyond the tested game environment. [5]

MotionStream focuses on motion control through painted tracks and camera controls. Its authors report sub-second latency and up to 29.5 frames per second on one H100 using a streaming Tiny VAE; at higher resolution, the reported speed falls. Its ablations suggest that a fixed attention sink plus a small local window can stabilize extrapolation, while larger windows can increase drift. Yume1.5 adds text-controlled events to keyboard-based exploration and reports 12 frames per second at 540p on one A100. The paper also reports artifacts such as backwards-moving vehicles and reverse-walking characters. These results show useful forms of user control, while making clear that control interfaces do not guarantee plausible outcomes. [6][7]

Other systems treat spatial memory as a design problem. WorldScape combines navigation and hand manipulation with depth and 3D Gaussian Splatting supervision, and uses geometry-aware memory. It reports 16 frames per second on one H100, while a separate paper setting on an A800 reports a substantially different throughput. Matrix-Game 3.0 describes camera-aware retrieval and reports 40 frames per second at 720p for an optimized deployment using eight GPUs for the diffusion Transformer and one for the VAE. The difference in hardware and inference setup makes these headline rates unsuitable for direct ranking. Moreover, geometric consistency measures or memory-revisit tests are not the same as demonstrating correct forces or general physical causality. [8][9][20]

A 2026 proposal, Memory Forcing, identifies another limitation of fixed-size attention caches: they can retain opening and recent frames while losing important events in the middle. It proposes separate archive and working cache regions without increasing cache size. The abstract gives no score values, though it reports the smallest drop from five to 60 seconds among methods reporting all four durations. This is comparative author-reported evidence, but it does not establish that the memory problem is solved. [21]

## Joint audio-video generation

Video generation is also shifting from pipelines that create a silent clip and add sound afterward toward joint systems. SyncFlow generates audio and video from text with dual diffusion Transformers, and uses staged modality training followed by joint fine-tuning. Its abstract reports improved correlation and audiovisual correspondence against baselines, but the short retrieved record does not give metric values. JavisDiT adds hierarchical spatial-temporal priors to guide synchronization and introduces JavisBench, described as 10,140 captioned videos with sound. The authors report gains in quality and synchronization. Together these papers show that alignment is becoming an explicit model and benchmark target; they do not prove that joint generation always beats a well-designed cascade. [10][11]

More recent work expands conditioning and output types. MOVA describes image-plus-text-to-video-and-audio generation, using separate pretrained video and audio components connected through cross-attention. The paper summary and retrieved text describe speech, environmental sounds, and music, but the full comparative results were not available in the retrieved material. Salt++, surfaced through HF Daily, targets few-step streaming audio-video generation and proposes post-training to align training context with causal inference. Its abstract does not report quantitative synchronization or quality results. I therefore treat these as evidence of active engineering directions, not measured proof of a universal quality gain. [12][13][22]

Joint generation offers a plausible benefit: sound and image streams can condition one another during synthesis rather than passing a finished output in only one direction. Cascaded approaches still have practical value when teams can reuse existing modality-specific models or need to add audio to a fixed video. The evidence reviewed here varies between abstract-level summaries and paper-level descriptions, and no common independent protocol compares synchronization across these systems. Better evaluation should measure timing and semantic correspondence as well as the quality of each stream alone. [10][11][12]

## Evaluation, safety, and efficiency

Video benchmarks have moved beyond treating a clip as one score. VBench breaks quality into 16 dimensions, including identity consistency, motion smoothness, flicker, spatial relationships, and prompt alignment, and reports correlation with human preferences in its tested setup. VBench++ extends the benchmark to image-to-video, long videos, and trustworthiness. This is useful because a model can look sharp while changing a subject's appearance or missing the requested motion. The benchmark authors' validations are informative, but results still depend on prompts, model samples, and human annotation choices. [14][15]

Automated video judges are promising but remain imperfect substitutes for people. UVE-Bench's HF-indexed abstract says the study evaluates 16 multimodal language models using human pairwise preferences across 15 aspects; it reports that advanced MLLMs surpass specialized metrics but still lag behind human evaluators. Because the checked abstract does not provide per-aspect detail, I keep this conclusion at the overall benchmark level. [16]

Safety also has temporal dimensions. T2VSafetyBench organizes risks into 14 aspects across four categories, including content that becomes unsafe through continuity or changes across frames. Its experiments cover nine released systems and use a 689-prompt subset, partly because of computational limits and model access restrictions. The authors report no system excels across all aspects and identify a usability-safety trade-off. The scope is useful for recognizing risks missed by frame-by-frame image filters, but its selected models and prompts cannot represent every policy or deployment setting. [17]

Generation remains costly, especially as temporal attention processes longer clips. Sparse Video-Gen proposes training-free spatial-temporal sparse attention and reports end-to-end speedups of up to 2.28 times on CogVideoX-v1.5 and 2.33 times on HunyuanVideo. These are meaningful engineering gains on the named systems, not evidence that high-resolution video generation is generally cheap or real-time on ordinary devices. [18]

## Trends and open problems

Across the papers, I see three connected pressures: longer generation, tighter control, and synchronized modalities. Compression and large Transformers make long clips more tractable, while causal caches and memory address streaming. Audio-video models bring multiple outputs into the same generation process. Each change adds new failure modes: history may be forgotten, objects may drift, sound may not match visible events, and increased speed may depend on substantial hardware. The recent papers offer useful systems and measurements, but most evidence remains author-reported and task-specific. [4][6][21][11][13][18]

Evaluation needs to catch up with these capabilities. Benchmarks should report clip duration, resolution, hardware, prompt design, and inference settings alongside quality; otherwise throughput and quality comparisons can mislead. Long-horizon tests should check whether a scene remains stable after revisiting, not only whether adjacent frames look smooth. Audio-video tests need to score timing and semantic alignment separately from sound quality and image quality. Human comparisons remain important, particularly because automated judges still have difficulty with motion and temporal behavior. [14][15][16]

A related open problem is what “world model” should mean. Interactive inputs and geometrically consistent views are meaningful milestones, but they do not show that a generator predicts outcomes under interventions or follows physical rules in new settings. The papers reviewed here support narrower claims about control, memory, throughput, and benchmark performance. Future work would be more convincing if it separated visual continuity from causal tests, used repeatable deployment conditions, and made training and evaluation details available. [3][5][7][8][9]

## References
[1] Video Diffusion Models. arxiv. https://arxiv.org/abs/2204.03458 (2022-04-07)
[2] CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. arxiv. https://arxiv.org/abs/2408.06072 (2024-08-12)
[3] Video generation models as world simulators. web. https://openai.com/index/video-generation-models-as-world-simulators/ (2024-02-15)
[4] Wan: Open and Advanced Large-Scale Video Generative Models. web. https://arxiv.org/abs/2503.20314v2 (2025-03-26)
[5] Matrix-game 2.0: An open-source real-time and streaming interactive world model. web. https://arxiv.org/abs/2508.13009 (2025)
[6] MotionStream: Real-Time Video Generation with Interactive Motion Controls. web. https://arxiv.org/abs/2511.01266 (2025)
[7] Yume1.5: A Text-Controlled Interactive World Generation Model. web. https://arxiv.org/abs/2512.22096 (2025-12-26)
[8] WorldScape: An Efficient Real-Time Interactive World Model with Universal Control. web. https://worldscape.io/ (2026)
[9] Matrix-Game 3.0: Real-Time and Streaming Interactive World Model with Long-Horizon Memory. hf-search. https://huggingface.co/papers/2604.08995 (2026-04-10)
[10] SyncFlow: Toward Temporally Aligned Joint Audio-Video Generation from Text. hf-search. https://huggingface.co/papers/2412.15220 (2024-12-03)
[11] JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization. web. https://arxiv.org/html/2503.23377 (2025-03-30)
[12] MOVA: Towards Scalable and Synchronized Video–Audio Generation. hf-search. https://huggingface.co/papers/2602.08794 (2026-02-09)
[13] MOVA: Towards Scalable and Synchronized Video–Audio Generation. web. https://arxiv.org/html/2602.08794v2 (2026-02-09)
[14] VBench: Comprehensive Benchmark Suite for Video Generative Models. web. https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.pdf (2024)
[15] VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models. hf-search. https://huggingface.co/papers/2411.13503 (2024-11-20)
[16] UVE: Are MLLMs Unified Evaluators for AI-Generated Videos?. hf-search. https://huggingface.co/papers/2503.09949 (2025-03-13)
[17] T2VSafetyBench: Evaluating the Safety of Text-to-Video Generative Models. web. https://papers.nips.cc/paper_files/paper/2024/file/74eed5f568354c2e77dd9b018f38a9d4-Paper-Datasets_and_Benchmarks_Track.pdf (2024-12-10)
[18] Sparse Video-Gen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity. web. https://proceedings.mlr.press/v267/xi25c.html (2025-10-06)
[19] LTX-Video: Realtime Video Latent Diffusion. hf-search. https://huggingface.co/papers/2501.00103 (2024-12-30)
[20] Matrix-Game 3.0: Real-Time and Streaming Interactive World Model with Long-Horizon Memory. web. https://arxiv.org/abs/2604.08995 (2026-04)
[21] Memory Forcing: Attendable Mid-Horizon History for Streaming Video Generation. arxiv. https://arxiv.org/abs/2610.11756 (2026-10-08)
[22] Salt++: Context-Aligned Post-Training for Few-Step Streaming Multimodal Generation. hf-daily. https://huggingface.co/papers/2609.36995 (2026-09-29)
