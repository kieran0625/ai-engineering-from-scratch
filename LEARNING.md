# My AI Engineering Path
<!-- Managed by the ai-engineering-from-scratch learning skills.
     Repo: https://github.com/rohitg00/ai-engineering-from-scratch -->

## Mission
Learn AI engineering from a research and academic perspective — understand the theory deeply and contribute to the field. Build goal: train and fine-tune my own model on custom data.

## Placement
- Date: 2026-08-15
- Score: 10/10 — Math & Statistics: 2/2, Classical ML: 2/2, Deep Learning: 2/2, NLP & Transformers: 2/2, Applied AI: 2/2
- Entry point: Phase 14 — Agent Engineering
- Pace: ~10 hours/week

## Path
| Phase | Name | Status | Est. hours |
|-------|------|--------|------------|
| 0 | Setup & Tooling | Skip | -- |
| 1 | Math Foundations | Skip | -- |
| 2 | ML Fundamentals | Skip | -- |
| 3 | Deep Learning Core | Skip | -- |
| 4 | Computer Vision | Skip | -- |
| 5 | NLP — Foundations to Advanced | Skip | -- |
| 6 | Speech & Audio | Skip | -- |
| 7 | Transformers Deep Dive | Skip | -- |
| 8 | Generative AI | Skip | -- |
| 9 | Reinforcement Learning | Skip | -- |
| 10 | LLMs from Scratch | Skip | -- |
| 11 | LLM Engineering | Skip | -- |
| 12 | Multimodal AI | Skip | -- |
| 13 | Tools & Protocols | Skip | -- |
| 14 | Agent Engineering | Do | 42 |
| 15 | Autonomous Systems | Do | 20 |
| 16 | Multi-Agent & Swarms | Do | 28 |
| 17 | Infrastructure & Production | Do | 32 |
| 18 | Ethics, Safety & Alignment | Do | 31 |
| 19 | Capstone Projects | Do | 620 |
| | | **Total** | **773** |

## Progress log
| Date | Lesson | Quiz | Note |
|------|--------|------|------|
| 2026-08-15 | 14/01-the-agent-loop | 2/2 | Discussed explicit finish vs implicit no-tool-call stop; user argued else-branch is safer, explored both sides. Strong intuition about error-as-observation feedback. |
| 2026-08-18 | 03/04-activation-functions | 3/3 | Detour: correctly explained nonlinearity, vanishing gradients, dead ReLU neurons, and stable softmax; verified scratch results against PyTorch. |
| 2026-08-18 | 03/05-loss-functions | 3/3 | Detour cont'd. Strong: derived MSE=0.25 game, softmax+CE gradient (p-1), label smoothing as regularization. Coached on nuance: MSE/BCE loss values aren't cross-comparable, and label smoothing caps logits (not gradients). Ran scratch comparison. |
| 2026-08-19 | 03/06-optimizers | 4/5 | Detour. Strong: Momentum accumulation, RMSProp denominator scaling, Adam bias-correction (m_hat > m via /(1-beta^t)), traced scratch Adam by hand. Missed: epsilon's role (numerical stability / avoid div-by-zero, not momentum). |
| 2026-08-19 | 03/training-loop (follow-up) | 5/5 | Ad-hoc lesson wiring optimizer into a full loop (not a curriculum dir; draws on 03/11-intro-to-pytorch). Strong: zero_grad→forward→loss→backward→step order, gradient accumulation if zero_grad skipped, eval()/no_grad(), loss.item() detaches graph. Corrected batch-mean gradient arithmetic (grad_w=-17, grad_b=-8). |
| 2026-08-19 | 03/07-regularization | 4/4 | Detour. Strong: overfit/underfit spectrum diagnosis, inverted dropout scaling derivation, gamma=1/beta=0 init reasoning. Coached through: why Adam+L2 ≠ AdamW (needed full derivation), why LayerNorm underperforms on CNN channels (initial guess wrong, corrected via channel-comparability argument), why forgetting model.eval() breaks determinism (needed concrete numeric example). Saved 2 Obsidian notes (AdamW derivation, BatchNorm/LayerNorm/RMSNorm comparison). |
| 2026-08-21 | 03/08-weight-initialization | 2/3 | Resumed from interruption at Step 3. Ran forward_deep 50-layer experiment live (real output showed Xavier/Kaiming decay gradually over 50 layers despite theory predicting flat signal -- correctly attributed the gap to finite-width/finite-sample drift and tied it back to lesson 07's BatchNorm/LayerNorm). Correctly derived GPT-2's 2N=24 residual additions (12 layers x 2 sublayers) and why 1/sqrt(2N) scaling caps variance at 1 regardless of depth. Initially flipped explode vs vanish for random N(0,1) init (corrected). Missed: quiz Q on Xavier vs Kaiming choice -- said Xavier is "universally better" instead of activation-matched (sigmoid/tanh vs ReLU). |
| 2026-08-22 | 03/09-learning-rate-schedules | 5/5 | Detour. Warm-up: nailed the Xavier/Kaiming re-check missed last time. Strong: cosine formula boundary conditions, why cos() re-rising past total_steps mirrors SGDR warm restarts, correctly distinguished warmup's purpose for Adam (moment-estimate reliability) vs large-batch SGD (Goyal et al. linear-scaling-rule instability). Corrected one miscalculation: step//step_size integer division (250//100=2, not 0.25) mis-costed a 100x lr drop as 3x. Ran the lesson's real compare_schedules/lr_sensitivity script live -- output contradicted the textbook story (Constant beat all scheduled variants, lr=1.0 didn't diverge); correctly reasoned this was because the toy task is a single shallow layer plus sigmoid+MSE bounding d_out <= 0.25, so neither depth-compounded variance (lesson 08) nor unbounded gradients could appear -- explained why schedules matter more at LLM scale, less on toy tasks. |
| 2026-08-22 | 03/10-mini-framework | 3/3 | Detour, taught in Chinese (中文). Warm-up 2/2 (1cycle high-LR-as-regularization, LR-too-high divergence). Ran the real ~500-line framework script live (Adam 98%, SGD 97%, Adam+Dropout 98% on circle classification). Correctly derived why Linear.parameters() must return position references (container, i, j) rather than values -- traced Python value-copy vs in-place mutation live to show why returning a plain float list would silently break the optimizer. Needed a second, slower pass (assembly-line metaphor + explicit chain-rule expansion) before grasping why Sequential.backward() must process modules in reverse order -- got it after the second explanation. Initially confused "moving zero_grad() after step()" (harmless reordering) with "deleting zero_grad() entirely" (causes runaway gradient accumulation via Linear.backward()'s +=) -- clarified with a 4-row comparison table of zero_grad placements. |
| 2026-08-22 | 03/11-intro-to-pytorch | 3/3 | Detour, taught in Chinese (中文). Warm-up 2/2 (Module's 3 responsibilities, DataLoader role). Ran real MNIST training live on MPS (2 epochs, 97.1% test acc, ~2-4s/epoch -- matched doc's claimed 45min-scratch vs seconds-on-GPU gap). Correctly derived that nn.Parameter assigned to self.x is a Python object reference (verified `p is layer.weight` -> True), connecting cleanly to lesson 10's position-reference requirement for parameters(). Correctly explained GPU speedup as C++/CUDA kernel dispatch across forward+backward, not just gradient computation (initial guess was "gradients run on GPU" -- too narrow). Two misses needing correction: (1) predicted repeated .backward() without zero_grad() would recompute the same gradient (2x+3) rather than accumulate (4x+6) -- despite having covered gradient accumulation deeply in the 03/10 zero_grad-placement discussion, the transfer to a fresh autograd example didn't happen automatically; (2) misread the images/255.0 normalization as being about binary pixel classification rather than matching input variance to Xavier/Kaiming's unit-variance assumption from lesson 08 -- needed the std=84 vs std=0.33 live demo to click. Also cold on float16 update-underflow reasoning (said "forgot") -- walked through 0.5+1e-7 rounding to 0 in float16 vs float32. |
| 2026-08-23 | 03/13-debugging-neural-networks | 5/5 | Detour, taught in Chinese (中文). Warm-up 2/2 (nn.Linear shape, GPU kernel speedup). Correctly identified vanishing gradients from a 20-layer sigmoid network but initially attributed it to "bad initialization" rather than the chain-rule multiplication of sigmoid's bounded derivative (<=0.25) across depth -- corrected with the 0.25^20 calculation, tied to lesson 08's 50-layer experiment. Ran real broken-network diagnostics live: Bug1 (lr=10) showed activations exploding into the millions while gradient diagnostics falsely read HEALTHY -- correctly reasoned this was softmax+CE gradient saturation (p-y bounded in [0,1], lesson 05 callback) after initial answer ("gradients are decreasing") was too vague. Bug3 (missing zero_grad) produced a genuine surprise: loss converged cleanly to 0 instead of oscillating: at 50 steps and lr=0.01, hypothesized "not enough accumulation yet," then live-verified at 500 steps loss still converges to exactly 0 -- correctly reasoned (with guidance) that accumulation is self-limiting when the model perfectly fits an easily-separable problem, since per-step gradients shrink to ~0 as p->y, so the accumulated sum converges rather than diverging (lesson's oscillation story only holds when the model can't reach near-zero loss). Independently diagnosed a real live bug: gradient_check() crashed converting integer class labels to .double(), correctly identifying x needs float precision for finite-difference but y is a discrete index for CrossEntropyLoss's gather op, not a continuous quantity -- fixed the code live, gradient check then passed at 1e-10 rel_diff. Needed correction on snapshot-vs-time-series diagnostics: proposed "snapshot for gradients, time-series for loss" (matches this implementation's specific choices but isn't the general principle) -- clarified the real axis is instantaneous/structural vs trend/drift, applying equally to loss, gradients, and activations. |

## Review queue
<!-- /learn adds lessons that quizzes flag for review -->
- 03/08-weight-initialization (2/3, 2026-08-21): missed activation-matched init choice — Xavier is not "universally better," it's for sigmoid/tanh specifically; Kaiming's extra factor of 2 exists because ReLU zeroes half the activations, which Xavier doesn't need.
