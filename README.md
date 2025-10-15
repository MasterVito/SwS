<h1 align="center">
<img src="./docs/static/images/icon.png" width="250" alt="SwS-Logo" />
<br>
<!-- <br style="display: block; content: ''; margin-top: 0.5em;" /> -->
SwS: A Weakness-driven Problem Synthesis Framework</span>
</h1>

<div align="center">

![](https://img.shields.io/badge/Task-LLM%20Reasoning-orange)
![](https://img.shields.io/badge/Demo%20Dataset-Released-blue)
![](https://img.shields.io/badge/Code%20License-MIT-green)

</div>

<p align="center">
  <a href="https://mastervito.github.io/MasterVito.SwS.github.io/"><b>[🌐 Website]</b></a> •
  <a href="https://huggingface.co/datasets/MasterVito/SwS-Demo-Dataset"><b>[🤗 Demo Dataset]</b></a> •
  <a href="https://arxiv.org/pdf/2506.08989"><b>[📜 Paper]</b></a> •
  <a href="https://github.com/MasterVito/SwS"><b>[🐱 GitHub]</b></a> •
  <a href="https://x.com/MasterVito0601/status/1933354531905286628"><b>[🐦 Twitter]</b></a> •
  <a href="https://www.xiaohongshu.com/discovery/item/684bd6800000000003038a48?source=webshare&xhsshare=pc_web&xsec_token=ABOzgM8hshsNCAt9EEVXB5uCw3v7frcLqS9Wft04M_9xQ=&xsec_source=pc_share"><b>[📕 Rednote]</b></a>
</p>





<p align="center">
Repo for "<a href="https://arxiv.org/pdf/2506.08989" target="_blank">SwS: Self-aware Weakness-driven Problem Synthesis in Reinforcement Learning for LLM Reasoning</a>"
</p>

<p align="center">
    <img src="docs\static\images\teaser.png" width="1000">
        <br>
    <em>Figure 1: 32B model performance across mainstream reasoning benchmarks and different domains.
    </em>
</p>



## 🔥 News

- [2025/10/14] 🔥 We release all code, including implementations for RL training and problem synthesis.
- [2025/09/18] SwS has been accepted to NeurIPS 2025! Welcome any discussions during the conference.
- [2025/06/13] We release all prompts used in the SwS framework in <a href="https://github.com/MasterVito/SwS/tree/master/prompts"><b>prompts</b></a>.
- [2025/06/13] We update the demo set of synthetic problems from SwS in <a href="https://github.com/MasterVito/SwS/tree/master/datasets"><b>datasets</b></a>, including 500 samples for each model and category. You can also find them in <a href="https://huggingface.co/datasets/MasterVito/SwS-Demo-Dataset"><b>Demo Dataset</b></a>.
- [2025/06/10] **Our full code and datasets are under review by Microsoft and will be released upon approval.**
- [2025/06/10] SwS paper, repo, website and demo datasets released.


## 💡 Introduction 

<div style="text-align: justify;">
The Self-aware Weakness-driven problem Synthesis framework (SwS) framework proposes to identifies model deficiencies and leverages them for problem augmentation. The weaknesses are defined as questions that the model consistently fails to learn through during RL training. SwS extracts the core concepts from these failure cases and synthesize new problems to strengthen the model's weak areas in subsequent augmented training, enabling it to focus on and gradually overcome its weaknesses.
</div>
<br>

<p align="center">
    <img src="./docs/static/images/method.png" width="800">
    <br>
    <em>Figure 2: An overview of our proposed weakness-driven problem synthesis framework that targets at mitigating the model’s reasoning limitations within the RLVR paradigm.
</em>
</p>

## 📊 Evaluation Results


### 7B Model Performance

| **Model**                                            | **GSM8K** | **MATH 500** | **Minerva Math** | **Olympiad Bench** | **GaoKao 2023** | **AMC23** | **AIME24 (Avg@1 / 32)** | **AIME25 (Avg@1 / 32)** | **Avg.** |
|-----------------------------------------------------|----------:|-------------:|------------------:|--------------------:|----------------:|----------:|--------------------------:|--------------------------:|---------:|
| [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)                         | 88.1      | 63.0         | 27.6              | 30.5                | 55.8            | 35.0      | 6.7 / 5.4                 | 0.0 / 1.2                 | 38.3     |
| [Qwen2.5-7B-IT](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)            | 91.7      | 75.6         | 38.2              | 40.6                | 63.9            | 50.0      | 16.7 / 10.5               | 13.3 / 6.7                | 48.8     |
| [Open-Reasoner-7B](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-7B)     | 93.6      | 80.4         | 39.0              | 45.6                | **72.0**        | **72.5**  | 10.0 / 16.8               | 13.3 / 17.9               | 53.3     |
| [SimpleRL-Base-7B](https://huggingface.co/hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zoo)      | 90.8      | 77.2         | 35.7              | 41.0                | 66.2            | 62.5      | 13.3 / 14.8               | 6.7 / 6.7                 | 49.2     |
| BaseRL-7B                                            | 92.0      | 78.4         | 36.4              | 41.6                | 63.4            | 45.0      | 10.0 / 14.5               | 6.7 / 6.5                 | 46.7     |
| **SwS-7B**                                           | **93.9**  | **82.6**     | **41.9**          | **49.6**            | 71.7            | 67.5      | **26.7** / **18.3**       | **20.0** / **18.5**       | **56.7** |
| Δ (vs. BaseRL)                                       | +1.9      | +4.2         | +5.5              | +8.0                | +8.3            | +22.5     | +16.7 / +3.8              | +13.3 / +12.0             | **+10.0** |



### 32B Model Performance


| **Model**                                            | **GSM8K** | **MATH 500** | **Minerva Math** | **Olympiad Bench** | **GaoKao 2023** | **AMC23** | **AIME24 (Avg@1 / 32)** | **AIME25 (Avg@1 / 32)** | **Avg.** |
|-----------------------------------------------------|----------:|-------------:|------------------:|--------------------:|----------------:|----------:|-------------------------:|-------------------------:|---------:|
| [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B)                         | 90.1      | 66.8         | 34.9              | 29.8                | 55.3            | 50.0      | 10.0 / 4.2                | 6.7 / 2.5                 | 42.9     |
| [Qwen2.5-32B-IT](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)            | 95.6      | 83.2         | 42.3              | 49.5                | 72.5            | 62.5      | 23.3 / 15.0               | 20.0 / 13.1               | 56.1     |
| [Open-Reasoner-32B](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-32B)     | 95.5      | 82.2         | 46.3              | 54.4                | 75.6            | 57.5      | 23.3 / 23.5               | 33.3 / 31.7               | 58.5     |
| [SimpleRL-Base-32B](https://huggingface.co/hkust-nlp/Qwen-2.5-32B-SimpleRL-Zoo)          | 95.2      | 81.0         | 46.0              | 47.4                | 69.9            | 82.5      | 33.3 / 26.2               | 20.0 / 15.0               | 59.4     |
| BaseRL-32B                                           | 96.1      | 85.6         | 43.4              | 54.7                | 73.8            | 85.0      | 40.0 / 30.7               | 6.7 / 24.6                | 60.7     |
| **SwS-32B**                                          | **96.3**  | **89.4**     | **47.1**          | **60.5**            | **80.3**        | **90.0**  | **43.3** / **33.0**       | **40.0** / **31.8**       | **68.4** |
| Δ (vs. BaseRL)                                       | +0.2      | +3.8         | +3.7              | +5.8                | +6.5            | +5.0      | +3.3 / +2.3               | +33.3 / +7.2              | **+7.7** |


<div style="text-align: justify;">
P.S: Additional results for Qwen2.5-3B and Qwen2.5-7B-Math are provided in the paper.
</div>
<br>


## 🚀 Quick Start

We recommend using [Conda](https://docs.conda.io/projects/miniconda) to manage your environment. We use [vLLM](https://github.com/vllm-project/vllm) (0.10.1.1) to accelerate inference. Run the following commands to setup your environment:

```sh
git git@github.com:MasterVito/SwS.git && cd SwS
conda create -n sws python=3.10.16
conda activate sws
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128 # CUDA 12.8 for example
pip install -r requirements.txt
```

**Model downloading:** Here we utilize the [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) model trained on the <a href="data/MATH_12k.parquet"><b>MATH-12k</b></a> dataset. You can download the model using the following command:

```sh
mkdir -p models
pip install -U "huggingface_hub[cli]"
huggingface-cli login # use your huggingface token
huggingface-cli download Qwen/Qwen2.5-7B --local-dir models/Qwen2.5-7B
```

## 1. Weakness Identification in Initial RL
We provide a bash script for running the weakness identification stage on the Qwen2.5-7B base model. During this stage, we do not filter out problems with 0% or 100% accuracy, as we set `data.accuracy_lower_bound=0.0` and `data.accuracy_upper_bound=1.0`. The indices of the selected problems from the training set will be saved to the specified `save_path`.

```bash
bash scripts/qwen25_7b_weakness_identification.sh
```

## 2. Problem Synthesis
The sampling accuracy of problems at each step is also stored in the model checkpoint path. You can compute and summarize these accuracies following the format in the <a href="record"><b>record</b></a> folder.

Given the recorded problems with low learning efficiency, we begin by extracting key concepts from the recorded problems using the <a href="https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct"><b>Llama-3.3-70B-Instruct</b></a>  model:

```
bash scripts/synthesis/step1_concepts_extraction.sh
```

Next, the extracted concepts are encoded into embeddings using the <a href="https://huggingface.co/meta-llama/Llama-3.1-8B"><b>Llama-3.1-8B</b></a>  model:

```
bash scripts/synthesis/step2_concepts_encoding.sh
```

After embedding the concepts, we aggregate them by category and allocate a sampling budget for each category based on their normalized failure ratios across categories:

```
bash scripts/synthesis/step3_concepts_sampling.sh
```

Here we start generating new questions using <a href="https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct"><b>Llama-3.3-70B-Instruct</b></a>  based on the sampled concepts derived from the model's low-efficiency learning problems, i.e., the weaknesses identified in our study.

```
bash scripts/synthesis/step4_problem_generation.sh
```

We then evaluate the quality of the synthetic questions using both the <a href="https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct"><b>Llama-3.3-70B-Instruct</b></a>  and <a href="https://huggingface.co/Qwen/Qwen2.5-72B-Instruct"><b>Qwen2.5-72B-Instruct</b></a>  models, filtering out those that do not meet our standard—specifically, requiring at least one perfect rating and one acceptable rating.

```
bash scripts/synthesis/step5_quality_evaluation.sh
```

Next, we generate reference answers for the high-quality synthetic problems using strong reasoning models such as <a href="https://huggingface.co/Qwen/QwQ-32B"><b>QwQ-32B</b></a> .

```
bash scripts/synthesis/step6_answer_verification.sh
```

After generating the reference answers, we prompt the initially trained model with the synthetic questions and retain only those that fall within an acceptable accuracy range and exhibit an appropriate level of difficulty. Finally, we incorporate the remaining questions into the original set and start the second round of the augmented RL training.

## 3. Augmented RL Training
Here is the bash script for running the augmented RL training on the [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) base model. During this stage, we set `data.accuracy_lower_bound=0.125` and `data.accuracy_upper_bound=0.875`. 

```bash
bash scripts/qwen25_7b_augment_training.sh
```

## 🔎 Evaluation
We provide a script for inference, simply config the `model_name_or_path` and `data_path` (default as using MATH-500 and AIME24 & AIME25 for evaluation) in [scripts/evaluation.sh](scripts/evaluation.sh) and run the following command:

```sh
bash scripts/evaluation.sh
```


## ☕️ Citation

If you find this repository helpful, please consider citing our paper:

```
@misc{liang2025swsselfawareweaknessdrivenproblem,
      title={SwS: Self-aware Weakness-driven Problem Synthesis in Reinforcement Learning for LLM Reasoning}, 
      author={Xiao Liang and Zhong-Zhi Li and Yeyun Gong and Yang Wang and Hengyuan Zhang and Yelong Shen and Ying Nian Wu and Weizhu Chen},
      year={2025},
      eprint={2506.08989},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.08989}, 
}
```

## 🙏 Acknowledgement
We sincerely appreciate the outstanding work of [BigMath](https://github.com/SynthLabsAI/big-math), [PromptCoT](https://github.com/zhaoxlpku/PromptCoT), and [veRL](https://github.com/volcengine/verl). The prompts used in the SwS framework are largely inspired by BigMath and PromptCoT, while the training code is adapted from the excellent veRL repository.

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mastervito/SwS&type=Date)](https://star-history.com/#mastervito/SwS&Date)
