<h1 align="center">
<img src="./docs/static/images/icon.png" width="250" alt="SwS-Logo" />
<br>
<!-- <br style="display: block; content: ''; margin-top: 0.5em;" /> -->
SwS: A Weakness-driven Problem Synthesis Framework</span>
</h1>

<div align="center">

![](https://img.shields.io/badge/LLM-Reasoning%20Reasoning-orange)
![](https://img.shields.io/badge/Code%20License-MIT-green)

</div>

<p align="center">
  <a href="https://mastervito.github.io/MasterVito.SwS.github.io/"><b>[🌐 Website]</b></a> •
  <a href=""><b>[🤗 Demo Dataset]</b></a> •
  <a href="https://arxiv.org/pdf/2506.08989"><b>[📜 Paper]</b></a> •
  <a href="https://github.com/MasterVito/SwS"><b>[🐱 GitHub]</b></a> •
  <a href="https://x.com/weizhuchen/status/1933194338114433470?s=46"><b>[🐦 Twitter]</b></a>
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

<!-- - [2023/10/13] 🔥🔥🔥 We release a demo for ToRA at [🐯 Gradio](https://9557c5365a6f44dc84.gradio.live), try it out!!! -->
- [2023/06/10] Our full code and datasets are under review by Microsoft and will be released upon approval.
- [2023/06/10] SwS paper, repo, website and demo datasets released.

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
---

| **Model**                                            | **GSM8K** | **MATH 500** | **Minerva Math** | **Olympiad Bench** | **GaoKao 2023** | **AMC23** | **AIME24 (Avg@1 / 32)** | **AIME25 (Avg@1 / 32)** | **Avg.** |
|-----------------------------------------------------|----------:|-------------:|------------------:|--------------------:|----------------:|----------:|--------------------------:|--------------------------:|---------:|
| [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)                         | 88.1      | 63.0         | 27.6              | 30.5                | 55.8            | 35.0      | 6.7 / 5.4                 | 0.0 / 1.2                 | 38.3     |
| [Qwen2.5-7B-IT](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)            | 91.7      | 75.6         | 38.2              | 40.6                | 63.9            | 50.0      | 16.7 / 10.5               | 13.3 / 6.7                | 48.8     |
| [Open-Reasoner-7B](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-7B)     | 93.6      | 80.4         | 39.0              | 45.6                | **72.0**        | **72.5**  | 10.0 / 16.8               | 13.3 / 17.9               | 53.3     |
| [SimpleRL-Base-7B](https://huggingface.co/hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zoo)      | 90.8      | 77.2         | 35.7              | 41.0                | 66.2            | 62.5      | 13.3 / 14.8               | 6.7 / 6.7                 | 49.2     |
| BaseRL-7B                                            | 92.0      | 78.4         | 36.4              | 41.6                | 63.4            | 45.0      | 10.0 / 14.5               | 6.7 / 6.5                 | 46.7     |
| **SwS-7B**                                           | **93.9**  | **82.6**     | **41.9**          | **49.6**            | 71.7            | 67.5      | **26.7** / **18.3**       | **20.0** / **18.5**       | **56.7** |
| Δ (vs. BaseRL)                                       | +1.9      | +4.2         | +5.5              | +8.0                | +8.3            | +22.5     | +16.7 / +3.8              | +13.3 / +12.0             | **+10.0** |



---

### 32B Model Performance

---

| **Model**                                            | **GSM8K** | **MATH 500** | **Minerva Math** | **Olympiad Bench** | **GaoKao 2023** | **AMC23** | **AIME24 (Avg@1 / 32)** | **AIME25 (Avg@1 / 32)** | **Avg.** |
|-----------------------------------------------------|----------:|-------------:|------------------:|--------------------:|----------------:|----------:|-------------------------:|-------------------------:|---------:|
| [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B)                         | 90.1      | 66.8         | 34.9              | 29.8                | 55.3            | 50.0      | 10.0 / 4.2                | 6.7 / 2.5                 | 42.9     |
| [Qwen2.5-32B-IT](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)            | 95.6      | 83.2         | 42.3              | 49.5                | 72.5            | 62.5      | 23.3 / 15.0               | 20.0 / 13.1               | 56.1     |
| [Open-Reasoner-32B](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-32B)     | 95.5      | 82.2         | 46.3              | 54.4                | 75.6            | 57.5      | 23.3 / 23.5               | 33.3 / 31.7               | 58.5     |
| [SimpleRL-Base-32B](https://huggingface.co/hkust-nlp/Qwen-2.5-32B-SimpleRL-Zoo)          | 95.2      | 81.0         | 46.0              | 47.4                | 69.9            | 82.5      | 33.3 / 26.2               | 20.0 / 15.0               | 59.4     |
| BaseRL-32B                                           | 96.1      | 85.6         | 43.4              | 54.7                | 73.8            | 85.0      | 40.0 / 30.7               | 6.7 / 24.6                | 60.7     |
| **SwS-32B**                                          | **96.3**  | **89.4**     | **47.1**          | **60.5**            | **80.3**        | **90.0**  | **43.3** / **33.0**       | **40.0** / **31.8**       | **68.4** |
| Δ (vs. BaseRL)                                       | +0.2      | +3.8         | +3.7              | +5.8                | +6.5            | +5.0      | +3.3 / +2.3               | +33.3 / +7.2              | **+7.7** |

---

<div style="text-align: justify;">
Additional results for Qwen2.5-3B and Qwen2.5-7B-Math are provided in the paper.
</div>
<br>