# OneIG-Bench
## OneIG-Bench: Omni-dimensional Nuanced Evaluation for Image Generation

<div class="is-size-4 publication-authors" align="center">
   <span class="author-block">
       <a href="https://scholar.google.com/citations?user=i1accPcAAAAJ&hl=en">Jingjing Chang</a><sup>1,2</sup>,
   </span>
   <span class="author-block">
       <a href="https://scholar.google.com/citations?user=yDNIBE0AAAAJ&hl=en">Yixiao Fang</a><sup>2&dagger;</sup>,
   </span>
   <span class="author-block">
       <a href="https://scholar.google.com/citations?user=k_jXsNYAAAAJ&hl=en">Peng Xing</a><sup>2</sup>,
   </span>
   <span class="author-block">
     <a href="https://openreview.net/profile?id=~Shuhan_Wu3">Shuhan Wu</a><sup>2</sup>,
   </span>
<span class="author-block">
       <a href="https://wchengad.github.io">Wei Cheng</a><sup>2</sup>,</span>
   </span>
</div>
<div class="is-size-4 publication-authors" align="center">
   <span class="author-block">
     <a href="https://wrong.wang/">Rui Wang</a><sup>2</sup>,
   </span>
   <span class="author-block">
     <a href="https://scholar.google.com/citations?user=tgDc0fsAAAAJ&hl=en">Xianfang Zeng</a><sup>2</sup>,
   </span>
   <span class="author-block">
       <a href="https://www.skicyyu.org/">Gang Yu</a><sup>2&ddagger;</sup>,</span>
   <span class="author-block">
       <a href="https://dmne.sjtu.edu.cn/dmne/faculty/chenhaibao/">Hai-Bao Chen</a><sup>1&ddagger;</sup>
   </span>
</div>
<div class="is-size-5 publication-authors" align="center">
   <span class="author-block"><sup>&dagger;</sup>Project lead,</span>
   <span class="author-block"><sup>&ddagger;</sup>Correspondence Authors</span>
</div>
<div class="is-size-5 publication-authors" align="center">
   <span class="author-block"><sup>1</sup>Shanghai Jiao Tong University,</span>
   <span class="author-block"><sup>2</sup>StepFun</span>
</div>
</div>

<br> 

<div align="center">
 <a href="https://oneig-bench.github.io/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a> &ensp;
 <a href="https://huggingface.co/datasets/OneIG-Bench/OneIG-Bench"><img src="https://img.shields.io/static/v1?label=Dataset&message=Huggingface&color=yellow"></a> &ensp;
  <a href="https://arxiv.org/abs/2506.07977"><img src="https://img.shields.io/static/v1?label=Tech%20Report&message=Arxiv&color=red"></a> &ensp;
</div>

<br>  

## 🔥🔥🔥 News
* **`2025.09.19`** 🌟 **OneIG-Bench has been accepted to NeurIPS 2025 DB Track**.
* **`2025.09.19`** 🌟 We updated the **Seedream 4.0**, **Gemini-2.5-Flash-Image(Nano Banana)**, **Step-3o Vision** and [HunyuanImage-2.1](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1) evaluation results on our leaderboard [here](https://oneig-bench.github.io).
* **`2025.09.19`** 🌟 We updated the [NextStep-1](https://github.com/stepfun-ai/NextStep-1), [Lumina-DiMOO](https://github.com/Alpha-VLLM/Lumina-DiMOO) and [IRG](https://github.com/Osilly/Interleaving-Reasoning-Generation) official evaluation results on our leaderboard [here](https://oneig-bench.github.io).
* **`2025.08.13`** 🌟 We updated the [Qwen-Image](https://github.com/QwenLM/Qwen-Image) official evaluation results on our leaderboard [here](https://oneig-bench.github.io).
* **`2025.08.13`** 🌟 We updated the **fine-grained analysis script** [here](fine_grained_analysis.py).
<!-- * **`2025.07.23`** 🌟 We updated the Step-3o Vision evaulation results on our leaderboard [here](https://oneig-bench.github.io). -->
* **`2025.07.03`** 🌟 We updated the [Ovis-U1](https://github.com/AIDC-AI/Ovis-U1) evaulation results on our leaderboard [here](https://oneig-bench.github.io).
* **`2025.06.25`** 🌟 We updated the [Show-o2](https://github.com/showlab/Show-o/tree/main/show-o2) and [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2) evaulation results on our leaderboard [here](https://oneig-bench.github.io).
* **`2025.06.23`** 🌟 We released the **T2I generation script**  [here](text2image.py).
* **`2025.06.10`** 🌟 We released the [OneIG-Bench](https://huggingface.co/datasets/OneIG-Bench/OneIG-Bench) benchmark on 🤗huggingface.
* **`2025.06.10`** 🌟 We released the [tech report](https://arxiv.org/abs/2506.07977) and the [project page](https://oneig-bench.github.io/).
* **`2025.06.10`** 🌟 We released the evaluation scripts. 

<p align="center">
    <img src="./assets/OneIG-Bench.png" width="92%">
</p>


## To Do List
- [x] Fine-grained Analysis Script
- [x] Real-time Updating Leaderboard
- [x] OneIG-Bench Release 
- [x] Evaluation Scripts, Technical Report & Project Page Release

## Introduction

We introduce OneIG-Bench, a meticulously designed comprehensive benchmark framework for fine-grained evaluation of T2I models across multiple dimensions, including subject-element alignment, text rendering precision, reasoning-generated content, stylization, and diversity. Specifically, these dimensions can be flexibly selected for evaluation based on specific needs.

Key contribution:

- We present **OneIG-Bench**, which consists of six prompt sets, with the first five — 245 *Anime and Stylization*, 244 *Portrait*, 206 *General Object*, 200 *Text Rendering*, and 225 *Knowledge and Reasoning* prompts — each provided in both English and Chinese, and 200 *Multilingualism* prompts, designed for the comprehensive evaluation of current text-to-image models.
- A systematic quantitative evaluation is developed to facilitate objective capability ranking through standardized metrics, enabling direct comparability across models. Specifically, our evaluation framework allows T2I models to generate images only for prompts associated with a particular evaluation dimension, and to assess performance accordingly within that dimension.
- State-of-the-art open-sourced methods as well as the proprietary model are evaluated based on our proposed benchmark to facilitate the development of text-to-image research.

## Get Started

### Dependencies and Installation:
We test our benchmark using torch==2.6.0, torchvision==0.21.0 with cuda-11.8, python==3.10.

Install requirements:
  
``` bash
pip install -r requirements.txt
```

The version of flash-attention is in the last line of [`requirements.txt`](requirements.txt).

To evaluate style performance, please download the [CSD model](https://drive.google.com/file/d/1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46/view?usp=sharing) and [CLIP model](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt), then put them under `./scripts/style/models`.
Also, you can download the [`OneIG-StyleEncoder`](https://huggingface.co/xingpng/OneIG-StyleEncoder) here.

For diversity metrics, some models and packages [link1](https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_ensemble_checkpoint.zip), [link2](https://github.com/facebookresearch/dino/zipball/main), [link3](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth) are needed to download and save in the folder [`models`] that is a sibling to [`assests`] and [`scripts`].
### Image Generation

**You can use the [`scirpt`](text2image.py) to generate images.** You only need to set up the inference function in the script for generating images. 

It's better for you to generate 4 images(with different seeds) for each prompt in OneIG-Bench. Each prompt's generated images should be saved into subfolders based on their category **Anime & Stylization, Portrait, General Object, Text Rendering, Knowleddge & Reasoning, Multilingualism**, corresponding to folders **anime, human, object, text, reasoning, multilingualism**. If any image cannot be generated, the script will save a black image with the specified filename. 

The filename for each image should follow the id assigned to that prompt in [`OneIG-Bench.csv`](OneIG-Bench.csv)/[`OneIG-Bench-ZH.csv`](OneIG-Bench-ZH.csv). The structure of the images to be saved should look like:

```bash
📁 images/
├── 📂 anime/                  
│   ├── 📂 gpt-4o/
│   │   ├── 000.webp
│   │   ├── 001.webp
│   │   └── ...
│   ├── 📂 imagen4/
│   └── ...
├── 📂 human/               
│   ├── 📂 gpt-4o/
│   ├── 📂 imagen4/
│   └── ...
├── 📂 object/                
│   └── ...
├── 📂 text/                  
│   └── ...
├── 📂 reasoning/             
│   └── ...
└── 📂 multilingualism/        # For OneIG-Bench-ZH
    └── ...
```

## Evaluation

### Scripts
```shell
./run_{overall, alignment, diversity, reasoning, style, text}.sh
```
The [`run_overall.sh`](run_overall.sh) script contains the execution of all metrics. By running `run_overall.sh`, you can obtain the results of all metrics in the results directory. You can also choose the metric you want to evaluate by running the corresponding script: `run_{metric_name}.sh`.

### Parameters Configuration for Evaluation

To ensure that the generated images are correctly loaded for evaluation, you can modify the following parameters in each script:

1. **`mode`** : Select whether **EN/ZH** to evaluate on **OneIG-Bench** or **OneIG-Bench-ZH**.  

2. **`image_dir`** : The directory where the images generated by your model are stored.  

3. **`model_names`** : The names or identifiers of the models you want to evaluate.  

4. **`image_grid`** : This corresponds to the number of images(with different seeds) generated by the model per prompt, where a value of **1** means 1 image, **2** means 4 images, and so on.

5. **`class_items`** : The prompt categories or image sets you want to evaluate.  

### Fined-grained Analysis for Evaluation Results

If you would like to perform a fine-grained analysis, please re-download the repo or uncomment the lines above  
`if os.path.exists(cache_dir):`  
in `alignment_score.py`, `diversity_score.py`, `reasoning_score.py`, `style_score.py`, and `text_score.py`.


You can copy all the CSV files generated for each prompt dimension (in particular, for the *style* dimension, the files are named `style_style*.csv`) into a subfolder named as the `model name` inside the `RESULT_DIR` directory. 

Then, in [`fine_grained_analysis.py`](fine_grained_analysis.py), adjust the `MODE`, `RESULT_DIR`, and `KEYS` 
parameters as needed to perform the fine-grained analysis.

## 📈 Results

We define the sets of images generated based on the OneIG-Bench prompt categories: *General Object* (**O**), *Portrait* (**P**), *Anime and Stylization* (**A**) for prompts without stylization, (**S**) for prompts with stylization, *Text Rendering* (**T**), *Knowledge and Reasoning* (**KR**), and *Multilingualism* (**L**). 

The correspondence between the evaluation metrics and the evaluated image sets in `OneIG-Bench` and `OneIG-Bench-ZH` is presented in the table below.

- **📊 Metrics and Image Sets Correspondence**

<div align="center">

|                    |                  Alignment                  |    Text    | Reasoning |   Style   |                  Diversity                 |
|--------------------|:-------------------------------------------:|:----------:|:---------:|:---------:|:------------------------------------------:|
| **OneIG-Bench**         | **O**, **P**, **A**, **S**                   | **T**      | **KR**    | **S**     | **O**, **P**, **A**, **S**, **T**, **KR**   |
| **OneIG-Bench-ZH**     | **O**<sub>zh</sub>, **P**<sub>zh</sub>, **A**<sub>zh</sub>, **S**<sub>zh</sub>, **L**<sub>zh</sub> | **T**<sub>zh</sub> | **KR**<sub>zh</sub> | **S**<sub>zh</sub> | **O**<sub>zh</sub>, **P**<sub>zh</sub>, **A**<sub>zh</sub>, **S**<sub>zh</sub>, **L**<sub>zh</sub>, **T**<sub>zh</sub>, **KR**<sub>zh</sub> |
</div>

- **Method Comparision on OneIG-Bench:**

<p align="center">
    <img src="./assets/result.png" width="96%">
</p>

- **Method Comparision on OneIG-Bench-ZH:**

<p align="center">
    <img src="./assets/result_ZH.png" width="96%">
</p>


- **Benchmark Comparison:**

&nbsp;&nbsp;&nbsp;&nbsp;**OneIG-Bench** (also referred to as **OneIG-Bench-EN**) denotes the English benchmark set.
<p align="center">
    <img src="./assets/benchmark_comparison.png" width="86%">
</p>


## Citation
If you find our work helpful for your research, please consider citing our work.

```bibtex
@article{chang2025oneig,
  title={OneIG-Bench: Omni-dimensional Nuanced Evaluation for Image Generation}, 
  author={Jingjing Chang and Yixiao Fang and Peng Xing and Shuhan Wu and Wei Cheng and Rui Wang and Xianfang Zeng and Gang Yu and Hai-Bao Chen},
  journal={arXiv preprint arxiv:2506.07977},
  year={2025}
}
```

## Acknowledgement
We would like to express our sincere thanks to the contributors of [Qwen](https://github.com/QwenLM/Qwen2.5-VL),  [CLIP](https://github.com/openai/CLIP), [CSD_Score](https://github.com/haofanwang/CSD_Score), [DreamSim](https://github.com/ssundaram21/dreamsim), and [HuggingFace](https://huggingface.co) teams, for their open research and exploration.

