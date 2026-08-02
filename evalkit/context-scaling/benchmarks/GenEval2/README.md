# GenEval 2: Addressing Benchmark Drift in Text-to-Image Evaluation


This repo contains data and evaluation codes for the paper [GenEval 2: Addressing Benchmark Drift in Text-to-Image Evaluation](https://arxiv.org/abs/2512.16853).

[**📑 Paper**](https://arxiv.org/abs/2512.16853) | [**💻 Code & data**](https://github.com/facebookresearch/GenEval2)


**[Meta FAIR](https://ai.meta.com/research/)**, UW, UCLA, AI2

Amita Kamath, Kai-Wei Chang, Ranjay Krishna, Luke Zettlemoyer, Yushi Hu*, Marjan Ghazvininejad*


## Intro

**GenEval 2** is a Text-to-Image (T2I) benchmark with improved coverage of primitive visual concepts (objects, attributes, relations, counting) and higher degrees of compositionality than existing benchmarks. It contains 800 prompts with varying degrees of compositionality.

GenEval 2 is evaluated with **Soft-TIFA**, a VQA-based evaluation method that combines judgments for visual primitives and is better-aligned with human judgment and less likely to drift from human-alignment over time than other evaluation methods. 

This repository contains the benchmark data of GenEval 2, as well as the evaluation script using Soft-TIFA.

<p align="center">
<img src="figures/soft_tifa.png" width="850">
</p>

## Benchmark Structure
The benchmark data is contained in `geneval2_data.jsonl`, where each line is a dictionary containing the information related to each prompt: the prompt itself (`prompt`), the compositionality a.k.a. atomicity of the prompt (`atom_count`), a list of VQA question-answer pairs for each atom in the prompt (`vqa_list`), and a list of the skill associated with each VQA pair (`skills`). 

For example: 
```
{
    "prompt": "four white bicycles in front of three plastic cows",
    "atom_count": 7,
    "vqa_list": [["How many bicycles are in the image?", "four"], ["Are the bicycles white?", "Yes"], ["Are there any bicycles in the image?", "Yes"], ["Are the bicycles in front of the cows?", "Yes"], ["How many cows are in the image?", "three"], ["Are the cows plastic?", "Yes"], ["Are there any cows in the image?", "Yes"]],
    "skills": ["count", "attribute", "object", "position", "count", "attribute", "object"]
}
```

Note: We do not consider "and" or "a" to contribute to prompt compositionality (i.e., atomicity); however, we evaluate "a" in case the T2I model generated more than one of the required object, hence, it is included in the VQA list, but not the atom count.

## Installation
```
git clone https://github.com/facebookresearch/GenEval2
cd GenEval2
conda create --name geneval2
conda activate geneval2
pip install torch transformers==4.57.0 pillow tqdm scipy
```

## Image Generation
Use any T2I model to generate images for each of the prompts in GenEval 2, and create a dictionary where the keys are the prompts and the values are the filepaths pointing to the corresponding generated image (this will be used by our evaluation script).

## Evaluation
Soft-TIFA uses a VQA model to query the generated image with each of the associated list of questions. It assigns a soft score to each question based on the VQA model's probability assigned to the correct answer when given the image. To obtain an _atom-level_ estimate of model performance, the arithmetic mean of the soft scores per prompt is calculated (Soft-TIFA AM); to obtain a _prompt-level_ estimate of model performance, the geometric mean of the soft scores per prompt is calculated (Soft-TIFA GM). Finally, these scores are averaged over the benchmark. 

For comparison, we also provide code for two other T2I evaluation methods: VQAScore (Lin et al., 2024) and TIFA (Hu et al., 2023). 

### Running the evaluation script
```
python evaluation.py \
    --benchmark_data geneval2_data.jsonl \
    --image_filepath_data your_image_paths.json \
    --method soft_tifa_gm \
    --output_file score_lists.json
```
Where:
- `--benchmark_data`: Path to the GenEval 2 data provided in this repository
- `--image_filepath_data`: Path to the JSON file mapping prompts to image filepaths, as generated in the previous step
- `--method`: Evaluation method (`vqascore`, `tifa`, or `soft_tifa`)
- `--output_file`: Path to save the per-atom output scores

## Analysis
GenEval 2 supports detailed analyses at the prompt- and atom-level. Each prompt is annotated with its atomicity and list of skills per atom, allowing per-skill analysis at the atom-level and per-atomicity analysis at the prompt-level. The former is estimated with Soft-TIFA AM and the latter with Soft-TIFA GM, which are suited to atom- and prompt-level estimation respectively.

### Running the analysis script
```
python soft_tifa_analysis.py \
    --benchmark_data geneval2_data.jsonl \
    --score_data score_lists.json
```
Where:
- `--benchmark_data`: Path to the GenEval 2 data provided in this repository
- `--score_data`: Path to the per-atom scores generated in the evaluation step

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0) - see the [LICENSE](LICENSE) file for details.

## Citation

If you use GenEval2 in your research, please consider citing our work.

**BibTeX:**
```bibtex
@article{kamath2025geneval,
  title={GenEval 2: Addressing Benchmark Drift in Text-to-Image Evaluation},
  author={Kamath, Amita and Chang, Kai-Wei and Krishna, Ranjay and Zettlemoyer, Luke and Hu, Yushi and Ghazvininejad, Marjan},
  journal={arXiv preprint arXiv:2512.16853},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
