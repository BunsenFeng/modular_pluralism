# Modular Pluralism Repository

This is the official repo for [Modular Pluralism: Pluralistic Alignment via Multi-LLM Collaboration](https://arxiv.org/abs/2406.15951).

### Environment

```
conda env create -f plural.yaml
conda activate plural
export OPENAI_API_KEY="YOUR_KEY"
```

### Part I. Community LM Messages

We provide 11 trained community LMs, 6 perspective-informed (default) and 5 culture-informed:

| Domain    | Checkpoints | Data Source |
| -------- | ------- | ------- |
| Perspective  | [1](https://huggingface.co/bunsenfeng/mistral-news_l), [2](https://huggingface.co/bunsenfeng/mistral-news_c), [3](https://huggingface.co/bunsenfeng/mistral-news_r), [4](https://huggingface.co/bunsenfeng/mistral-reddit_l), [5](https://huggingface.co/bunsenfeng/mistral-reddit_c), [6](https://huggingface.co/bunsenfeng/mistral-reddit_r) | [paper](https://arxiv.org/abs/2404.15238) |
| Culture | [1](https://huggingface.co/bunsenfeng/mistral-africa_culture), [2](https://huggingface.co/bunsenfeng/mistral-asia_culture), [3](https://huggingface.co/bunsenfeng/mistral-europe_culture), [4](https://huggingface.co/bunsenfeng/mistral-northamerica_culture), [5](https://huggingface.co/bunsenfeng/mistral-southamerica_culture) | [paper](https://arxiv.org/abs/2305.08283) |

Run the following script to generate community LM msgs for a given task.

#### `generate_community_lm_msg.py`

```
generate_community_lm_msg.py [-h] [-i INPUT] [-t TYPE] [-c CHECKPOINT]

optional arguments:
  -i INPUT, --input INPUT
                        input file name
  -t TYPE, --type TYPE  type of operation: generate or probability
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        checkpoint path
```

By default you don't need to set `-c`: it will loop over all perspective-informed community LMs by default. If you set a specific checkpoint, it will generate msgs for that checkpoint and you can parallelize it across checkpoints.

`-i` and `-t` use the following settings: `(overton_test_valuekaleidoscope, generate)`, `(steerable_test_valuekaleidoscope, generate)`, `(steerable_test_opinionqa, probability)`, `(distributional_test_moralchoice, probability)`, `(distributional_test_globalopinionqa)`, representating the five tasks. The generated msgs will be in `community_lm_msgs/`. The `MoE` and `ours` methods require the community LM msgs for a given task to operate.

### Part II. Methods

We provide the implementation of the three baselines (vanilla LLM, prompting, moe) and proposed approaches in the paper. Shared parameter for each approach:

```
-m MODEL, --model MODEL
                        which language model to use, details below
-i INPUT, --input INPUT
                        which input task file to use without ".json", see "input/" for file names
-t TYPE, --type TYPE
                        "generate" or "probability", details below
-o PORTION, --portion PORTION
                        portion of the data to use, 0-1
```

For `-m`, we provide `{llama2_7b, llama2_13b, llama2_70b, llama3_8b, llama3_70b, gemma_7b, chatgpt}_{aligned, unaligned}` in `lm_utils.py` by default.

`-i` and `-t` use the following settings: `(overton_test_valuekaleidoscope, generate)`, `(steerable_test_valuekaleidoscope, generate)`, `(steerable_test_opinionqa, probability)`, `(distributional_test_moralchoice, probability)`, `(distributional_test_globalopinionqa)`, representating the five tasks.

Portion (0-1) means only evaluating on the first `x%` of the dataset in case the LLM is large and evaluation is slow.

#### `vanilla_lm.py`

```
vanilla_lm.py [-h] [-m MODEL] [-i INPUT] [-t TYPE] [-o PORTION]
```

Directly employing the LLM for the task inputs, saving results in `output/`.

#### `prompting_lm.py`

```
prompting_lm.py [-h] [-m MODEL] [-i INPUT] [-t TYPE] [-o PORTION]
```

Prompting for pluralism, e.g. `Please make sure to reflect diverse values and perspectives in the response.`

#### `moe_lm.py`

```
moe_lm.py [-h] [-m MODEL] [-i INPUT] [-t TYPE] [-o PORTION]
```

Selecting a community LM based on attribute (e.g. `reddit left`), then employ its message for grounded generation.

#### `ours_overton.py`

```
ours_overton.py [-h] [-m MODEL] [-i INPUT] [-t TYPE] [-o PORTION] [-c COMMUNITY_SETTING]

optional arguments:
  -c COMMUNITY_SETTING, --community_setting COMMUNITY_SETTING
                        community setting, default "perspective", or "culture" or "mixed"
```

By default employ the perspective-informed community LMs. `(-i, -t)` should be `(overton_test_valuekaleidoscope, generate)`.

#### `ours_steerable.py`

```
ours_steerable.py [-h] [-m MODEL] [-i INPUT] [-t TYPE] [-o PORTION] [-c COMMUNITY_SETTING]

optional arguments:
  -c COMMUNITY_SETTING, --community_setting COMMUNITY_SETTING
                        community setting
```

`(-i, -t)` should be `(steerable_test_valuekaleidoscope, generate)` or `(steerable_test_opinionqa, probability)`.

#### `ours_distributional.py`

```
ours_distributional.py [-h] [-m MODEL] [-i INPUT] [-t TYPE] [-o PORTION] [-c COMMUNITY_SETTING]

optional arguments:
  -c COMMUNITY_SETTING, --community_setting COMMUNITY_SETTING
                        community setting
```

`(-i, -t)` should be `(distributional_test_moralchoice, probability)`, `(distributional_test_globalopinionqa)`.

### Part III. Evaluation

To evaluate `(overton_test_valuekaleidoscope, generate)` with an NLI model:

```
evaluate_overton_valuekaleidoscope.py [-h] [-o OUTPUT]

optional arguments:
  -o OUTPUT, --output OUTPUT
                        output file name, in output/, without ".json"
```

We by default employ the `Accuracy at 0.33 threshold` metric.

To evaluate `(steerable_test_valuekaleidoscope, generate)`:

```
evaluate_steerable_valuekaleidoscope.py [-h] [-o OUTPUT]

optional arguments:
  -o OUTPUT, --output OUTPUT
                        output file name, in output/, without ".json"
```

The first four numbers are the three-way metrics (support, oppose, either), the last four are the binary (where either cases/predictions are removed).

To evaluate `(steerable_test_opinionqa, probability)`, `(distributional_test_moralchoice, probability)` and `(distributional_test_globalopinionqa)`:

```
evaluate_distributions.py [-h] [-o OUTPUT] [-a ATTRIBUTE]

optional arguments:
  -o OUTPUT, --output OUTPUT
                        output file name
  -a ATTRIBUTE, --attribute ATTRIBUTE
                        attribute to evaluate only, default None, e.g. "EDUCATION" for OpinionQA
```

We employ the `Most likely correctness` metric for `(steerable_test_opinionqa, probability)` and `Average distance` for `(distributional_test_moralchoice, probability)` and `(distributional_test_globalopinionqa)`.

We provide example outputs in the `output/` directory.

### Your Approach

Fill in the `output` fields of generative tasks or the `pred_distribution` fields of distributional tasks in `input/` with your approach and use the evaluation scripts.

### Citation

```
@article{feng2024modular,
  title={Modular Pluralism: Pluralistic Alignment via Multi-LLM Collaboration},
  author={Feng, Shangbin and Sorensen, Taylor and Liu, Yuhan and Fisher, Jillian and Park, Chan Young and Choi, Yejin and Tsvetkov, Yulia},
  journal={arXiv preprint arXiv:2406.15951},
  year={2024}
}
```