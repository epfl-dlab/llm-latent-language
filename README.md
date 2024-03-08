# Logit lens plot colab

[Try out your own prompts here.](https://colab.research.google.com/drive/1l6qN-hmCV4TbTcRZB5o6rUk_QPHBZb7K?usp=sharing)

# Installation

Set up a python environment and run
`pip install -r requirements.txt`

# Usage 

## Translation 

`papermill Translation.ipynb out.ipynb -p input_lang fr -p target_lang zh`

## Cloze

`papermill Cloze.ipynb out.ipynb -p target_lang fr`

# Precomputed latents

For your convenience, we also provide some precomputed latents on [huggingface.](https://huggingface.co/datasets/wendlerc/llm-latent-language) Here are some [preliminary steering experiments](https://colab.research.google.com/drive/1EhCk3_CZ_nSfxxpaDrjTvM-0oHfN9m2n?usp=sharing) using the precomputed latents.

# Acknowledgements

Starting point of this repo was [Nina Rimsky's](https://github.com/nrimsky/LM-exp/blob/main/intermediate_decoding/intermediate_decoding.ipynb) Llama-2 wrapper.

# Citation
```
@article{wendler2024llamas,
  title={Do Llamas Work in English? On the Latent Language of Multilingual Transformers},
  author={Wendler, Chris and Veselovsky, Veniamin and Monea, Giovanni and West, Robert},
  journal={arXiv preprint arXiv:2402.10588},
  year={2024}
}
```
