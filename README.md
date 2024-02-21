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

# Acknowledgements

Starting point of this repo was [Nina Rimsky's](https://github.com/nrimsky/LM-exp/blob/main/intermediate_decoding/intermediate_decoding.ipynb) Llama-2 wrapper.
