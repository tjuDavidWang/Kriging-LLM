# Kriging-LLM
Implement the [Time-LLM](https://github.com/KimMeen/Time-LLM) approach for Kriging tasks.

## Requirements

Use python 3.11 from MiniConda

To install all dependencies:

```sh
pip install -r requirements.txt
```

## Datasets

You can access the well pre-processed datasets from [[Google Drive]](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view?usp=sharing), then place the downloaded contents under `./dataset`

## Quick Demos

1. Download datasets and place them under `./dataset`

2. Tune the model. We provide five experiment scripts for demonstration purpose under the folder `./scripts`. For example, you can evaluate on ETT datasets by:

   ```sh
   bash ./scripts/TimeLLM_ETTh1.sh 
   bash ./scripts/TimeLLM_ETTh2.sh 
   bash ./scripts/TimeLLM_ETTm1.sh 
   bash ./scripts/TimeLLM_ETTm2.sh
   ```

## Detailed usage

Please refer to ```run_main.py```, ```run_m4.py``` and ```run_pretrain.py``` for the detailed description of each hyperparameter.
