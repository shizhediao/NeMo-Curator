
# Climb Data Curation

## Step 1: Compute_embeddings
```bash
cd /lustre/fsw/portfolios/llmservice/users/sdiao/NeMo-Curator/nemo_curator/scripts/climb_data_curation

python compute_embeddings.py --input-data-dir /lustre/fsw/portfolios/llmservice/users/sdiao/data/cc-19+synthetic+smollm-20clusters-sampled --input-file-type "jsonl" --input-file-extension "jsonl" --input-text-field "text"  --config-file /lustre/fsw/portfolios/llmservice/users/sdiao/NeMo-Curator/config/climb_config.yaml
```


## Step 2: Clustering
```bash
python clustering.py --id-column "doc_id" --config-file /lustre/fsw/portfolios/llmservice/users/sdiao/NeMo-Curator/config/climb_config.yaml
```

## Step 3: Cluster pruning
```bash
python cluster_pruning.py --config-file /lustre/fsw/portfolios/llmservice/users/sdiao/NeMo-Curator/config/climb_config.yaml
```


## Step 4: Generate training configs for different data ratio configurations
```bash
python synthesize_mixture.py
```

## Step 5: Train proxy model
Use some training frameworks (e.g., NeMo) to train the proxy model based on the synthesized configurations.


## Step 6: Predictor training
```bash
python predictor_training.py
```

# TODOs and Requirements

## TODO

1. **Step 5 is currently empty**  
   Implement the training process using **Nemo** or any other preferred framework.

2. **Improve compatibility between Step 4/6 and nemo-curator**  
   Steps 4 and 6 currently rely on custom code, which has poor compatibility with **nemo-curator**. This needs to be optimized.

## Requirements

- `seaborn`
- `lightgbm`