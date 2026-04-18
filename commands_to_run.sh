#!/usr/bin/env bash

mkdir -p logs

nohup env CUDA_VISIBLE_DEVICES=0 python ./run_incremental_pvalue_figure.py --model-name EleutherAI/pythia-410m-deduped --model-label Deduped --model-size-b 0.41 --dataset-names all --sample-size 500 --batch-size 50 --feature-mode full --num-random 1 --normalize train --outliers clip --output-dir results/enhanced_incremental_pvalue_figure_041_deduped > logs/enhanced_pythia_410m_deduped.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python ./run_incremental_pvalue_figure.py --model-name EleutherAI/pythia-410m --model-label Non-Deduped --model-size-b 0.41 --dataset-names all --sample-size 500 --batch-size 50 --feature-mode full --num-random 1 --normalize train --outliers clip --output-dir results/enhanced_incremental_pvalue_figure_041_non_deduped > logs/enhanced_pythia_410m_non_deduped.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python ./run_incremental_pvalue_figure.py --model-name EleutherAI/pythia-1.4b-deduped --model-label Deduped --model-size-b 1.4 --dataset-names all --sample-size 500 --batch-size 50 --feature-mode full --num-random 1 --normalize train --outliers clip --output-dir results/enhanced_incremental_pvalue_figure_14b_deduped > logs/enhanced_pythia_14b_deduped.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python ./run_incremental_pvalue_figure.py --model-name EleutherAI/pythia-1.4b --model-label Non-Deduped --model-size-b 1.4 --dataset-names all --sample-size 500 --batch-size 50 --feature-mode full --num-random 1 --normalize train --outliers clip --output-dir results/enhanced_incremental_pvalue_figure_14b_non_deduped > logs/enhanced_pythia_14b_non_deduped.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=4 python ./run_incremental_pvalue_figure.py --model-name EleutherAI/pythia-6.9b-deduped --model-label Deduped --model-size-b 6.9 --dataset-names all --sample-size 500 --batch-size 50 --feature-mode full --num-random 1 --normalize train --outliers clip --output-dir results/enhanced_incremental_pvalue_figure_69b_deduped > logs/enhanced_pythia_69b_deduped.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=5 python ./run_incremental_pvalue_figure.py --model-name EleutherAI/pythia-6.9b --model-label Non-Deduped --model-size-b 6.9 --dataset-names all --sample-size 500 --batch-size 50 --feature-mode full --num-random 1 --normalize train --outliers clip --output-dir results/enhanced_incremental_pvalue_figure_69b_non_deduped > logs/enhanced_pythia_69b_non_deduped.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=6 python ./run_incremental_pvalue_figure.py --model-name EleutherAI/pythia-12b-deduped --model-label Deduped --model-size-b 12.0 --dataset-names all --sample-size 500 --batch-size 50 --feature-mode full --num-random 1 --normalize train --outliers clip --output-dir results/enhanced_incremental_pvalue_figure_12b_deduped > logs/enhanced_pythia_12b_deduped.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=7 python ./run_incremental_pvalue_figure.py --model-name EleutherAI/pythia-12b --model-label Non-Deduped --model-size-b 12.0 --dataset-names all --sample-size 500 --batch-size 50 --feature-mode full --num-random 1 --normalize train --outliers clip --output-dir results/enhanced_incremental_pvalue_figure_12b_non_deduped > logs/enhanced_pythia_12b_non_deduped.log 2>&1 &

# Run this after all jobs finish:
# python ./merge_incremental_pvalue_histories.py --history-files results/enhanced_incremental_pvalue_figure_041_deduped/history.json,results/enhanced_incremental_pvalue_figure_041_non_deduped/history.json,results/enhanced_incremental_pvalue_figure_14b_deduped/history.json,results/enhanced_incremental_pvalue_figure_14b_non_deduped/history.json,results/enhanced_incremental_pvalue_figure_69b_deduped/history.json,results/enhanced_incremental_pvalue_figure_69b_non_deduped/history.json,results/enhanced_incremental_pvalue_figure_12b_deduped/history.json,results/enhanced_incremental_pvalue_figure_12b_non_deduped/history.json --output-dir results/enhanced_incremental_pvalue_figure_merged
