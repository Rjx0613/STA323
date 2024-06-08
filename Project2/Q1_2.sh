# pip install transformers datasets -U "ray[data,train,tune,serve]" sentencepiece
# pip install accelerate -U

rm -r outputs_20
rm -r running_20
rm -r best_model_20

python Q1_2.py \
    --train_file /data/lab/Project2/111.csv \
    --valid_file /data/lab/Project2/222.csv \
    --model_name /data/lab/Project2/Task1/flan-t5-small/ \
    --train_dataset_path /data/lab/Project2/mapping/train_dataset.pkl \
    --valid_dataset_path /data/lab/Project2/mapping/valid_dataset.pkl \
    --output_dir /data/lab/Project2/outputs/ \
    --local_dir_path /data/lab/Project2/running/ \
    --num_samples 1 \
    --cpus_per_trial 3 \
    --gpus_per_trial 0