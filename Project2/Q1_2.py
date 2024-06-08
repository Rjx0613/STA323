import os

# Set the necessary environment variables
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import argparse
import shutil
import pickle
import tempfile
import pandas as pd
import ray
from ray import tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

output_dir = "/openbayes/home/outputs_20/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_dataset(train_file, valid_file):
    train_df = pd.read_csv(train_file, keep_default_na=False)
    valid_df = pd.read_csv(valid_file, keep_default_na=False)
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    return train_dataset, valid_dataset

def preprocess_function(examples, tokenizer):
    inputs = examples['Input']
    targets = examples['Output']

    model_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=384, return_tensors="pt")
    labels = tokenizer(targets, padding='max_length', truncation=True, max_length=30, return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_func(config):
    train_file = config["train_file"]
    valid_file = config["valid_file"]
    train_dataset, valid_dataset = load_dataset(train_file, valid_file)

    model_name = config["model_name"]
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    try:
        with open(config["train_dataset_path"], 'rb') as f:
            train_dataset = pickle.load(f)
        with open(config["valid_dataset_path"], 'rb') as f:
            valid_dataset = pickle.load(f)
        print("Loaded preprocessed datasets from disk.")
    except FileNotFoundError:
        train_dataset = train_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
        valid_dataset = valid_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
        with open(config["train_dataset_path"], 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(config["valid_dataset_path"], 'wb') as f:
            pickle.dump(valid_dataset, f)
        print("Preprocessed datasets and saved to disk.")

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    model = T5ForConditionalGeneration.from_pretrained(model_name)

    trial_id = ray.train.get_context().get_trial_id()
    output_dir = os.path.join(config["output_dir_path"], f"checkpoint_{trial_id}")
    if not os.path.exists(output_dir):
        print(f"创建目录: {output_dir}")
        os.makedirs(output_dir)
    else:
        print(f"目录已存在: {output_dir}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_valid_batch_size"],
        weight_decay=0.01,
        save_total_limit=1,  # 只保留一个 checkpoint
        num_train_epochs=config["num_train_epochs"],
        predict_with_generate=True,
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator
    )

    # Call train method once to train for all epochs
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    val_loss = eval_results.get("eval_loss")

    if val_loss is None:
        raise ValueError("Evaluation results did not contain 'eval_loss'")

    # Report metrics
    metrics = {"eval_loss": val_loss, "epoch": config["num_train_epochs"]}
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        trainer.save_model(temp_checkpoint_dir)
        tokenizer.save_pretrained(temp_checkpoint_dir)
        session.report(
            metrics,
            checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
        )
    if ray.train.get_context().get_world_rank() == 0:
        print(metrics)

def tune_transformer(args):
    tune_config = {
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "per_device_train_batch_size": tune.choice([16, 32]),
        "per_device_valid_batch_size": 4,
        "num_train_epochs": tune.choice([2, 3, 4]),
        "max_steps": 1 if args.smoke_test else -1,
        "train_file": args.train_file,
        "valid_file": args.valid_file,
        "model_name": args.model_name,
        "train_dataset_path": args.train_dataset_path,
        "valid_dataset_path": args.valid_dataset_path,
        "output_dir_path": args.output_dir_path
    }

    reporter = CLIReporter(
        parameter_columns={
            "learning_rate": "lr",
            "per_device_train_batch_size": "train_bs/gpu",
            "num_train_epochs": "num_epochs",
        },
        metric_columns=["eval_loss", "epoch"],
        max_report_frequency=10,  # 控制报告频率
        print_intermediate_tables=False  # 关闭中间表格输出
    )

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=2,
        hyperparam_mutations={
            "learning_rate": tune.loguniform(1e-5, 1e-3),
            "per_device_train_batch_size": [4, 6, 8],
        }
    )

    analysis = tune.run(
        tune.with_parameters(train_func),
        resources_per_trial={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        config=tune_config,
        num_samples=args.num_samples,
        scheduler=pbt,
        metric="eval_loss",
        mode="min",
        progress_reporter=reporter,
        storage_path=args.local_dir_path
    )

    print("Best hyperparameters found were: ", analysis.best_config)
    best_trial = analysis.get_best_trial(metric="eval_loss", mode="min")
    best_model_path = best_trial.checkpoint.path

    shutil.copytree(best_model_path, "/openbayes/home/best_model_20")
    print("The best model has been successfully saved as 'best_model'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training file.")
    parser.add_argument("--valid_file", type=str, required=True, help="Path to the validation file.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path.")
    parser.add_argument("--train_dataset_path", type=str, required=True, help="Path to save/load the processed training dataset.")
    parser.add_argument("--valid_dataset_path", type=str, required=True, help="Path to save/load the processed validation dataset.")
    parser.add_argument("--output_dir_path", type=str, required=True, help="Path to save the checkpoints")
    parser.add_argument("--local_dir_path", type=str, required=True, help="Path to save the Running analysis")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples for hyperparameter tuning.")
    parser.add_argument("--cpus_per_trial", type=int, default=3, help="Number of CPUs per trial.")
    parser.add_argument("--gpus_per_trial", type=int, default=0, help="Number of GPUs per trial.")
    parser.add_argument("--smoke_test", action="store_true", help="Run a smoke test.")
    

    args = parser.parse_args()

    # 初始化 Ray 并运行调优
    ray.init(ignore_reinit_error=True)
    tune_transformer(args)
