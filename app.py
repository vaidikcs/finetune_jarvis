import argparse
from datasets import Dataset
import torch
import pandas as pd
from datetime import datetime
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from datasets import load_dataset
from models import set_status, create_session

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(
        offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

import subprocess

token = "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"
command = f"huggingface-cli login --token {token}"

# Run the command
subprocess.run(command, shell=True)

# -------------------

def main():
    try:
        parser = argparse.ArgumentParser(description='Description of your script.')

        parser.add_argument('--job_id', default=1,
                            help='Path to input file')
        parser.add_argument('--input', default='default_input.txt',
                            help='Path to input file')
        # parser.add_argument(
        #     '--output', default='default_input.txt', help='Path to output file')
        parser.add_argument(
            '--final_model', default='0to60ai/first', help='Path to output file')
        parser.add_argument(
            '--database_url', default="mysql+pymysql://root:12345@localhost:3306/bruData?charset=utf8mb4", help='SQL url')
        
        args = parser.parse_args()

        repo_name = args['final_model']

        create_session(args['database_url'])
        set_status(args['job_id'], 'base model loading started')

        base_model_id = "SkunkworksAI/phi-2"
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, token=token, trust_remote_code=True, torch_dtype=torch.float16, load_in_8bit=True)

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            padding_side="left",
            token=token,
            add_eos_token=True,
            add_bos_token=True,
            use_fast=False,  # needed for now, should be fixed soon
        )
        tokenizer.pad_token = tokenizer.eos_token
        set_status(args['job_id'], 'dataset loading started')
        # -------------------

        data = ['Instruct: Create a new experiment named "MLflowDemo" with a custom artifact location..\nOutput: <<<api_call>>>: 2.0/mlflow/experiments/create\n<<<api_call_type>>>: POST\n<<<params>>>: {\n    "name": "MLflowDemo",\n    "artifact_location": "/custom/location"\n}\n<<<explanation>>>: This API call creates a new MLflow experiment named "MLflowDemo" with a custom artifact location set to "/custom/location".. <s>',
                'Instruct: Find MLflow experiments with a maximum of 5 results, filtered by name \'MLflowDemo\', ordered by start time, and showing only active experiments..\nOutput: <<<api_call>>>: 2.0/mlflow/experiments/search\n<<<api_call_type>>>: POST\n<<<params>>>: {"max_results": 5, "filter": "name=\'MLflowDemo\'", "order_by": ["start_time"], "view_type": "ACTIVE"}\n<<<explanation>>>: Use this API to search for MLflow experiments with a maximum of 5 results, filtered by name \'MLflowDemo\', ordered by start time, and showing only active experiments.. <s>',
                'Instruct: Fetch information about the MLflow experiment with ID 123..\nOutput: <<<api_call>>>: 2.0/mlflow/experiments/get\n<<<api_call_type>>>: GET\n<<<params>>>: {"experiment_id": "123"}\n<<<explanation>>>: Retrieve details of the MLflow experiment with ID 123 using this API call.. <s>',
                'Instruct: Update the name of the MLflow experiment with ID 456 to "NewExperimentName"..\nOutput: <<<api_call>>>: 2.0/mlflow/experiments/update\n<<<api_call_type>>>: POST\n<<<params>>>: {"experiment_id": "456", "new_name": "NewExperimentName"}\n<<<explanation>>>: Use this API to update the name of the MLflow experiment with ID 456 to "NewExperimentName".. <s>',
                'Instruct: Register a new model under the name "MyModel" with optional tags and description..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/create\n<<<api_call_type>>>: POST\n<<<params>>>: {"name": "MyModel", "tags": [], "description": "}\n<<<explanation>>>: Register models under the name \'MyModel\' with optional tags and an empty description.. <s>',
                'Instruct: Get details of the registered model with the name "MyModel"..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/get\n<<<api_call_type>>>: GET\n<<<params>>>: {"name": "MyModel"}\n<<<explanation>>>: Retrieve information about the registered model with the name \'MyModel\'.. <s>',
                'Instruct: Rename the registered model with the name "MyModel" to "UpdatedModelName"..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/rename\n<<<api_call_type>>>: POST\n<<<params>>>: {"name": "MyModel", "new_name": "UpdatedModelName"}\n<<<explanation>>>: Use this API to rename the registered model with the name \'MyModel\' to \'UpdatedModelName\'.. <s>',
                'Instruct: Update the description of the registered model with the name "MyModel" to "Updated description"..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/update\n<<<api_call_type>>>: PATCH\n<<<params>>>: {"name": "MyModel", "description": "Updated description"}\n<<<explanation>>>: If provided, update the description for the registered model with the name \'MyModel\' to \'Updated description\'.. <s>',
                'Instruct: Delete the registered model with the name "MyModel"..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/delete\n<<<api_call_type>>>: DELETE\n<<<params>>>: {"name": "MyModel"}\n<<<explanation>>>: Delete the registered model with the name \'MyModel\'.. <s>',
                'Instruct: Search for model versions based on a filter condition and order the results by the last updated timestamp..\nOutput: <<<api_call>>>: 2.0/mlflow/model-versions/search\n<<<api_call_type>>>: GET\n<<<params>>>: {"filter": "name LIKE \'my-model-name\'", "max_results": 100, "order_by": ["last_updated"]}\n<<<explanation>>>: Search for model versions with a filter condition, like \'name LIKE \'my-model-name\'\', order the results by the last updated timestamp, and limit to a maximum of 100 results.. <s>',
                'Instruct: As a data scientist, create a new experiment named "HyperparameterTuning" with a custom artifact location..\nOutput: <<<api_call>>>: 2.0/mlflow/experiments/create\n<<<api_call_type>>>: POST\n<<<params>>>: {\n    "name": "HyperparameterTuning",\n    "artifact_location": "/data_scientist/experiments"\n}\n<<<explanation>>>: This API call allows data scientists to create a new MLflow experiment named "HyperparameterTuning" with a custom artifact location set to "/data_scientist/experiments".. <s>',
                'Instruct: As a machine learning engineer, search for MLflow experiments with a maximum of 10 results, filtered by name \'ProductionModels\', ordered by start time, and showing all experiments including inactive ones..\nOutput: <<<api_call>>>: 2.0/mlflow/experiments/search\n<<<api_call_type>>>: POST\n<<<params>>>: {"max_results": 10, "filter": "name=\'ProductionModels\'", "order_by": ["start_time"], "view_type": "ALL"}\n<<<explanation>>>: Use this API to enable machine learning engineers to search for MLflow experiments with a maximum of 10 results, filtered by name \'ProductionModels\', ordered by start time, and showing all experiments, including inactive ones.. <s>',
                'Instruct: As a DevOps engineer, retrieve details of the MLflow experiment with ID 789 to monitor its progress..\nOutput: <<<api_call>>>: 2.0/mlflow/experiments/get\n<<<api_call_type>>>: GET\n<<<params>>>: {"experiment_id": "789"}\n<<<explanation>>>: This API call is useful for DevOps engineers to retrieve detailed information about the MLflow experiment with ID 789, aiding in monitoring its progress.. <s>',
                'Instruct: As a project manager, update the name of the MLflow experiment with ID 234 to "ProjectX_Exp1"..\nOutput: <<<api_call>>>: 2.0/mlflow/experiments/update\n<<<api_call_type>>>: POST\n<<<params>>>: {"experiment_id": "234", "new_name": "ProjectX_Exp1"}\n<<<explanation>>>: Use this API to allow project managers to update the name of the MLflow experiment with ID 234 to "ProjectX_Exp1".. <s>',
                'Instruct: As a model reviewer, register a new model named "SentimentAnalysisModel" with optional tags and description..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/create\n<<<api_call_type>>>: POST\n<<<params>>>: {"name": "SentimentAnalysisModel", "tags": ["NLP", "SentimentAnalysis"], "description": "Model for sentiment analysis"}\n<<<explanation>>>: Model reviewers can use this API to register a new model named "SentimentAnalysisModel" with tags [\'NLP\', \'SentimentAnalysis\'] and a description indicating it\'s a model for sentiment analysis.. <s>',
                'Instruct: As a data engineer, get details of the registered model with the name "CustomerChurnPrediction"..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/get\n<<<api_call_type>>>: GET\n<<<params>>>: {"name": "CustomerChurnPrediction"}\n<<<explanation>>>: This API call is designed for data engineers to retrieve detailed information about the registered model named "CustomerChurnPrediction".. <s>',
                'Instruct: As a model deployer, rename the registered model with the name "ImageClassificationModel" to "UpdatedImageModel"..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/rename\n<<<api_call_type>>>: POST\n<<<params>>>: {"name": "ImageClassificationModel", "new_name": "UpdatedImageModel"}\n<<<explanation>>>: Model deployers can use this API to rename the registered model with the name "ImageClassificationModel" to "UpdatedImageModel".. <s>',
                'Instruct: As a project lead, update the description of the registered model with the name "CustomerSegmentation" to "Updated model for segmenting customer data"..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/update\n<<<api_call_type>>>: PATCH\n<<<params>>>: {"name": "CustomerSegmentation", "description": "Updated model for segmenting customer data"}\n<<<explanation>>>: Project leads can use this API to update the description of the registered model with the name "CustomerSegmentation" to "Updated model for segmenting customer data".. <s>',
                'Instruct: As a model administrator, delete the registered model with the name "LegacyModel"..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/delete\n<<<api_call_type>>>: DELETE\n<<<params>>>: {"name": "LegacyModel"}\n<<<explanation>>>: Model administrators can use this API to delete the registered model with the name "LegacyModel".. <s>',
                'Instruct: As a data scientist, search for model versions based on a filter condition and order the results by the creation timestamp..\nOutput: <<<api_call>>>: 2.0/mlflow/model-versions/search\n<<<api_call_type>>>: GET\n<<<params>>>: {"filter": "name LIKE \'image-classifier\'", "max_results": 50, "order_by": ["creation_timestamp"]}\n<<<explanation>>>: Data scientists can use this API to search for model versions with a filter condition, like \'name LIKE \'image-classifier\'\', order the results by the creation timestamp, and limit to a maximum of 50 results.. <s>']

        train_dataset = Dataset.from_dict({'text': data[:14]})
        eval_dataset = Dataset.from_dict({'text': data[14:]})

        max_length = 512  # This was an appropriate max length for my dataset

        def generate_and_tokenize_prompt2(prompt):
            result = tokenizer(
                prompt['text'],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result


        tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt2)
        tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt2)

        # -----------------

        set_status(args['job_id'], 'configuring lora settings')

        config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=[
                'Wqkv',
                'out_proj',
                "fc1",
                "fc2"
            ],
            # target_modules=[
            #     "Wqkv",
            #     "fc1",
            #     "fc2",
            # ],
            bias="none",
            lora_dropout=0.1,  # Conventional
            task_type="CAUSAL_LM",
        )
        model.config.gradient_checkpointing = False
        model = get_peft_model(model, config)
        # print_trainable_parameters(model)

        # -------------------

        model = accelerator.prepare_model(model)

        # -------
        set_status(args['job_id'], 'training started')

        if torch.cuda.device_count() > 1:  # If more than 1 GPU
            model.is_parallelizable = True
            model.model_parallel = True
        project = "best"
        base_model_name = "phi2"
        run_name = base_model_name + "-" + project
        output_dir = "./" + run_name

        trainer = transformers.Trainer(
            model=model,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            args=transformers.TrainingArguments(
                output_dir=output_dir,
                warmup_steps=1,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=3,
                num_train_epochs=20,
                learning_rate=2.3e-5,  # Want a small lr for finetuning
                optim="paged_adamw_8bit",
                logging_steps=25,              # When to start reporting loss
                logging_dir="./logs",        # Directory for storing logs
                evaluation_strategy="steps",  # Evaluate the model every logging step
                eval_steps=25,               # Evaluate and save checkpoints every 50 steps
                do_eval=True,                # Perform evaluation at the end of training
                overwrite_output_dir = 'True',
                load_best_model_at_end=False,
                save_strategy = "no",
                save_total_limit = 1,
                run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(
                tokenizer, mlm=False),
        )

        # silence the warnings. Please re-enable for inference!
        model.config.use_cache = False
        trainer.train()

        set_status(args['job_id'], 'saving model')
        trainer.model.push_to_hub(repo_name)
        set_status(args['job_id'], 'training finished')
        
    except Exception as e:
        set_status(args['job_id'], f'error : {e[:200]}')

if __name__ == '__main__':
    main()
