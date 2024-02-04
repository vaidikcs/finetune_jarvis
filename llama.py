import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer

from datasets import load_dataset
from huggingface_hub import create_repo
import subprocess, os
# -------------------

def main():
    parser = argparse.ArgumentParser(description='pass params for finetuning.')
    parser.add_argument('--train_data', help='train data')
    parser.add_argument('--val_data', help='val data')
    parser.add_argument(
        '--lora_rank', default='16', help='LoRA Rank')
    parser.add_argument(
        '--epochs', default='5', help='Train epochs')
    parser.add_argument(
        '--batch', default='2', help='train batch size')
    parser.add_argument(
        '--learning_rate', default=2.3e-5, help='learning rate')
    parser.add_argument(
        '--final_model', help='Path to output file')
    parser.add_argument(
        '--max_length', default='512', help='max length of training data.')
    parser.add_argument(
        '--hf_token', help='huggingface write token')

    args = parser.parse_args()

    token = str(args.hf_token)
    command = f"huggingface-cli login --token {token}"
    subprocess.run(command, shell=True)
    
    repo_name = args.final_model
    create_repo(repo_name, private=True, exist_ok=True)
    print(repo_name)

    try:
        base_model_name = "meta-llama/Llama-2-7b-chat-hf"

        # Tokenizer
        llama_tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True, use_auth_token="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")
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
        llama_tokenizer.padding_side = "right"  # Fix for fp16

        # Quantization Config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False
        )

        # Model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quant_config,
            device_map={"": 0},
            use_auth_token="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"
        )
        base_model.config.use_cache = False
        base_model.config.pretraining_tp = 1
    except Exception as e:
        raise Exception(f"something went wrong while downloading base model. {e}")
    # set_status(args['job_id'], 'dataset loading started')
    # -------------------
    try:
        import re
        def create_prompt(p):
            i = p['instruction']
            e = p['explanation'].replace('\n', ' ')
            i = f"{i}"
            p = "<s>[INST]\n"
            p+=i
            p+="\n[/INST]\n"
            p+=e
            p+="</s>"
            return {'prompts': p}
        
        train_data = str(args.train_data)
        val_data = str(args.val_data)
        train_dataset = load_dataset("csv", data_files=train_data, split="train")
        eval_dataset = load_dataset("csv", data_files=val_data, split="train")
        train_dataset = train_dataset.map(create_prompt)
        eval_dataset = eval_dataset.map(create_prompt)

    except Exception as e:
        raise Exception(f"something went wrong while reading data. {e}")
        # print(e)
        # data = ['Instruct: Create a new experiment named "MLflowDemo" with a custom artifact location..\nOutput: <<<api_call>>>: 2.0/mlflow/experiments/create\n<<<api_call_type>>>: POST\n<<<params>>>: {\n    "name": "MLflowDemo",\n    "artifact_location": "/custom/location"\n}\n<<<explanation>>>: This API call creates a new MLflow experiment named "MLflowDemo" with a custom artifact location set to "/custom/location".. <s>',
        #         'Instruct: Find MLflow experiments with a maximum of 5 results, filtered by name \'MLflowDemo\', ordered by start time, and showing only active experiments..\nOutput: <<<api_call>>>: 2.0/mlflow/experiments/search\n<<<api_call_type>>>: POST\n<<<params>>>: {"max_results": 5, "filter": "name=\'MLflowDemo\'", "order_by": ["start_time"], "view_type": "ACTIVE"}\n<<<explanation>>>: Use this API to search for MLflow experiments with a maximum of 5 results, filtered by name \'MLflowDemo\', ordered by start time, and showing only active experiments.. <s>',
        #         'Instruct: Fetch information about the MLflow experiment with ID 123..\nOutput: <<<api_call>>>: 2.0/mlflow/experiments/get\n<<<api_call_type>>>: GET\n<<<params>>>: {"experiment_id": "123"}\n<<<explanation>>>: Retrieve details of the MLflow experiment with ID 123 using this API call.. <s>',
        #         'Instruct: Update the name of the MLflow experiment with ID 456 to "NewExperimentName"..\nOutput: <<<api_call>>>: 2.0/mlflow/experiments/update\n<<<api_call_type>>>: POST\n<<<params>>>: {"experiment_id": "456", "new_name": "NewExperimentName"}\n<<<explanation>>>: Use this API to update the name of the MLflow experiment with ID 456 to "NewExperimentName".. <s>',
        #         'Instruct: Register a new model under the name "MyModel" with optional tags and description..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/create\n<<<api_call_type>>>: POST\n<<<params>>>: {"name": "MyModel", "tags": [], "description": "}\n<<<explanation>>>: Register models under the name \'MyModel\' with optional tags and an empty description.. <s>',
        #         'Instruct: Get details of the registered model with the name "MyModel"..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/get\n<<<api_call_type>>>: GET\n<<<params>>>: {"name": "MyModel"}\n<<<explanation>>>: Retrieve information about the registered model with the name \'MyModel\'.. <s>',
        #         'Instruct: Rename the registered model with the name "MyModel" to "UpdatedModelName"..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/rename\n<<<api_call_type>>>: POST\n<<<params>>>: {"name": "MyModel", "new_name": "UpdatedModelName"}\n<<<explanation>>>: Use this API to rename the registered model with the name \'MyModel\' to \'UpdatedModelName\'.. <s>',
        #         'Instruct: Update the description of the registered model with the name "MyModel" to "Updated description"..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/update\n<<<api_call_type>>>: PATCH\n<<<params>>>: {"name": "MyModel", "description": "Updated description"}\n<<<explanation>>>: If provided, update the description for the registered model with the name \'MyModel\' to \'Updated description\'.. <s>',
        #         'Instruct: Delete the registered model with the name "MyModel"..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/delete\n<<<api_call_type>>>: DELETE\n<<<params>>>: {"name": "MyModel"}\n<<<explanation>>>: Delete the registered model with the name \'MyModel\'.. <s>',
        #         'Instruct: Search for model versions based on a filter condition and order the results by the last updated timestamp..\nOutput: <<<api_call>>>: 2.0/mlflow/model-versions/search\n<<<api_call_type>>>: GET\n<<<params>>>: {"filter": "name LIKE \'my-model-name\'", "max_results": 100, "order_by": ["last_updated"]}\n<<<explanation>>>: Search for model versions with a filter condition, like \'name LIKE \'my-model-name\'\', order the results by the last updated timestamp, and limit to a maximum of 100 results.. <s>',
        #         'Instruct: As a data scientist, create a new experiment named "HyperparameterTuning" with a custom artifact location..\nOutput: <<<api_call>>>: 2.0/mlflow/experiments/create\n<<<api_call_type>>>: POST\n<<<params>>>: {\n    "name": "HyperparameterTuning",\n    "artifact_location": "/data_scientist/experiments"\n}\n<<<explanation>>>: This API call allows data scientists to create a new MLflow experiment named "HyperparameterTuning" with a custom artifact location set to "/data_scientist/experiments".. <s>',
        #         'Instruct: As a machine learning engineer, search for MLflow experiments with a maximum of 10 results, filtered by name \'ProductionModels\', ordered by start time, and showing all experiments including inactive ones..\nOutput: <<<api_call>>>: 2.0/mlflow/experiments/search\n<<<api_call_type>>>: POST\n<<<params>>>: {"max_results": 10, "filter": "name=\'ProductionModels\'", "order_by": ["start_time"], "view_type": "ALL"}\n<<<explanation>>>: Use this API to enable machine learning engineers to search for MLflow experiments with a maximum of 10 results, filtered by name \'ProductionModels\', ordered by start time, and showing all experiments, including inactive ones.. <s>',
        #         'Instruct: As a DevOps engineer, retrieve details of the MLflow experiment with ID 789 to monitor its progress..\nOutput: <<<api_call>>>: 2.0/mlflow/experiments/get\n<<<api_call_type>>>: GET\n<<<params>>>: {"experiment_id": "789"}\n<<<explanation>>>: This API call is useful for DevOps engineers to retrieve detailed information about the MLflow experiment with ID 789, aiding in monitoring its progress.. <s>',
        #         'Instruct: As a project manager, update the name of the MLflow experiment with ID 234 to "ProjectX_Exp1"..\nOutput: <<<api_call>>>: 2.0/mlflow/experiments/update\n<<<api_call_type>>>: POST\n<<<params>>>: {"experiment_id": "234", "new_name": "ProjectX_Exp1"}\n<<<explanation>>>: Use this API to allow project managers to update the name of the MLflow experiment with ID 234 to "ProjectX_Exp1".. <s>',
        #         'Instruct: As a model reviewer, register a new model named "SentimentAnalysisModel" with optional tags and description..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/create\n<<<api_call_type>>>: POST\n<<<params>>>: {"name": "SentimentAnalysisModel", "tags": ["NLP", "SentimentAnalysis"], "description": "Model for sentiment analysis"}\n<<<explanation>>>: Model reviewers can use this API to register a new model named "SentimentAnalysisModel" with tags [\'NLP\', \'SentimentAnalysis\'] and a description indicating it\'s a model for sentiment analysis.. <s>',
        #         'Instruct: As a data engineer, get details of the registered model with the name "CustomerChurnPrediction"..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/get\n<<<api_call_type>>>: GET\n<<<params>>>: {"name": "CustomerChurnPrediction"}\n<<<explanation>>>: This API call is designed for data engineers to retrieve detailed information about the registered model named "CustomerChurnPrediction".. <s>',
        #         'Instruct: As a model deployer, rename the registered model with the name "ImageClassificationModel" to "UpdatedImageModel"..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/rename\n<<<api_call_type>>>: POST\n<<<params>>>: {"name": "ImageClassificationModel", "new_name": "UpdatedImageModel"}\n<<<explanation>>>: Model deployers can use this API to rename the registered model with the name "ImageClassificationModel" to "UpdatedImageModel".. <s>',
        #         'Instruct: As a project lead, update the description of the registered model with the name "CustomerSegmentation" to "Updated model for segmenting customer data"..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/update\n<<<api_call_type>>>: PATCH\n<<<params>>>: {"name": "CustomerSegmentation", "description": "Updated model for segmenting customer data"}\n<<<explanation>>>: Project leads can use this API to update the description of the registered model with the name "CustomerSegmentation" to "Updated model for segmenting customer data".. <s>',
        #         'Instruct: As a model administrator, delete the registered model with the name "LegacyModel"..\nOutput: <<<api_call>>>: 2.0/mlflow/registered-models/delete\n<<<api_call_type>>>: DELETE\n<<<params>>>: {"name": "LegacyModel"}\n<<<explanation>>>: Model administrators can use this API to delete the registered model with the name "LegacyModel".. <s>',
        #         'Instruct: As a data scientist, search for model versions based on a filter condition and order the results by the creation timestamp..\nOutput: <<<api_call>>>: 2.0/mlflow/model-versions/search\n<<<api_call_type>>>: GET\n<<<params>>>: {"filter": "name LIKE \'image-classifier\'", "max_results": 50, "order_by": ["creation_timestamp"]}\n<<<explanation>>>: Data scientists can use this API to search for model versions with a filter condition, like \'name LIKE \'image-classifier\'\', order the results by the creation timestamp, and limit to a maximum of 50 results.. <s>']
    
        # train_dataset = Dataset.from_dict({'text': data[:14]})
        # eval_dataset = Dataset.from_dict({'text': data[14:]})

      # This was an appropriate max length for my dataset
    try:
        max_length = int(args.max_length)
        
        def generate_and_tokenize_prompt2(prompt):
            result = llama_tokenizer(
                prompt['text'],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result
    
        train_dataset = train_dataset.map(generate_and_tokenize_prompt2)
        eval_dataset = eval_dataset.map(generate_and_tokenize_prompt2)
    except Exception as e:
        raise Exception(f"something went wrong while parsing train data. {e}")

    # -----------------
    rank = int(args.lora_rank)
    try:
        peft_parameters = LoraConfig(
            lora_alpha=rank*2,
            r=rank,
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            task_type="CAUSAL_LM"
        )

    except Exception as e:
        raise Exception(f"something went wrong while setting up LoRA. {e}")
    # print_trainable_parameters(model)

    # ------------------
    project = "best"
    base_model_name = "llama"
    run_name = base_model_name + "-" + project
    output_dir = "./" + run_name

    try:
        epochs = int(args.epochs)
        batch = int(args.batch)
        learning_rate = float(args.learning_rate)
        
        train_params = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=2,
            optim="paged_adamw_32bit",
            eval_steps=25,
            logging_steps=25,
            learning_rate=learning_rate,
            weight_decay=0.001,
            fp16=False,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            do_eval=True,
            warmup_ratio=0.03,
            group_by_length=True,
            overwrite_output_dir='True',
            save_strategy="steps",       # Save the model checkpoint every logging step
            save_steps=10,
            save_total_limit=1,
            lr_scheduler_type="cosine"
        )
    
        fine_tuning = SFTTrainer(
            model=base_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_parameters,
            dataset_text_field="prompts",
            tokenizer=llama_tokenizer,
            args=train_params
        )
        fine_tuning.train()
    except Exception as e:
        raise Exception(f"something went wrong during training. {e}")

    # set_status(args['job_id'], 'saving model')
    try:
        del fine_tuning, base_model

        from peft import LoraConfig, AutoPeftModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoPeftModelForCausalLM.from_pretrained(
            path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="cuda")
    
        path = os.listdir(run_name)[-1]
        print(run_name+ '/'+path)
        
        model.push_to_hub(repo_id=repo_name)
        print('****')
        tokenizer.push_to_hub(repo_id=repo_name)
        print('finished')
    except Exception as e:
        raise Exception(f"something went wrong while saving model. {e}")
        

if __name__ == '__main__':
    main()
