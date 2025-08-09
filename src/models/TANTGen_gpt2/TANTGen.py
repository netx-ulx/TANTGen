import torch
import torch.nn as nn
import argparse
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling
)

MODEL_NAME = "gpt2"
OUTPUT_DIR = "./gpt2-finetuned-traffic"

# Define TANTGen model based on GPT-2
class TANTGen(nn.Module):
    def __init__(self):
        super(TANTGen, self).__init__()

        # Load tokenizer and pre-trained model
        self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        # Set pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def read_model(self, model_path):
        """ Load a pre-trained model from a specified path. """
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
    
    def generate_adversarial_traffic(self, prompt, max_new_tokens=150):
        """ Generates adversarial traffic from an input prompt. """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones_like(input_ids)
        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def generate_edit_instructions(self, packet_summaries, max_new_tokens=150):
        prompt = "\n".join(packet_summaries)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id
        )
        instructions = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return instructions.splitlines()
    
    def fine_tune(self, dataset_path, output_dir=OUTPUT_DIR, epochs=5):
        """Fine-tune the model on a text file with network traffic lines."""

        def load_dataset(file_path, tokenizer, block_size=128):
            return TextDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=block_size
            )

        def get_data_collator(tokenizer):
            return DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )

        print("*** Loading dataset...")
        dataset = load_dataset(dataset_path, self.tokenizer)
        data_collator = get_data_collator(self.tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            logging_steps=50
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset
        )

        print("*** Starting fine-tuning...")
        trainer.train()

        print(f"*** Saving model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument('-m', '--mode' ,
                       type=str,
                       default='test',
                       help="Mode: train or test")
    
    parse.add_argument('-d', '--dataset',
                       type=str,
                       default='dataset.txt',
                       help="Path to the dataset for training")
    
    parse.add_argument('-mp', '--model_path',
                       type=str,
                       default='./gpt2-finetuned-traffic',
                       help="Path to the trained model")
    
    parse.add_argument('-p', '--prompt',
                       type=str,
                       default="TCP Packet: src_ip=10.0.0.1, dst_ip=10.0.0.2, src_port=46734, dst_port=23, flags=S",
                       help="Input prompt for generating adversarial traffic")
    
    arg = parse.parse_args()

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("*** Using GPU for training/testing.")
    else:
        device = torch.device("cpu")
        print("*** Using CPU for training/testing.")
    # Initialize the TANTGen model
    tantgen = TANTGen()
    tantgen.to(device)

    # Training mode
    if arg.mode == "train1" or arg.mode == "train":
        print("*** Fine-tuning the model...")
        tantgen.fine_tune(dataset_path=arg.dataset, output_dir=OUTPUT_DIR, epochs=3)
    elif arg.mode == "train2":
        print("*** Fine-tuning the model...")
        tantgen.read_model(arg.model_path)
        tantgen.fine_tune(dataset_path=arg.dataset, output_dir=OUTPUT_DIR, epochs=3)
    # Testing mode
    elif arg.mode == "test":
        print("*** Loading the model...")
        tantgen.read_model(arg.model_path)
    else:
        raise ValueError("Invalid mode. Use 'train' or 'test'.")

    # Prompt for generating adversarial traffic
    print("*** Prompt:", arg.prompt)

    # Generate adversarial traffic
    generated = tantgen.generate_adversarial_traffic(arg.prompt)
    print("*** Generated Adversarial Traffic:", generated)

    # # save generated instructions to a file
    # with open("instructions.txt", "w") as f:
    #     f.write(generated)
    # Uncomment the following lines to save the generated instructions to a file
    # with open("generated_instructions.txt", "w") as f:
    #     for line in generated_list:
    #         f.write(line + "\n")
  

    # generated_list = generated.split("\n")
    # apply_modifications("original_traffic.pcap", "instructions.txt", output_pcap="generated_traffic.pcap")