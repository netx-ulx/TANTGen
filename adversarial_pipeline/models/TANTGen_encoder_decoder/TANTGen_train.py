import torch
from torch.utils.data import DataLoader
from TANTGen import PacketTransformer
from utils import PacketDataset

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# TODO get_ds
# TODO predict

# TODO: pode ser uma ideia
# Define settings for building and training the transformer model
def get_config():
    return{
        'batch_size': 16,
        'num_epochs': 10,
        'lr': 1e-4,
        'seq_len': 1, # TODO: verify seq_len
        'data_path': 'pcaps/original/Mirai3.csv',
        'model_folder': 'models/TANTGen_encoder_decoder',
        'model_name': 'tantgen_model.pth',
        'experiment_name': 'models/TANTGen_encoder_decoder/TANTGen_encoder_decoder_experiment'
    }


def evaluate(model, dataloader, tokenizer_tgt, device):
    """
    Evaluates the model on the validation set and prints the source, target, and predicted sequences.
    Args:
        model: The trained PacketTransformer model.
        dataloader: DataLoader for the validation set.
        tokenizer_tgt: Tokenizer for the target language (not used in this function).
        device: Device to run the evaluation on (CPU or GPU).
    """

    model.eval()

    # Initializing counter to keep track of how many examples have been processed
    count = 0 
    num_examples = 2 # Number of examples to print during evaluation
    console_width = 80 # Fixed witdh for printed messages
    
    # Creating evaluation loop
    with torch.no_grad(): # Ensuring that no gradients are computed during this process
        for (src, tgt) in dataloader:
            count += 1
            src = src.transpose(0, 1).to(device)
            tgt = tgt.transpose(0, 1).to(device)
            
            # Ensuring that the batch_size of the validation set is 1
            assert src.size(0) ==  1, 'Batch size must be 1 for validation.'
            
            # Predicting the output using the model TODO: verify
            model_out = model(src, tgt)

            # Printing results
            print(f'-'*console_width)
            print(f'SOURCE: {src}')
            print(f'TARGET: {tgt}')
            print(f'PREDICTED: {model_out}')
            
            # After two examples, we break the loop
            if count == num_examples:
                break

def train(config):
    """
    Trains the PacketTransformer model on the dataset specified in the config.
    Args:
        config: Dictionary containing training configuration parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    writer = SummaryWriter(config['experiment_name'])
    global_step = 0

    dataset = PacketDataset(config['data_path'], seq_len=config['seq_len'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model = PacketTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = torch.nn.MSELoss()

    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        batch_iterator = tqdm(dataloader)

        model.train()
        total_loss = 0
        for (src, tgt) in batch_iterator:
            # transpose para [seq_len, batch, features]
            src = src.transpose(0, 1).to(device)
            tgt = tgt.transpose(0, 1).to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Create target mask to prevent attending to future tokens
            tgt_mask = torch.triu(torch.ones(tgt.size(0), tgt.size(0)) * float('-inf'), diagonal=1).to(device)
            
            # Forward pass and loss calculation
            output = model(src, tgt, tgt_mask=tgt_mask)
            loss = loss_fn(output, tgt)

            # Updating progress bar
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
            
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            global_step += 1

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Total Loss = {total_loss / len(dataloader):.4f}")

    # TODO: run validation
    # evaluate(model, dataloader, device)

    # Save the trained model
    torch.save(model.state_dict(), f"{config['model_folder']}/{config['model_name']}")
    print(f"[✓] Modelo treinado e salvo como {config['model_name']}")

if __name__ == "__main__":
    config = get_config()
    train(config)
