import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import scapy.all as scapy

MODEL_NAME = "gpt2"

# Define TANTGen model based on GPT-2
class TANTGen(nn.Module):
    def __init__(self):
        super(TANTGen, self).__init__()
        # Load tokenizer and pre-trained model
        self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask).logits
    
    def generate_adversarial_traffic(self, prompt, max_length=100):
        """ Generates adversarial traffic from an input prompt. """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones_like(input_ids)
        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def train_model(self, train_data, epochs=3):
        """ Train the model with the given data. """
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_data:
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            avg_loss = total_loss / len(train_data)
            print(f"*** Epoch {epoch+1}, Loss: {avg_loss:.4f}")

tantgen = TANTGen()

# Example dataset of traffic
training_data = [
    "TCP SYN Packet: src_ip=192.168.1.1, dst_ip=192.168.1.2, src_port=12345, dst_port=80, flags=SYN",
    "TCP SYN-ACK Packet: src_ip=192.168.1.2, dst_ip=192.168.1.1, src_port=80, dst_port=12345, flags=SYN, ACK",
    "UDP Flood: src_ip=10.0.0.1, dst_ip=10.0.0.2, src_port=53, dst_port=53",
    "ICMP Echo Request: src_ip=192.168.0.1, dst_ip=8.8.8.8, icmp_type=8, icmp_code=0"
]

# Train the model
tantgen.train_model(training_data, epochs=3)

# Example usage
prompt = "TCP SYN Packet: { src_ip=192.168.1.1, dst_ip=192.168.1.2, src_port=12345, dst_port=80, flags=SYN }"
print("*** Prompt:", prompt)
adv_traffic = tantgen.generate_adversarial_traffic(prompt)
print("*** Generated Adversarial Traffic:", adv_traffic)

# # Function to manipulate network packets using Scapy
# def modify_packet(packet):
#     """ Modifies network packets to create adversarial examples. """
#     if packet.haslayer(scapy.IP):
#         packet[scapy.IP].ttl = 64  # Example modification
#     return packet

# # Example of capturing and modifying packets
# def capture_and_modify():
#     """ Captures network packets and applies modifications. """
#     packets = scapy.sniff(count=10)
#     modified_packets = [modify_packet(pkt) for pkt in packets]
#     scapy.wrpcap("modified_traffic.pcap", modified_packets)

# capture_and_modify()