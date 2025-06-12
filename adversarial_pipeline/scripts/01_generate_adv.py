import os
from models.TANTGen_gpt2.TANTGen import TANTGen

input_pcap = "pcaps/original/Mirai.pcap"
output_pcap = "pcaps/adversarial/Mirai_adv.pcap"

model = TANTGen()
model.read_model("models/TANTGen_gpt2/gpt2-finetuned-traffic")
adv_pcap = model.generate_adversarial_traffic(input_pcap)
adv_pcap.save(output_pcap)

print(f"Adversarial PCAP saved in {output_pcap}")