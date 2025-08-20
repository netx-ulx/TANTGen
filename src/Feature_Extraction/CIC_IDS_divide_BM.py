import pyshark
import pandas as pd
import nest_asyncio
from scapy.all import RawPcapWriter

path = "../../data/original/CIC-IDS-2017/"
csv_subpath = "CSVs/TrafficLabelling/"
pcap_subpath = "PCAPs/"

# csv_friday_1 = pd.read_csv(path + csv_subpath + 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
# csv_friday_2 = pd.read_csv(path + csv_subpath + 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
# csv_friday_3 = pd.read_csv(path + csv_subpath + 'Friday-WorkingHours-Morning.pcap_ISCX.csv')
# csv_friday = pd.concat([csv_friday_1, csv_friday_2, csv_friday_3])

# Monday is BENIGN

# csv_thursday_1 = pd.read_csv(path + csv_subpath + 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv')
# csv_thursday_2 = pd.read_csv(path + csv_subpath + 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', skip_blank_lines=True)
# csv_thursday = pd.concat([csv_thursday_1, csv_thursday_2])

# csv_tuesday = pd.read_csv(path + csv_subpath + 'Tuesday-WorkingHours.pcap_ISCX.csv')

csv_wednesday = pd.read_csv(path + csv_subpath + 'Wednesday-workingHours.pcap_ISCX.csv')

pcap_files = [
    # path + pcap_subpath + 'Friday-WorkingHours.pcap',
    # path + pcap_subpath + 'Thursday-WorkingHours.pcap',
    # path + pcap_subpath + 'Tuesday-WorkingHours.pcap',
    path + pcap_subpath + 'Wednesday-workingHours.pcap'
]

protocol = {
    "TCP": 6,
    "UDP": 17,
    "ICMP": 1
}

benign_writer = RawPcapWriter(path + pcap_subpath + 'wednesday_benign.pcap', append=False, sync=True)
malicious_writer = RawPcapWriter(path + pcap_subpath + 'wednesday_malicious.pcap', append=False, sync=True)

nest_asyncio.apply()

for df, pcap_file_name in zip([csv_wednesday], pcap_files):

    # Considerando colunas típicas: 'Source IP', 'Destination IP', 'Protocol', 'Label'
    benign_df = df[df[' Label'] == 'BENIGN']

    # Criar um set de 5-tuplas para lookup rápido
    benign_flows = set(zip(
        benign_df[' Source IP'], 
        benign_df[' Destination IP'], 
        benign_df[' Source Port'], 
        benign_df[' Destination Port'], 
        benign_df[' Protocol']
    ))

    # Abrir PCAP
    cap = pyshark.FileCapture(pcap_file_name, keep_packets=False, include_raw=True, use_json=True)

    for pkt in cap:
        try:
            transport_layer = protocol.get(pkt.transport_layer, 0)
            flow = (
                pkt.ip.src,
                pkt.ip.dst,
                int(pkt[pkt.transport_layer].srcport),
                int(pkt[pkt.transport_layer].dstport),
                transport_layer
            )
            if flow in benign_flows:
                benign_writer.write(pkt.get_raw_packet())
            else:
                malicious_writer.write(pkt.get_raw_packet())
        except AttributeError:
            continue

    print(f"Processed {pcap_file_name}")
    cap.close()