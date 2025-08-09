from scapy.all import rdpcap, raw, IP, TCP, UDP, ICMP, ARP
from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether

def fast_read_pcap(filename):
    for pkt_data, _ in RawPcapReader(filename):
        yield Ether(pkt_data)