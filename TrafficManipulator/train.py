import numpy as np
import pickle as pkl
from scapy.all import *
import argparse

from FEKitsune import Kitsune
from KitsuneTools import *
from utils import *

import platform
import os
import sys

import datetime
from decimal import *
import argparse

# statistics for eval

STA_X_list = []
STA_feature_list = []
STA_pktList_list = []
STA_gbl_dis_list = []
STA_avg_dis_list = []
STA_all_feature_list = []


class Manipulator:

    # Manipulator Parameters
    grp_size = 5
    min_time_extend = 0.
    max_time_extend = 5.
    max_cft_pkt = 5
    max_crafted_pkt_prob = 1.

    # Data Members
    pktList = []
    global_FE = None
    mimic_set = None

    def __init__(
            self,
            mal_pcap_file,
            mimic_set,
            knormer_file,
            init_pcap_file="./data/empty.pcap",  # preparatory traffic
    ):
        print("@Manipulator: Initializing ...")

        self.mimic_set = np.load(mimic_set)

        print("self.mimic_set.shape", self.mimic_set.shape)

        # Normalizer
        with open(knormer_file, 'rb') as f:
            self.knormer = pkl.load(f)

        self.pktList = rdpcap(mal_pcap_file)
        print("  read %d packets in malicious pcap" % (len(self.pktList)))

        # Create global feature extractor
        init_scapy_in = rdpcap(init_pcap_file)
        self.global_FE = Kitsune(init_scapy_in, np.Inf)

        # compile preparatory traffic if exists
        if init_pcap_file != "./data/empty.pcap":
            RunFE(self.global_FE)

    def save_mutated_traffic(self, mut_pcap_path):
        global STA_X_list
        global STA_pktList_list

        true_time = []
        for x in STA_X_list:
            for i in range(x.mal.shape[0]):
                for j in range(int(round(x.mal[i][1]))):
                    true_time.append(x.mal[i][0] - x.craft[i][j][0])
                true_time.append(x.mal[i][0])
        cnt = 0
        pkt_List = []
        for p in STA_pktList_list:
            for pp in p:
                pp.time = true_time[cnt]
                cnt = cnt + 1
                pkt_List.append(pp)
                
        wrpcap(mut_pcap_path, pkt_List)

    def process(
        self,
        sta_file,
        start_no=0,
        limit=np.Inf,
        heuristic=False,
    ):

        # Timers
        FE_time = 0
        import time
        timer = time.time()

        acc_ics_time = 0
        last_end_time = float(self.pktList[0].time)
        begin_timestamp = float(self.pktList[0].time)

        st = start_no
        ed = st + self.grp_size

        print("@Mani: Begin processing...")
        while True:
            print("@Manipulator: Processing pkt ( %d to %d ) ..." % (st, ed))

            # print("@Manipulator: Create Transformer")
            # ---- initialize Transformer--------------------------------------------+
            # Create model instance
            tantgen = TANTGen() # (max_iter=self.pso_iter, particle_num=self.pso_num, grp_size=self.pso_size)

            # ---- load a new pkt group--------------------------------------+
            groupList = self.pktList[st:ed]

            # ---- increase initial time of the new pkt group----------------+
            for pkt in groupList:
                # pkt.time += Decimal(acc_ics_time)
                # pkt.time += acc_ics_time
                pkt.time = float(pkt.time) + acc_ics_time

            # ---- execute PSO-----------------------------------------------+
            pso_show_info = True
            if self.grp_size < 50:
                pso_show_info = False
            ics_time, cur_end_time, \
            STA_best_X, STA_best_feature, STA_best_pktList, STA_gbl_dis, STA_avg_dis,STA_best_all_feature,fe_time\
                        = pso.execute(  last_end_time, groupList,
                                        self.max_time_extend,self.max_cft_pkt, self.min_time_extend, self.max_crafted_pkt_prob,
                                        self.mimic_set,
                                        self.global_FE.FE.nstat,
                                        self.knormer,
                                        self.w,self.c1,self.c2,
                                        pso_show_info,heuristic)

            FE_time += fe_time
            # ---- prepare for next pkt group--------------------------------+
            acc_ics_time += ics_time
            last_end_time = cur_end_time

            ttime1 = time.perf_counter()
            nstat = self.global_FE.FE.nstat
            self.global_FE = Kitsune(STA_best_pktList, np.Inf, False)
            self.global_FE.FE.nstat = safelyCopyNstat(nstat, False)
            RunFE(self.global_FE)
            ttime2 = time.perf_counter()
            FE_time += (ttime2 - ttime1)

            # ---- Update statistics ----------------------------------------------+
            global STA_X_list
            global STA_feature_list
            global STA_pktList_list
            global STA_gbl_dis_list
            global STA_avg_dis_list

            STA_X_list.append(STA_best_X)
            STA_feature_list.append(STA_best_feature)
            STA_pktList_list.append(STA_best_pktList)
            STA_gbl_dis_list.append(STA_gbl_dis)
            STA_avg_dis_list.append(STA_avg_dis)
            STA_all_feature_list.append(STA_best_all_feature)

            # ---plt and dump info-------------------------------------------+
            if st != 0 and (st % 1000 == 0
                            or ed == len(self.pktList)) or ed == limit:
                print("@Manipulator:Time elapsed:", time.time() - timer)
                with open(sta_file, "wb") as f:
                    pkl.dump(STA_X_list, f)
                    pkl.dump(STA_feature_list, f)
                    pkl.dump(STA_pktList_list, f)
                    pkl.dump(STA_gbl_dis_list, f)
                    pkl.dump(STA_avg_dis_list, f)
                    pkl.dump(STA_all_feature_list, f)
                print("@Manipulator:statistics.pkl is updated...")

            # ---------------update `st` and `ed` for next loop--------------+
            if ed == len(self.pktList) or ed == limit:
                print("@Manipulator:All Finished!", ed,
                      "Pkts Processed,Time elapsed:",
                      time.time() - timer, "FE_time:", FE_time)
                break

            st = ed
            ed += self.grp_size
            if ed >= len(self.pktList):
                ed = len(self.pktList)
                self.grp_size = ed - st
            if ed >= limit:
                ed = limit
                self.grp_size = ed - st

        self.save_mutated_traffic("example/mutated_traffic.pcap")


parse = argparse.ArgumentParser()
parse.add_argument('-m',
                   '--mal_pcap',
                   type=str,
                   required=True,
                   help="input malicious traffic (.pcap)")

parse.add_argument('-b',
                   '--mimic_set',
                   type=str,
                   required=True,
                   help="benign features to mimic (.npy)")

parse.add_argument('-n',
                   '--normalizer',
                   type=str,
                   required=True,
                   help="compiled feature normalizer (.pkl)")

parse.add_argument('-i',
                   '--init_pcap',
                   type=str,
                   default='./_empty.pcap',
                   help="preparatory traffic (ignore this if you don't need)")

parse.add_argument('-o',
                   '--sta_file',
                   type=str,
                   default='./example/statistics.pkl',
                   help="file saving the final statistics (.pkl)")

arg = parse.parse_args()

m = Manipulator(arg.mal_pcap, arg.mimic_set, arg.normalizer, arg.init_pcap)

m.change_manipulator_params(grp_size=100,
                            min_time_extend=3.,
                            max_time_extend=6.,
                            max_cft_pkt=1,
                            max_crafted_pkt_prob=0.01)

m.process(arg.sta_file, limit=np.Inf, heuristic=False)