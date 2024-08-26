# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import inspect
# import os
import random
import sys
import pyportus as portus

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# grandparentdir = os.path.dirname(parentdir)
# sys.path.insert(0, parentdir)
# sys.path.insert(0, grandparentdir)
    
import sender_obs
from simple_arg_parse import arg_or_default
import loaded_agent
import time

if not hasattr(sys, 'argv'):
    sys.argv  = ['']

MIN_RATE = 0.5
MAX_RATE = 300.0
DELTA_SCALE = 0.05

RESET_RATE_MIN = 5.0
RESET_RATE_MAX = 100.0

RESET_RATE_MIN = 6.0
RESET_RATE_MAX = 6.0

MODEL_PATH = arg_or_default("--model-path", "./model_0.6_0.3_0.1")

for arg in sys.argv:
    arg_str = "NULL"
    try:
        arg_str = arg[arg.rfind("=") + 1:]
    except:
        pass

    if "--reset-target-rate=" in arg:
        RESET_RATE_MIN = float(arg_str)
        RESET_RATE_MAX = float(arg_str)

class PccGymDriver():
    
    flow_lookup = {}
    
    def __init__(self, flow_id, datapath, datapath_info):
        global RESET_RATE_MIN
        global RESET_RATE_MAX

        self.id = flow_id
        
        self.datapath = datapath
        self.datapath_info = datapath_info

        self.rate = random.uniform(RESET_RATE_MIN, RESET_RATE_MAX)
        
        self.datapath.set_program("default", [("Rate", self.rate * 1e6)])
        self.start_flow_time = time.time()
        self.send_start_time = self.start_flow_time
        self.recv_start_time = self.start_flow_time
        self.recv_end_time = self.start_flow_time
        self.first_acked = true
        
        self.history_len = arg_or_default("--history-len", 10)
        self.features = arg_or_default("--input-features",
                                       default="sent latency inflation,"
                                             + "latency ratio,"
                                             + "send ratio").split(",")
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features,
                                                self.id)
        self.got_data = False

        self.weights = arg_or_default("--weight_throughput", 0.6), arg_or_default("--weight_delay", 0.3), arg_or_default("--weight_loss", 0.1)

        self.agent = loaded_agent.LoadedModelAgent(MODEL_PATH, self.weights)

        PccGymDriver.flow_lookup[flow_id] = self

    def get_rate(self):
        if self.has_data():
            rate_delta = self.agent.act(self.history.as_array())
            self.rate = apply_rate_delta(self.rate, rate_delta)
        return self.rate * 1e6

    def has_data(self):
        return self.got_data

    def set_current_rate(self, new_rate):
        self.current_rate = new_rate

    def record_observation(self, new_mi):
        self.history.step(new_mi)
        self.got_data = True

    def reset_rate(self):
        self.current_rate = random.uniform(RESET_RATE_MIN, RESET_RATE_MAX)

    def reset_history(self):
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features,
                                                self.id)
        self.got_data = False

    def reset(self):
        self.agent.reset()
        self.reset_rate()
        self.reset_history()

    def give_sample(self, bytes_sent, bytes_acked, bytes_lost,
                    send_start_time, send_end_time, recv_start_time,
                    recv_end_time, rtt_samples, packet_size, utility):
        self.record_observation(
            sender_obs.SenderMonitorInterval(
                self.id,
                bytes_sent=bytes_sent,
                bytes_acked=bytes_acked,
                bytes_lost=bytes_lost,
                send_start=send_start_time,
                send_end=send_end_time,
                recv_start=recv_start_time,
                recv_end=recv_end_time,
                rtt_samples=rtt_samples,
                packet_size=packet_size
            )
        )

    def get_by_flow_id(flow_id):
        return PccGymDriver.flow_lookup[flow_id]
    
    def on_report(self, r):
        if r.finish_interval:
            bytes_lost = self.datapath_info.mss * r.packets_lost
            bytes_sent = r.bytes_acked + r.bytes_sacked + r.bytes_lost + r.bytes_in_flight
            packets_sent = r.packets_in_flight + r.packets_acked + r.packets_lost
            utility = r.bytes_acked * 8 / (self.recv_end_time - self.recv_start_time)
            self.give_sample(bytes_sent, r.bytes_acked, r.bytes_lost, self.send_start_time - self.start_flow_time, time.time() - self.start_flow_time, self.recv_start_time - self.start_flow_time, self.recv_end_time - self.start_flow_time, r.rtt_samples / 1000000, packets_sent, utility)
            updated_rate = self.get_rate()
            self.send_start_time = time.time()
            self.datapath.update_field("Rate", updated_rate)
        
        else:
            if self.first_acked:
                recv_start_time = time.time()
                recv_end_time = recv_start_time
                self.first_acked = False
            else:
                recv_end_time = time.time()
        
class MOCC(portus.AlgBase):
    def datapath_programs(self):
        return {
                "default" : """\
                (def 
                    (Report
                        (volatile bytes_in_flight 0)
                        (bytes_acked 0)
                        (bytes_sacked 0)
                        (volatile packets_lost 0)
                        (packets_acked 0)
                        (volatile packets_in_flight 0)
                        (finish_interval false)
                        (volatile rtt_samples 0)
                    )
                )
                (when true
                    (:= Report.bytes_in_flight Flow.bytes_in_flight)
                    (:= Report.bytes_acked (+ Report.bytes_acked Ack.bytes_acked))
                    (:= Report.bytes_sacked (+ Report.bytes_sacked Ack.packets_misordered))
                    (:= Report.packets_lost Ack.lost_pkts_sample)
                    (:= Report.packets_acked (+ Report.packets_acked Ack.packets_acked))
                    (:= Report.packets_in_flight Flow.packets_in_flight)
                    (:= Report.rtt_samples Flow.rtt_sample_us)
                    (report)
                    (fallthrough)
                )
                (when (> Micros Flow.rtt_sample_us)
                    (:= Report.finish_interval true)
                    (report)
                    (:= Report.finish_interval false)
                    (:= Micros 0)
                )
            """
        }


def give_sample(flow_id, bytes_sent, bytes_acked, bytes_lost,
                send_start_time, send_end_time, recv_start_time,
                recv_end_time, rtt_samples, packet_size, utility):
    driver = PccGymDriver.get_by_flow_id(flow_id)
    driver.give_sample(bytes_sent, bytes_acked, bytes_lost,
                       send_start_time, send_end_time, recv_start_time,
                       recv_end_time, rtt_samples, packet_size, utility)

def apply_rate_delta(rate, rate_delta):
    global MIN_RATE
    global MAX_RATE
    global DELTA_SCALE
    
    rate_delta *= DELTA_SCALE

    # We want a string of actions with average 0 to result in a rate change
    # of 0, so delta of 0.05 means rate * 1.05,
    # delta of -0.05 means rate / 1.05
    if rate_delta > 0:
        rate *= (1.0 + rate_delta)
    elif rate_delta < 0:
        rate /= (1.0 - rate_delta)
    
    # For practical purposes, we may have maximum and minimum rates allowed.
    if rate < MIN_RATE:
        rate = MIN_RATE
    if rate > MAX_RATE:
        rate = MAX_RATE

    return rate
    
def reset(flow_id):
    driver = PccGymDriver.get_by_flow_id(flow_id)
    driver.reset()

def get_rate(flow_id):
    #print("Getting rate")
    driver = PccGymDriver.get_by_flow_id(flow_id)
    return driver.get_rate()

def init(flow_id):
    driver = PccGymDriver(flow_id)


