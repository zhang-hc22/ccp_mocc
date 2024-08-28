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
import os
import random
import sys
import socket
import json

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

class PccGymDriver():
    
    flow_lookup = {}
    
    def __init__(self, flow_id):
        global RESET_RATE_MIN
        global RESET_RATE_MAX

        self.id = flow_id

        self.rate = random.uniform(RESET_RATE_MIN, RESET_RATE_MAX)
        
        self.start_flow_time = time.time()
        self.send_start_time = self.start_flow_time
        self.recv_start_time = self.start_flow_time
        self.recv_end_time = self.start_flow_time
        self.first_acked = True
        self.first_ack_latency_sec = 0
        self.last_ack_latency_sec = 0
        self.last_call = time.time()
        
        self.history_len = arg_or_default("--history-len", 10)
        self.features = arg_or_default("--input-features",
                                       default="sent latency inflation,"
                                             + "latency ratio,"
                                             + "send ratio").split(",")
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features,
                                                self.id)

        self.weights = arg_or_default("--weight_throughput", 0.6), arg_or_default("--weight_delay", 0.3), arg_or_default("--weight_loss", 0.1)

        print("Initialized MOCC driver with weights: " + str(self.weights))

        model_path = "./model_" + str(self.weights[0]) + "_" + str(self.weights[1]) + "_" + str(self.weights[2])
        
        self.agent = loaded_agent.LoadedModelAgent(model_path, self.weights)
        
        print("Loaded model from " + model_path)

        PccGymDriver.flow_lookup[flow_id] = self

    def get_rate(self):
        rate_delta = self.agent.act(self.history.as_array())
        self.rate = apply_rate_delta(self.rate, rate_delta)
        return self.rate * 1e6

    def set_current_rate(self, new_rate):
        self.current_rate = new_rate

    def record_observation(self, new_mi):
        self.history.step(new_mi)
        self.got_data = True

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
        # 处理收到的数据
        recent_call = time.time()
        print("-------------------------------")
        print("last call: ", self.last_call)
        print("recent call: ", recent_call)
        print("call interval: ", int((recent_call - self.last_call) * 1000000))
        self.last_call = recent_call
        print("inflight: ", r['bytes_in_flight'])
        print("bytes_acked: ", r['bytes_acked'])
        print("packets_lost: ", r['packets_lost'])
        print("bytes_sacked: ", r['bytes_sacked'])
        print("rtt_samples: ", r['rtt_samples'])
        print("finish_interval: ", r['finish_interval'])

        if int(r['finish_interval']) == 1:
            print('update rate')
            bytes_lost = 1440 * r['packets_lost']
            bytes_sent = r['bytes_acked'] + r['bytes_sacked'] + bytes_lost + r['bytes_in_flight']
            utility = 0.0
            if self.recv_end_time == self.recv_start_time:
                self.recv_end_time = time.time()
            rtt_samples = [self.first_ack_latency_sec, self.last_ack_latency_sec]
            self.give_sample(bytes_sent, r['bytes_acked'], bytes_lost, self.send_start_time - self.start_flow_time, time.time() - self.start_flow_time, self.recv_start_time - self.start_flow_time, self.recv_end_time - self.start_flow_time, rtt_samples, 1440, utility)
            updated_rate = self.get_rate()
            self.send_start_time = time.time()
            self.first_acked = True
            print('update rate to', updated_rate)
            print("---------------------------")
            return updated_rate
        
        else:
            print('ack received')
            if self.first_acked:
                self.recv_start_time = time.time()
                self.recv_end_time = self.recv_start_time
                self.first_acked = False
                self.first_ack_latency_sec = r['rtt_samples'] / 1000000
                self.last_ack_latency_sec = self.first_ack_latency_sec
                print("-------------------------------")
            else:
                self.recv_end_time = time.time()
                self.last_ack_latency_sec = r['rtt_samples'] / 1000000
                print('update recv_end_time')
                print("-------------------------------")
            return 0.0

def main():
    # Unix域套接字路径
    socket_path = "/tmp/uds_socket"

    # 确保套接字文件不存在
    if os.path.exists(socket_path):
        os.remove(socket_path)
    # 创建并绑定Unix域套接字
    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_socket.bind("/tmp/uds_socket")
    server_socket.listen(1)
    
    print("Server is listening...")
    
    driver = PccGymDriver(0)

    while True:
        # 等待连接
        conn, _ = server_socket.accept()
        with conn:
            print("Connected to a client.")
            data = conn.recv(1024)
            if not data:
                break
            
            # 将JSON字符串解析为Python字典
            r = json.loads(data.decode('utf-8'))
            
            # 处理接收到的数据并获取返回值
            result = driver.on_report(r)
            
            # 如果有返回值，将其发送回Rust进程
            response = json.dumps({"updated_rate": result})
            conn.sendall(response.encode('utf-8'))
        
    print("Connection closed.")

if __name__ == "__main__":
    main()