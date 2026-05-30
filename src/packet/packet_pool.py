"""
作用：
    管理 social packet 的通信池。第一版只实现 all-to-all 方式，
    即每个 agent 接收除自己外所有 sender 发来的 packet。

主要部分：
    1. PacketPool:
       - add_packet(packet): 向通信池中加入一个 packet
       - get_all_packets(): 获取所有 packet
       - get_packets_for_receiver(receiver_id): 获取某个 receiver 可接收的 packet

输入输出：
    - 输入：SocialPacket 对象
    - 输出：packet 列表
"""

from typing import List

from src.packet.packet_dataclass import SocialPacket


class PacketPool:
    def __init__(self):
        self.packets: List[SocialPacket] = []

    def add_packet(self, packet: SocialPacket) -> None:
        self.packets.append(packet)

    def get_all_packets(self) -> List[SocialPacket]:
        return self.packets

    def get_packets_for_receiver(self, receiver_id: str) -> List[SocialPacket]:
        """
        第一版 all-to-all：
        receiver 接收所有非自己发送的 packet。
        """
        return [pkt for pkt in self.packets if pkt.sender_id != receiver_id]