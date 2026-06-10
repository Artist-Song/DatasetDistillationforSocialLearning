"""Packet pool helpers for v2 all-to-all agent communication."""

from typing import List

from src.packet.packet_dataclass import SocialPacket


class PacketPool:
    def __init__(self):
        self.packets: List[SocialPacket] = []

    def add_packet(self, packet: SocialPacket) -> None:
        self.packets.append(packet)

    def get_all_packets(self) -> List[SocialPacket]:
        return self.packets

    def get_packets_for_receiver(self, receiver_id: int) -> List[SocialPacket]:
        """Return packets sent by all agents except the receiver."""
        return [pkt for pkt in self.packets if pkt.sender_id != receiver_id]
