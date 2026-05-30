"""
Utilities for selecting a subset of agents from command line arguments.
"""

from typing import List, Optional


def parse_agent_ids(agent_ids: Optional[str], num_agents: int) -> List[int]:
    if agent_ids is None or agent_ids == "all":
        return list(range(num_agents))

    selected = []
    for part in agent_ids.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(part))

    unique_selected = []
    seen = set()
    for agent_id in selected:
        if agent_id < 0 or agent_id >= num_agents:
            raise ValueError(f"agent_id out of range: {agent_id}, num_agents={num_agents}")
        if agent_id not in seen:
            unique_selected.append(agent_id)
            seen.add(agent_id)

    if not unique_selected:
        raise ValueError("No valid agent ids selected.")

    return unique_selected
