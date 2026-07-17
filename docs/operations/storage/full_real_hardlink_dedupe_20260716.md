# Full-real packet hardlink deduplication, 2026-07-16

Scope: only files named `full_real_packet.pt` and files under `packet_hub/full_real/`.

Pre-check:

- Data volume: 50 GiB, 47 GiB used, 3.8 GiB available (93%).
- Repository outputs: 44 GiB.
- Candidate paths: 102.
- Every link candidate was compared byte-for-byte by `hardlink --dry-run --content`.

Action:

- Replaced 94 duplicate directory entries with hard links to identical content.
- Deleted no experiment directory, metric, manifest, checkpoint, or packet path.
- Reported physical saving: 10.59 GiB.

Post-check:

- Data volume: 50 GiB, 36 GiB used, 15 GiB available (72%).
- Repository outputs: 33 GiB.

Second pass, same-run packet/hub copies:

- Compared each `packet_hub/<method>/agent_*_packet.pt` with the corresponding
  `agents/agent_*/packets/<method>_packet.pt` from the same run.
- Hard-linked 1,149 byte-identical hub copies.
- Additional physical saving: 6,167,753,632 bytes (5,882 MiB).
- No cross-run packet was linked in this pass.
- Final data-volume state: 31 GiB used, 20 GiB available (61%).
- Final repository outputs: 28 GiB.
- Remaining same-run packet/hub duplicates after verification: 0.

The operation preserves all historical paths. These packet artifacts must remain read-only;
future regeneration should write a new file atomically rather than mutate a packet in place.
Core packet save, logit attachment, and packet-hub registration now use atomic replacement.
