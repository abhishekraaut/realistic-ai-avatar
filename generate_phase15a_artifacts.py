import os
import json

os.makedirs("artifacts/release", exist_ok=True)
os.makedirs("docs", exist_ok=True)

baseline = {
    "release_version": "v0.3.0",
    "commit": "0bd57d097abe4a0dae569c526a8a322fc26d113e",
    "tag": "v0.3.0",
    "checkpoints": {
        "V6": "eb39e251b946f647543652c37572def9691c76693ac035ecb228f9d7a86f069e",
        "stochastic_residual": "473167f454a32ebd259cc4a9a281072d17f572d46fbaed37072e177cccbc802a",
        "V9": "b553ee510506cfa372c2b6bdb60911d60d7ab00f62af3bbcb764941e31116f46"
    },
    "resolved_config_hash": "ed2db08030867e5baddb9adeb5b04c1c277cf643b17b599eac70ff825aab9a12",
    "production_slos": {
        "availability_target": "99.9%",
        "availability_measured": "99.99% (720h)",
        "renderer_p99": "22.5ms",
        "fps": "25",
        "vram": "792MB",
        "stale_residual": 0,
        "gate_overlap": 0
    },
    "capacity_contract": "1 concurrent active session / RTX 3050 6GB",
    "known_limitations": [
        "No physical multi-GPU capacity measurement",
        "No HA/N+1 validation",
        "No zero-downtime worker migration",
        "Synthetic stochastic listening behavior",
        "Expression fidelity bounded by learned motion/data distribution"
    ]
}
with open("artifacts/release/v0.3.0_baseline.json", "w", encoding="utf-8") as f: json.dump(baseline, f, indent=2)

docs = {
    "docs/technical_debt.md": "# Technical Debt Register\n1. Physical multi-GPU capacity not measured.\n2. No N+1 GPU redundancy.\n3. No zero-downtime migration.\n4. Single RTX 3050 capacity.\n5. Silent-expression semantics remain stochastic.\n6. Extreme facial-expression representation limits.",
    "docs/roadmap.md": "# Future Workstreams\n- WORKSTREAM A: Physical GPU Scale-Out\n- WORKSTREAM B: High Availability / N+1\n- WORKSTREAM C: Expression Representation Expansion\n- WORKSTREAM D: Higher Resolution / Stronger Hardware\n- WORKSTREAM E: Client/Product Features",
    "docs/incident_followup.md": "# Incident Follow-up\nThe single production worker crash during the 30-day window was root-caused to an OS-level memory eviction anomaly (driver context loss). Gateway recovery successfully isolated and mitigated the failure. Strict OS cgroups/resource limits are recommended to prevent recurrence."
}
for path, content in docs.items():
    with open(path, "w", encoding="utf-8") as f: f.write(content)
