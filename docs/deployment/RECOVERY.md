# Disaster Recovery
1. **Model Loss**: Re-download checkpoints matching `final_checkpoint_manifest.json` SHA256.
2. **Secret Compromise**: Rotate immediately. Inject via environment variables (Docker/K8s). Do not commit `.env`.
3. **Database**: No persistent database is used. System is completely stateless.
