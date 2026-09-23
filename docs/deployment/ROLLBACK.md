# Rollback Procedure
If deployment fails:
1. Stop all services.
2. `git checkout v0.1.0-avatar-rc`
3. Verify `pip install -r artifacts/release/python_dependencies.txt` matches exactly.
4. Restart services.
