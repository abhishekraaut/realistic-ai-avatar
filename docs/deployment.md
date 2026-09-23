# Deployment (v0.3.0)
Architecture: Gateway -> Worker Registration -> Admission -> Session Routing -> Cleanup
Current capacity contract: RTX 3050 (max_capacity=1). 
Workers must be in READY state to receive admission. UNHEALTHY and DRAINING workers are dynamically isolated.
