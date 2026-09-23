# FINAL ACCEPTANCE & INTEGRITY

1. **Release Tag:** v0.1.0-avatar-rc
2. **Commit SHA:** 169fcbfc780997704d01985f2639ca6ab84f4969
3. **Checkpoint SHA-256:**
   - V6: `4552099401b073c9702acda7b2a926e5532785a47b3ee26f7c43aa484356950f`
   - V8 Renderer: `143261ad47f63bb24e361e93f7049301695834b35b821272db95fd05323246e8`
4. **Dataset Hashes:**
   - Manifest: `8f25002ea72552ae4539b049a15748d6c2721ff2d4fad3a635a57b02576e7464`
   - Split: `ff9d69927e6743984a0a9fc9a5c609737210e72e7d58ceb08b5ded6188dd8fa8`
5. **Exact Environment:** Windows 11 Native, Python 3.12.10, Node v24.21.0, npm 11.19.0.
6. **Dependency Versions:** python_dependencies.txt and node_dependencies.json strictly locked and verified.
7. **Clean-room Result:** Verified. All dependencies loaded cleanly.
8. **Test Result:** 452/452 tests passed cleanly.
9. **Benchmark Result:** 1280x720, 25 FPS, 17.5ms renderer latency.
10. **User-perceived Browser Timing:** T13 = 884ms, T15 = 906ms, T16 = 905ms.
11. **Known Limitations:** 1280x720 maximum bound, latency floor of 884ms to WebRTC receipt, localized yaw artifacts > 35°.
