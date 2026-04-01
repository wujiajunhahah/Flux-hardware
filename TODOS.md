# Deferred TODOs

## Contract Harness Follow-Ups

- V1B: extend operation coverage beyond the initial six core-chain cases once V1A is stable in CI.
- Add simulator-driven end-to-end automation after the platform contract harness is reliable.
- Add screenshot and UI rendering assertions only after an iOS test target exists.
- Add a WebSocket control server and event sink only when there is a real remote runner consumer.
- Revisit automatic promotion from captured failures to durable contract cases after the manual review flow proves useful.
- Evaluate whether any remaining `smoke_test.py` helpers should be retired once the harness is the primary regression entrypoint.
