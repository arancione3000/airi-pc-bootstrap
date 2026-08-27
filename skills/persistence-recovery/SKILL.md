# persistence-recovery

## Description
Safe checkpoint, remote persistence and recovery workflow.

## Instructions
Every mutation declares scope, creates a checkpoint, verifies tests and guardrails, records the decision, persists to the canonical Git repository, and verifies remote HEAD. Never call a failed local commit persistently complete.

## Tools
- computer_recovery_checkpoint
- computer_recovery_read
- computer_persistence_status
- computer_persist
- computer_decision_record
