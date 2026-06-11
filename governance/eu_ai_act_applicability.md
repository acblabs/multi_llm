# EU AI Act Applicability Reasoning

The MVP is framed as administrative prior-authorization decision support with mandatory human review. Under that framing, the system does not autonomously determine access to care.

Risk increases materially if the system:

- directly approves or denies access to care;
- materially influences payer decisions without meaningful human oversight;
- is used as a clinical decision system rather than administrative support;
- processes sensitive health data without appropriate governance and transparency.

The architecture therefore enforces:

- high-risk classification for prior-authorization workflows;
- human-in-the-loop escalation;
- auditability of policy and privacy decisions;
- explicit system-card limitations.

This file is reasoning support, not legal advice.
