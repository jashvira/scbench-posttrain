# AGENTS.md

## Docs First
- For any external framework, library, service, CLI, or platform the user asks about or we depend on, check the official docs first before answering, configuring, or patching. Do not guess from memory when the documented surface is available.
- After checking docs, verify against the local installed code or repo state before making claims about what works in this project.

## Inspect Work
- For any Inspect-related feature or refactor, read the official docs first and follow the documented surface instead of inventing local UI or logging patterns: [Inspect Docs](https://inspect.aisi.org.uk/), [Tasks](https://inspect.aisi.org.uk/tasks.html), [Scorers](https://inspect.aisi.org.uk/scorers.html), [Custom Agents / Transcripts](https://inspect.aisi.org.uk/agent-custom.html), [Log Viewer](https://inspect.aisi.org.uk/log-viewer.html).
- For evaluator diagnostics or visualizations, prefer transcript events, score metadata, sample metadata, or store over mutating the sample message history.
