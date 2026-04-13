# Implementation Plans

When entering plan mode or working on a feature that may already have an implementation plan, always check the `plans/` directory at the repo root for relevant `.md` files before creating a new plan.

```
/home/ryan/SkyRL/plans/
```

These persistent plans contain detailed, audited implementation specifications with code snippets, correctness proofs, and architectural decisions. They survive across conversations and should be treated as the source of truth for ongoing implementation work.

## Workflow

1. **Before planning**: Read any matching plan in `plans/` to avoid re-deriving decisions already made.
2. **During implementation**: Follow the plan. If you discover the plan is wrong, fix the plan file first, then implement.
3. **After implementation**: Update the plan to mark completed sections and capture any deviations.
4. **New features**: If no plan exists, create one in `plans/` with a descriptive filename (e.g., `plans/feature-name.md`).
