# Specification: [Short Feature, Refactor, or Bug Name]

## Metadata

* **Status:** [Draft | Ready for Dev | In Progress | Done]
* **Target Release:** [e.g., v0.2]
* **Primary Developer/Agent:** [Name]
* **Priority Reference:** [e.g., "PLAN.md Priority 3: Code Deduplication"]

## Objective

[What specific problem are we solving? Provide context on why this matters for the retinanalysis codebase.]

## User Story (if applicable)

"As a [lab member / developer / new user], I want to [action] so that [result/benefit]."

## Acceptance Criteria (Definition of Done)

*Must be strictly binary (Pass/Fail) and testable.*

* **AC1:** [Observable behavior that must be true]
* **AC2:** [Edge case or regression that must stay fixed]
* **AC3:** [e.g., "Importing `retinanalysis` with Docker stopped must not raise an exception"]

## Scope & Affected Files

* **Files Modified:** [Explicitly list files, e.g., `src/retinanalysis/config/schema.py`, `src/retinanalysis/utils/vision_utils.py`]
* **Files Created:** [e.g., `src/retinanalysis/utils/cell_typing.py`]
* **Data Contracts:** [Expected inputs/outputs if changing function signatures]

## Technical Approach

[Describe the implementation strategy. Include:]
* What will be changed and why
* Any new abstractions or patterns introduced
* Migration path for existing code (if breaking changes)

## Backward Compatibility

* **API Changes:** [List any function/class signature changes]
* **Import Changes:** [Will `import retinanalysis as ra; ra.xyz` still work?]
* **Database Changes:** [Any schema modifications?]

## Test Plan

* **Unit:** [Pure logic tests, e.g., `tests/unit/test_ei_corr.py`]
* **Integration:** [Tests requiring fixtures or mock DB, e.g., `tests/integration/test_pipeline_creation.py`]
* **Manual Verification:** [Steps to verify in a Jupyter notebook if applicable]

## Out Of Scope

* [Things this spec deliberately does NOT change to prevent scope creep.]

## Notes & Open Questions

* [Any unresolved design decisions or questions for the team.]
