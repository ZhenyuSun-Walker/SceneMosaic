# Multi-Agent Relation Delivery Policy

This is the static production policy for relation extraction. It defines output and evidence constraints without prescribing a fixed DAG or fixed task names.

## Delivery scope

- Read scene inputs from the relation-ready package generated from SAM3D outputs.
- Write final `objects.json` and `relations.json` only inside the requested run output directory.
- Preserve intermediate evidence, rejected candidates, repairs, and timing under `debug/`.
- Do not use historical relation outputs as inference inputs.

## Objects and walls

- Final `objects.json` contains exactly four walls and the complete SAM3D object id set.
- Walls use ids `wall_001` through `wall_004` and OpenGL top-down `[x, z]` coordinates.
- Formal wall geometry comes from `obb_from_mesh_footprint_min_rect` and is canonicalized to screen-clockwise ids before relation merge.
- Each final object contains base fields `id`, `category`, `description`, boolean `upright`, and a local-validated `physics` object with `mass`, `linear_damping`, `angular_damping`, `static_friction`, and `dynamic_friction`.
- Only objects that anchor at least one final `contain` relation additionally contain `contain_anchor=true` and `render_direction=front|up`; non-anchors omit both fields.
- Upright has one semantic owner: the independent Physics Sidecar returns a relation-agnostic prior together with the five simulation values. Final assembly then applies deterministic relation overrides: wall-only and ceiling attach force `upright=false`; floor/object-supported furniture and displays keep the sidecar prior. Local semantic/image/geometry logic is fallback only when the sidecar prior is unavailable.
- Agents may propose object revision/property patches; they must not replace, merge, delete, or renumber the object list.

## Relations

- Final relation types are `attach` and `contain`.
- Both use `parameters.object_anchor` and `parameters.object_target`.
- Attach anchors may be `floor`, confirmed `ceiling`, a canonical wall id, or an existing object id.
- Contain anchors and targets must both be existing object ids.
- The same anchor-target pair cannot be both attach and contain. Neither type wins by default; conflicting independent Agent verdicts are arbitrated from a local original-image crop in the existing DeliverySchemaGuard call.
- Multiple directional contacts are allowed when physically valid, for example floor attach plus wall attach.
- Each object target may have at most one wall attach relation; final assembly deterministically removes additional wall anchors without removing floor/object support.
- Every object should be covered as an attach or contain target unless the run is explicitly left partial for repair.

## Semantics

- Attach means physical support, contact, mounting, hanging, or fixation. Nearness or facing alone is insufficient.
- Contain means the target is inside an ordinary container or on/within any functional level of multi-level furniture. Every occupied level of a bunk bed, bookcase/shelving unit, multi-tier service cart, or rack, including the top, is contain and the same pair must not use attach. A plain exterior top is attach only for an ordinary single-level anchor.
- Local contain candidates are high-recall review selectors, not semantic evidence and not a closed set; agents must reject false positives and check for directly visible missed pairs.
- After Agent acceptance, local mesh-footprint OBB validation removes only clear contain contradictions. Missing geometry or near-boundary cases remain uncertain and preserve Agent judgement.
- Ceiling anchors require positive evidence.
- Weak, inferred, closure-only, or not-visible walls do not anchor ordinary freestanding objects without strong wall-mounted evidence.

## Physical properties

- Relation Agents may output `is_floor_covering` with confidence and reasons. They must not output `upright`; that field belongs to the independent Physics Sidecar.
- Floor-covering anchors are normalized to floor support by final assembly.
- Objects covered only by wall attach, ceiling-attached objects, and ceiling fixtures are not gravity-upright.
- Explicitly wall-mounted objects with accepted wall attach must not also receive floor attach solely for coverage.

## Evidence priority

- Preserve mask/object grounding unless strong evidence proves it wrong.
- Use the canonical top-down image, its matching render-grid affine, object footprints, original image, and wall evidence together.
- Agent outputs are semantic proposals. Canonical wall-id mapping, schema validation, deduplication, conflict resolution, and coverage accounting are local deterministic responsibilities.

## Agent coordination

- SceneDeliveryManager chooses the task DAG from the capability contract and current scene.
- Skeleton-async planning may begin from the object list and original image before heavy artifacts finish.
- Hydration binds logical artifact keys to generated files after preprocessing.
- Independent tasks in the same DAG layer may run concurrently.
- DynamicTaskAgent performs scoped evidence/reasoning tasks.
- ExecutionSupervisor evaluates task completeness.
- DeliverySchemaGuard checks candidate coverage and schema readiness.
- FinalAssembler creates the deterministic final payloads and audit files.
