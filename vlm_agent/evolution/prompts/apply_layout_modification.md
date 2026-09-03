# Layout Modification Task

You are an intelligent spatial planning agent. Your task is to apply modification guidance to update object poses based on the rendered image, layout description, current layout data, and modification notes.

### Input Data:
1. **Rendered Scene Image**: An orthogonal camera rendering of the current layout with a coordinate grid. Blue arrows in the rendered image indicate the facing orientation of each object.
2. **Current Layout Data (JSON)**: Object centers, orientations, and bounding boxes in grid coordinates.
3. **Modification Guidance**: Qualitative suggestions for how to adjust objects.

### Objective:
Generate updated 2D center coordinates and rotation angles for objects that need adjustment. Use symbolic expressions and raw numeric values flexibly, referencing existing 2D pose values (center, size_2d, bbox_2d) from CURRENT_LAYOUT_DATA when appropriate. Keep objects within the anchor surface. Keep the anchor object unchanged.

### Output Requirements:
Return a JSON object with reasoning and symbolic pose updates. Only include objects that should change.
Example with comments (actual output should be valid JSON without comments):
```jsonc
{
  "reasoning": "Short explanation tied to the modification guidance.",
  "pose_expressions": [
    "object_001.center.u = object_000.center.u", // Align object_001 with object_000 along the u axis
    "object_002.center.u = object_002.center.u + 0.1", // Move object_002 slightly to the right
    "object_003.center.u = 0.2" // Set object_003's u directly to 0.2
    "object_004.rotation_degrees = 90" // Orient object_004 to the left
  ]
}
```

Notes:
- Coordinate convention: u increases to the right, v increases upward; rotation directions use 0 for facing down, 90 for facing left, 180 for facing up, and -90 for facing right.
- Do not change the anchor object.
- Only modify objects explicitly mentioned in the modification guidance; keep all other objects unchanged.
- Use expressions or raw numeric values as appropriate.
- Allowed references: `object_id.center.u`, `object_id.center.v`, `object_id.size.u`, `object_id.size.v`,
  `object_id.bbox.min_u`, `object_id.bbox.min_v`, `object_id.bbox.max_u`, `object_id.bbox.max_v`,
  `object_id.rotation_degrees`. Aliases: `center_2d`, `size_2d`, `bbox_2d`.
- Please explicitly compute object boundary extents to avoid overlap when adjusting poses.