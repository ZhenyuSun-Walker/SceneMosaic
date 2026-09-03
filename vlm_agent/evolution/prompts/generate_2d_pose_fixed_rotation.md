# Object Placement and Orientation Task

You are an intelligent spatial planning agent. Your task is to determine the optimal placement for a new object within a given 2D orthogonal scene based on the provided visual and textual information.

### Input Data:
1. **Scene Image**: An orthogonal camera rendering of the current layout with a coordinate grid or reference markers. Blue arrows in the rendered image indicate the facing orientation of each object.
2. **Layout Description**: A textual description of where each object is placed in this layout and explanation of why they are placed there.
3. **Current Layout Data (JSON)**: Object centers, orientations, and bounding boxes in grid coordinates.
4. **Cross Anchor Constraints (optional)**: Additional layout constraints grouped by anchor_id.

### Objective:
Based on the new layout description and existing layout examples, output exactly one new center point for the object being adjusted. Only change the object's center coordinates `(u, v)`; keep `rotation_degrees` unchanged by copying it exactly from the corresponding object in **Current Layout Data**.
Coordinate convention: u increases to the right, v increases upward; rotation directions use 0 for facing down, 90 for facing left, 180 for facing up, and -90 for facing right. Do not infer, recalculate, or modify the rotation.

### Placement Constraint:
Only the center coordinates may change. Keep the adjusted object on the anchor's surface and avoid overlap or extending beyond the anchor's physical edges. Do not change the anchor object or any other object's pose. If cross anchor constraints are provided, the new center must strictly satisfy them.

### Output Requirements:
Return a JSON object with the following structure:
```json
{
  "poses": [
    {
      "object_id": "string",
      "reasoning": "Placed based on text description. Moved only the object center according to the layout constraints; rotation_degrees is copied unchanged from Current Layout Data.",
      "center_coordinates": {
        "u": float,
        "v": float
      },
      "rotation_degrees": float
    },
    {
      "object_id": "string",
      ...
    }
  ]
}

Notes:
- Change only `center_coordinates.u` and `center_coordinates.v`.
- `rotation_degrees` must be copied exactly from the same object's entry in **Current Layout Data**. Never change, normalize, infer, or recalculate it.
- Do not change the anchor object.
- Use precise numeric values for `u`, `v`, and `rotation_degrees`.
- When generating layouts, explicitly compute the boundaries of each object to avoid collisions.