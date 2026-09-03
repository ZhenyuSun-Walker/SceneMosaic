# Object Placement and Orientation Task

You are an intelligent spatial planning agent. Your task is to determine the optimal placement for a new object within a given 2D orthogonal scene based on the provided visual and textual information.

### Input Data:
1. **Scene Image**: An orthogonal camera rendering of the current layout with a coordinate grid or reference markers. Blue arrows in the rendered image indicate the facing orientation of each object.
2. **Layout Description**: A textual description of where each object is placed in this layout and explanation of why they are placed there.
3. **Current Layout Data (JSON)**: Object centers, orientations, and bounding boxes in grid coordinates.
4. **Cross Anchor Constraints (optional)**: Additional layout constraints grouped by anchor_id.

### Objective:
Based on the new layout description and existing layout examples, generate a layout that is different from the existing one, identifying the precise 2D center coordinates (u, v) and the clockwise rotation angle (in degrees) around the axis perpendicular to the screen for the object.
Coordinate convention: u increases to the right, v increases upward; rotation directions use 0 for facing down, 90 for facing left, 180 for facing up, and -90 for facing right.

### Placement Constraint:
All objects must be realistically supported by the anchor's surface. Ensure accurate bounding box calculations to prevent objects from overlapping or extending beyond the anchor's physical edges.
The rotation angle represents the object's facing direction, so it must follow common sense and the scene context: for example, chairs should face toward the center of the table they are placed around, and objects near the edge of a floor should face inward toward the interior of the floor.
If cross anchor constraints are provided, the generated pose should strictly satisfy them.

### Output Requirements:
Return a JSON object with the following structure:
```json
{
  "poses": [
    {
      "object_id": "string",
      "reasoning": "Placed based on text description. Coordinates and rotation are calculated using bounding boxes to fit within the anchor's physical edges and avoid overlapping.",
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
- The rotation angle is absolute. Set this value to its original angle if no orientation change is required.
- Do not change the anchor object.
- Use precise numeric values for `u`, `v`, and `rotation_degrees`.
- When generating layouts, explicitly compute the boundaries of each object to avoid collisions.