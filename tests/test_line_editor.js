const assert = require("node:assert/strict");
const { directionForPoint } = require("../web/static/js/line_editor.js");

assert.equal(directionForPoint([0, 50], [100, 50], [50, 70]), "forward");
assert.equal(directionForPoint([0, 50], [100, 50], [50, 30]), "reverse");
assert.equal(directionForPoint([50, 0], [50, 100], [30, 50]), "forward");
assert.equal(directionForPoint([50, 0], [50, 100], [70, 50]), "reverse");
assert.equal(directionForPoint([0, 50], [100, 50], [50, 53], "reverse"), "reverse");

console.log("line editor direction checks passed");
