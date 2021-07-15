function data = segRead(filename)

inp = load(filename);
f = fields(inp);
data = uint8(inp.(f{1}));