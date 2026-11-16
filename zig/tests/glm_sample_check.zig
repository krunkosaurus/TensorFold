//! Check the real GLM sampler and its production bindings against the host reference without a checkpoint.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const dispatch = @import("draw_dispatch");
const draw = dispatch.rules;
const sources = @import("kernel_sources");
const Masks = @import("grammar_mask").Masks;
const sentinel: u32 = 0xDEAD_BEEF;
const rows = 4;

const Case = struct {
    name: []const u8,
    vocab: usize,
    top_k: u32 = 0,
    top_p: f64 = 1,
    min_p: f64 = 0,
    temperature: f64 = 1,
    shape: enum { flat, levels, min_p, shuffled } = .flat,
};

const cases = [_]Case{
    .{ .name = "min_p normalization", .vocab = 3, .top_p = 0.5, .min_p = 0.8, .shape = .min_p },
    .{ .name = "arrival order", .vocab = 2049, .top_p = 0.5, .shape = .levels },
    .{ .name = "large top_k", .vocab = 4096, .top_k = 3000, .top_p = 0.5 },
    .{ .name = "large top_k unfiltered", .vocab = 4096, .top_k = 3000 },
    .{ .name = "large nucleus", .vocab = 2048, .top_p = 0.75 },
    .{ .name = "tiny vocabulary", .vocab = 1 },
    .{ .name = "top_k one", .vocab = 17, .top_k = 1, .shape = .shuffled },
    .{ .name = "small filtered top_k", .vocab = 1031, .top_k = 41, .top_p = 0.7, .temperature = 0.7, .shape = .shuffled },
    .{ .name = "top_k 1024", .vocab = 4097, .top_k = 1024, .top_p = 0.5 },
    .{ .name = "top_k 1025", .vocab = 4097, .top_k = 1025, .top_p = 0.5 },
    .{ .name = "top_k at vocabulary", .vocab = 2048, .top_k = 2048, .top_p = 0.75 },
    .{ .name = "top_k beyond vocabulary", .vocab = 2048, .top_k = 9999, .top_p = 0.75 },
    .{ .name = "shuffled ties", .vocab = 4097, .top_p = 0.9, .shape = .shuffled },
    .{ .name = "large vocabulary", .vocab = 65539, .top_p = 0.9, .temperature = 0.7, .shape = .shuffled },
    .{ .name = "hot distribution", .vocab = 1031, .top_p = 0.9, .temperature = 1.5, .shape = .shuffled },
};

fn value(c: Case, i: usize) f32 {
    return switch (c.shape) {
        .flat => 0,
        .levels => if (i < 1024) 1 else if (i < 2048) 10 else -10,
        .min_p => ([_]f32{ 0, -0.10009765625, -1 })[i],
        .shuffled => -0.25 * @as(f32, @floatFromInt((i * 37 + 11) % 19)),
    };
}

fn run(queue: mtl.Queue, sample_pipeline: mtl.Pipeline, argmax_pipeline: mtl.Pipeline, logits: mtl.Buffer, offset: usize, picks: mtl.Buffer, payload: *const draw.Payload) !void {
    @memset(picks.slice(u32, rows + 2), sentinel);
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    dispatch.encode(enc, sample_pipeline, argmax_pipeline, logits, offset, picks, 4, payload, @intCast(payload.rules[0].vocab));
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |reason| {
        std.debug.print("Metal failure: {s}\n", .{reason});
        return error.GpuFailed;
    }
    const got = picks.slice(u32, rows + 2);
    if (got[0] != sentinel or got[payload.header.n + 1] != sentinel) return error.OutputBounds;
}

/// Structured output: the production mask binding then the production draw, against the host reference over -inf rows.
fn masked(gpa: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, sample_pipeline: mtl.Pipeline, argmax_pipeline: mtl.Pipeline) !void {
    const vocab: usize = 4097; // the last word of a row's bits is partial
    const shape: Case = .{ .name = "grammar masks", .vocab = vocab, .shape = .shuffled };
    var masks = try Masks.init(device, rows, vocab);
    defer masks.deinit();
    const bits = try gpa.alloc(u32, rows * masks.words);
    defer gpa.free(bits);
    @memset(bits, 0);
    const allow = struct {
        fn f(all: []u32, words: usize, r: usize, t: usize) void {
            all[r * words + t / 32] |= @as(u32, 1) << @intCast(t % 32);
        }
    }.f;
    for ([_]usize{ 5, 100, 4096 }) |t| allow(bits, masks.words, 0, t); // a few tokens, the vocabulary's last among them
    @memset(bits[masks.words..][0..masks.words], 0xffff_ffff); // a free row
    for (0..vocab) |t| if (t % 2 == 0) allow(bits, masks.words, 2, t); // half the vocabulary, not the free argmax (11)
    allow(bits, masks.words, 3, 4000); // one token: a forced draw, sampled or not
    try masks.put(0, bits);
    const reference = try gpa.alloc(f32, rows * vocab);
    defer gpa.free(reference);
    for (0..rows) |r| for (0..vocab) |t| {
        const on = (bits[r * masks.words + t / 32] >> @intCast(t % 32)) & 1 == 1;
        reference[r * vocab + t] = if (on) value(shape, t) else -std.math.inf(f32);
    };
    const logits = try device.buffer(rows * vocab * 2, mtl.ResourceOptions.shared);
    defer logits.deinit();
    const picks = try device.buffer((rows + 2) * 4, mtl.ResourceOptions.shared);
    defer picks.deinit();
    const sampling: lanes.Sampling = .{ .seed = 0x8000_0000_0000_0001, .temperature = 0.7, .top_k = 41, .top_p = 0.9 };
    // every mix of sampled and greedy rows: a grammar's row may sit in either kind of dispatch run
    for (0..@as(usize, 1) << rows) |mix| {
        var d: draw.Draws = .{};
        d.n = rows;
        var expected: [rows]u32 = undefined;
        for (0..rows) |r| {
            const position: u32 = 17 + @as(u32, @intCast(r));
            const row = reference[r * vocab ..][0..vocab];
            const sampled = (mix >> @intCast(r)) & 1 == 1;
            expected[r] = if (sampled) try lanes.gpu_full.sample(gpa, row, sampling, position, null) else lanes.gpu_rule.argmax(row);
            d.rules[r] = if (sampled) draw.ruleOf(sampling, position, @intCast(vocab)) else draw.Rule.greedy(@intCast(vocab));
        }
        const words = logits.slice(u16, rows * vocab);
        for (0..rows) |r| for (0..vocab) |t| {
            words[r * vocab + t] = @truncate(@as(u32, @bitCast(value(shape, t))) >> 16);
        };
        const payload = draw.Payload.init(&d, rows, @intCast(vocab));
        @memset(picks.slice(u32, rows + 2), sentinel);
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        dispatch.mask(enc, &masks, logits, 0, vocab, 0, rows);
        dispatch.encode(enc, sample_pipeline, argmax_pipeline, logits, 0, picks, 4, &payload, @intCast(vocab));
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |reason| {
            std.debug.print("Metal failure: {s}\n", .{reason});
            return error.GpuFailed;
        }
        const got = picks.slice(u32, rows + 2);
        if (got[0] != sentinel or got[rows + 1] != sentinel) return error.OutputBounds;
        for (got[1 .. rows + 1], expected, 0..) |g, want, r| {
            const on = (bits[r * masks.words + g / 32] >> @intCast(g % 32)) & 1 == 1;
            if (g != want or !on) {
                std.debug.print("FAIL grammar masks: mix {b:0>4} row {d}, got {d} (allowed: {}), expected {d}\n", .{ mix, r, g, on, want });
                return error.MaskedMismatch;
            }
        }
        // the free row draws what it draws with no mask at all, and the single-token row is forced
        if (got[1 + 3] != 4000) return error.MaskedMismatch;
    }
    // the control: the same greedy rows without the mask pick the free argmax, which rows 0, 2 and 3 disallow
    const free = lanes.gpu_rule.argmax(reference[vocab..][0..vocab]);
    for ([_]usize{ 0, 2, 3 }) |r| if ((bits[r * masks.words + free / 32] >> @intCast(free % 32)) & 1 == 1) return error.MaskedControl;
    std.debug.print("PASS grammar masks: masked sampled and greedy rows in every mix, a free row between them, a forced row\n", .{});
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, sources.nemotron_sample, mtl.CompileOptions.mlx()); // the shared sampler's library
    defer lib.deinit();
    const glue_lib = try mtl.Library.fromSource(device, sources.glm_glue, mtl.CompileOptions.mlx());
    defer glue_lib.deinit();
    const pipeline = try mtl.Pipeline.init(device, lib, dispatch.kernelName(false), false);
    defer pipeline.deinit();
    const argmax = try mtl.Pipeline.init(device, glue_lib, "glm_argmax", false);
    defer argmax.deinit();
    std.debug.print("GLM sampling device: {s}\n", .{device.name()});
    for (cases) |c| {
        const row = try gpa.alloc(f32, c.vocab);
        defer gpa.free(row);
        for (row, 0..) |*v, i| v.* = value(c, i);
        const logits = try device.buffer(rows * c.vocab * 2, mtl.ResourceOptions.shared);
        defer logits.deinit();
        const words = logits.slice(u16, rows * c.vocab);
        for (0..rows) |r| for (row, 0..) |v, i| {
            words[r * c.vocab + i] = @truncate(@as(u32, @bitCast(v)) >> 16);
        };
        const picks = try device.buffer((rows + 2) * 4, mtl.ResourceOptions.shared);
        defer picks.deinit();
        const configs = [_]?lanes.Sampling{
            .{ .seed = 0, .temperature = c.temperature, .top_k = c.top_k, .top_p = c.top_p, .min_p = c.min_p },
            null,
            .{ .seed = 0x8000_0000_0000_0001, .temperature = c.temperature, .top_k = c.top_k, .top_p = c.top_p, .min_p = c.min_p },
            null,
        };
        var expected: [rows]u32 = undefined;
        var d: draw.Draws = .{};
        d.n = rows - 1;
        for (configs, 0..) |cfg, r| {
            const position: u32 = 17 + @as(u32, @intCast(r));
            expected[r] = if (cfg) |s| try lanes.gpu_full.sample(gpa, row, s, position, null) else lanes.gpu_rule.argmax(row);
            d.rules[r] = if (cfg) |s| draw.ruleOf(s, position, @intCast(c.vocab)) else draw.Rule.greedy(@intCast(c.vocab));
        }
        for (0..8) |repeat| {
            const reverse = repeat % 2 == 1;
            const payload = draw.Payload.init(&d, rows, @intCast(c.vocab));
            var ordered = payload;
            if (reverse) std.mem.reverse(draw.Rule, ordered.rules[0..rows]);
            try run(queue, pipeline, argmax, logits, 0, picks, &ordered);
            for (picks.slice(u32, rows + 2)[1 .. rows + 1], 0..) |got, r| {
                const want = expected[if (reverse) rows - 1 - r else r];
                if (got != want) {
                    std.debug.print("FAIL {s}: repeat {d} row {d}, got {d}, expected {d}\n", .{ c.name, repeat, r, got, want });
                    return error.SampleMismatch;
                }
            }
        }
        for (0..rows) |r| {
            var solo: draw.Draws = .{};
            solo.rules[0] = d.rules[r];
            solo.n = 1;
            const payload = draw.Payload.init(&solo, 1, @intCast(c.vocab));
            try run(queue, pipeline, argmax, logits, r * c.vocab * 2, picks, &payload);
            if (picks.slice(u32, rows + 2)[1] != expected[r]) return error.SoloMismatch;
        }
        std.debug.print("PASS {s}: repeated, reordered, mixed, trailing and solo rows\n", .{c.name});
    }
    try masked(gpa, device, queue, pipeline, argmax);
}
