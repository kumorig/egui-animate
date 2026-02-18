//! Custom animations and transitions for egui.
//!
//! # Core animation API
//! - [`animate`] — UI-mutation transitions keyed on a state value (opacity, slide, etc.)
//! - [`animate_value`] — smooth scalar interpolation with custom easing
//! - [`decay_on_change`] — 1.0→0.0 decay whenever a key changes (highlight fade-outs)
//!
//! # Easing functions
//! [`ease_out_cubic`], [`ease_in_out_cubic`], [`ease_out_quart`], [`ease_out_expo`]

use egui::{Id, Ui};
use std::any::Any;

// ═══════════════════════════════════════════════════════════════
// Easing functions
// ═══════════════════════════════════════════════════════════════

/// Smooth deceleration — `1 - (1-t)³`. Great for UI transitions.
pub fn ease_out_cubic(t: f32) -> f32 {
    1.0 - (1.0 - t.clamp(0.0, 1.0)).powi(3)
}

/// Symmetric — slow start, fast middle, slow end. Best for crossfades / sidebars.
pub fn ease_in_out_cubic(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    if t < 0.5 { 4.0 * t * t * t } else { 1.0 - (-2.0 * t + 2.0_f32).powi(3) / 2.0 }
}

/// Stronger deceleration — `1 - (1-t)⁴`. Good for highlights / glows.
pub fn ease_out_quart(t: f32) -> f32 {
    1.0 - (1.0 - t.clamp(0.0, 1.0)).powi(4)
}

/// Very sharp start, silky finish — `1 - 2^(-10t)`. Ideal for XP / progress bars.
pub fn ease_out_expo(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    if t == 0.0 { 0.0 } else { 1.0 - (2.0_f32).powf(-10.0 * t) }
}

// ═══════════════════════════════════════════════════════════════
// animate() — UI-mutation state transitions
// ═══════════════════════════════════════════════════════════════

/// A single phase of an animation.
/// `duration` seconds; `f` called each frame with progress 0.0 → 1.0.
#[derive(Clone, Copy)]
pub struct AnimationSegment {
    pub duration: f32,
    pub f: fn(&mut Ui, f32),
}

impl AnimationSegment {
    pub const fn new(duration: f32, f: fn(&mut Ui, f32)) -> Self {
        Self { duration, f }
    }
}

/// An animation composed of an *out* segment (plays while leaving old state)
/// and an *in* segment (plays while entering new state).
#[derive(Clone, Copy)]
pub struct Animation {
    pub out_seg: AnimationSegment,
    pub in_seg: AnimationSegment,
}

fn noop(_ui: &mut Ui, _t: f32) {}

impl Animation {
    pub const EMPTY: Self = Self {
        out_seg: AnimationSegment { duration: 0.0, f: noop },
        in_seg: AnimationSegment { duration: 0.0, f: noop },
    };

    /// Full animation with separate out/in functions, same duration.
    pub const fn new(duration: f32, out_fn: fn(&mut Ui, f32), in_fn: fn(&mut Ui, f32)) -> Self {
        Self {
            out_seg: AnimationSegment { duration, f: out_fn },
            in_seg: AnimationSegment { duration, f: in_fn },
        }
    }

    /// Out-only (in is instant).
    pub const fn new_out(duration: f32, out_fn: fn(&mut Ui, f32)) -> Self {
        Self {
            out_seg: AnimationSegment { duration, f: out_fn },
            in_seg: AnimationSegment { duration: 0.0, f: noop },
        }
    }

    /// In-only (out is instant).
    pub const fn new_in(duration: f32, in_fn: fn(&mut Ui, f32)) -> Self {
        Self {
            out_seg: AnimationSegment { duration: 0.0, f: noop },
            in_seg: AnimationSegment { duration, f: in_fn },
        }
    }

    /// Build from explicit segments (supports asymmetric durations).
    pub const fn from_segments(out_seg: AnimationSegment, in_seg: AnimationSegment) -> Self {
        Self { out_seg, in_seg }
    }

    pub fn duration(&self) -> f32 {
        self.out_seg.duration + self.in_seg.duration
    }
}

impl Default for Animation {
    fn default() -> Self { Self::EMPTY }
}

// ─── Run state ───────────────────────────────────────────────────────────────

/// Current animation phase and normalised progress (0.0 – 1.0).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RunState {
    Out(f32),
    In(f32),
    Idle,
}

impl RunState {
    pub fn is_animating(&self) -> bool {
        !matches!(self, Self::Idle)
    }
}

// ─── Internal memory helpers ─────────────────────────────────────────────────

trait CloneAnySendSync: Any + Send + Sync {
    fn clone_box(&self) -> Box<dyn CloneAnySendSync>;
    fn as_any(&self) -> &dyn Any;
}

impl<T: Clone + Any + Send + Sync> CloneAnySendSync for T {
    fn clone_box(&self) -> Box<dyn CloneAnySendSync> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }
}

impl Clone for Box<dyn CloneAnySendSync> {
    fn clone(&self) -> Self { self.clone_box() }
}

#[derive(Clone)]
struct AnimMem {
    current: Box<dyn CloneAnySendSync>,
    phase: RunState,
}

// ─── Public animate() ────────────────────────────────────────────────────────

/// Drive an animation keyed on `value`.
///
/// When `value` changes: out segment plays (mutates `ui` for old state),
/// then in segment plays (mutates `ui` for new state), then `add_contents`
/// is called with the final value.
pub fn animate<T, R>(
    ui: &mut Ui,
    id: impl Into<Id>,
    value: T,
    animation: Animation,
    add_contents: impl FnOnce(&mut Ui, T) -> R,
) -> R
where
    T: 'static + Any + Clone + Send + Sync + Default + PartialEq,
{
    let id: Id = id.into();
    let dt = ui.ctx().input(|i| i.stable_dt);

    let mut mem: AnimMem = ui.ctx().memory_mut(|m| m.data.get_temp(id))
        .unwrap_or_else(|| AnimMem { current: Box::new(value.clone()), phase: RunState::Idle });

    let value_changed = mem.current.as_any().downcast_ref::<T>().map(|v| v != &value).unwrap_or(true);
    if value_changed {
        mem.phase = if animation.out_seg.duration > 0.0 { RunState::Out(0.0) } else { RunState::In(0.0) };
        mem.current = Box::new(value.clone());
    }

    mem.phase = match mem.phase {
        RunState::Out(t) => {
            let next = t + dt / animation.out_seg.duration.max(1e-6);
            if next >= 1.0 { RunState::In(0.0) } else { RunState::Out(next) }
        }
        RunState::In(t) => {
            let next = t + dt / animation.in_seg.duration.max(1e-6);
            if next >= 1.0 { RunState::Idle } else { RunState::In(next) }
        }
        RunState::Idle => RunState::Idle,
    };

    if mem.phase.is_animating() {
        ui.ctx().request_repaint();
    }

    match mem.phase {
        RunState::Out(t) => (animation.out_seg.f)(ui, smoothstep(t)),
        RunState::In(t)  => (animation.in_seg.f)(ui, smoothstep(t)),
        RunState::Idle   => {}
    }

    ui.ctx().memory_mut(|m| m.data.insert_temp(id, mem));
    add_contents(ui, value)
}

/// Query animation run state without advancing or rendering.
pub fn run_state<T>(ui: &Ui, id: impl Into<Id>, _value: &T) -> RunState
where
    T: 'static + Any + Clone + Send + Sync,
{
    ui.ctx()
        .memory(|m| m.data.get_temp::<AnimMem>(id.into()))
        .map(|m| m.phase)
        .unwrap_or(RunState::Idle)
}

// ═══════════════════════════════════════════════════════════════
// animate_value() — smooth scalar with custom easing
// ═══════════════════════════════════════════════════════════════

#[derive(Clone, Default)]
struct ValueMem {
    from: f32,
    to: f32,
    /// Progress 0.0 → 1.0.
    t: f32,
}

/// Animate a `f32` towards `target` using a custom easing function.
///
/// - `duration` — seconds to complete the transition (0.0 snaps instantly)
/// - `easing`   — e.g. [`ease_out_cubic`], [`ease_out_expo`]
///
/// Consecutive calls with the same `id` chain correctly: re-targeting mid-animation
/// starts from the current animated position. Two calls in the same frame with
/// `duration = 0.0` followed by a real duration implement teleport-then-animate.
pub fn animate_value(
    ui: &mut Ui,
    id: impl Into<Id>,
    target: f32,
    duration: f32,
    easing: fn(f32) -> f32,
) -> f32 {
    let id: Id = id.into();
    let dt = ui.ctx().input(|i| i.stable_dt);

    let mut mem: ValueMem = ui.ctx().memory_mut(|m| m.data.get_temp(id)).unwrap_or_default();

    if (mem.to - target).abs() > 1e-6 {
        // Re-target: snap from to the current animated position
        let cur = mem.from + (mem.to - mem.from) * easing(mem.t.clamp(0.0, 1.0));
        mem.from = cur;
        mem.to = target;
        mem.t = 0.0;
    }

    if mem.t < 1.0 {
        let advance = if duration > 1e-6 { dt / duration } else { 1.0 };
        mem.t = (mem.t + advance).min(1.0);
        if mem.t < 1.0 {
            ui.ctx().request_repaint();
        }
    }

    let result = mem.from + (mem.to - mem.from) * easing(mem.t.clamp(0.0, 1.0));
    ui.ctx().memory_mut(|m| m.data.insert_temp(id, mem));
    result
}

// ═══════════════════════════════════════════════════════════════
// decay_on_change() — highlight fade-out on key change
// ═══════════════════════════════════════════════════════════════

#[derive(Clone, Default)]
struct DecayMem {
    state: u64,
    elapsed: f32,
}

/// Returns a value that jumps to **1.0** when `state` changes, then decays to **0.0**
/// over `duration` seconds. State `0` is treated as "no event" and returns 0.0 immediately.
///
/// Typical use — row highlight fade driven by a stable timestamp key:
/// ```ignore
/// let key = game.last_update
///     .filter(|&ts| ts >= *QUIET_END)
///     .map(|ts| ts.duration_since(*APP_START).as_nanos() as u64)
///     .unwrap_or(0);
/// let glow = egui_animate::decay_on_change(ui, row_id, key, SECS, egui_animate::ease_out_quart);
/// ```
pub fn decay_on_change(
    ui: &mut Ui,
    id: impl Into<Id>,
    state: u64,
    duration: f32,
    easing: fn(f32) -> f32,
) -> f32 {
    let id: Id = id.into();
    let dt = ui.ctx().input(|i| i.stable_dt);

    let mut mem: DecayMem = ui.ctx().memory_mut(|m| m.data.get_temp(id)).unwrap_or_default();

    if mem.state != state {
        mem.state = state;
        mem.elapsed = 0.0;
    }

    let result = if state == 0 || mem.elapsed >= duration {
        0.0
    } else {
        mem.elapsed += dt;
        let t = (mem.elapsed / duration).clamp(0.0, 1.0);
        let v = (1.0 - easing(t)).max(0.0);
        if v > 0.001 {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(16));
        }
        v
    };

    ui.ctx().memory_mut(|m| m.data.insert_temp(id, mem));
    result
}

// ═══════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════

/// Smoothstep — internal easing for the animate() state-machine.
fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
