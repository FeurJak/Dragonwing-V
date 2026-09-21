//! `dragonwing-edge` — top-level façade for the workspace.
//!
//! Downstream crates (and external projects like `cortex-edge-rs`) should
//! depend on **this crate alone**, not on the internal `dragonwing-core` or
//! `dragonwing-hal` crates directly. That way internal refactors don't
//! leak into your `Cargo.toml`.
//!
//! # What you get
//!
//! * [`Backend`] / [`BackendBuffer`] / [`BufferKind`] — the backend trait.
//! * [`HardwareCapabilities`] — produced by [`hal::probe_all`].
//! * [`Error`] / [`Result`] — the workspace error type.
//! * Re-exports of [`core`] and [`hal`] for explicit access when needed.
//!
//! # Integration recipe
//!
//! 1. Add `dragonwing-edge = { git = "https://github.com/FeurJak/Dragonwing-V" }`
//!    to your `Cargo.toml`. Pick a tag once the project releases them.
//! 2. Run [`hal::probe_all`] once at process startup to discover what the
//!    device can do.
//! 3. Based on the capability snapshot, instantiate the right backend
//!    crate's `Backend` impl (`dragonwing-vulkan` or `dragonwing-cpu`),
//!    pass it as `Arc<dyn Backend<Buffer = ...>>` or via a small enum to
//!    your inference pipeline.
//! 4. Call op functions from the backend crate (e.g.
//!    `dragonwing_vulkan::ops::axpy_f32`). Op signatures are identical
//!    across backends; parity is enforced by the `dragonwing-test` crate.
//!
//! See `docs/integration-guide.md` in the repository for a full,
//! copy-pasteable example.

pub use dragonwing_core as core;
pub use dragonwing_hal as hal;

pub use dragonwing_core::{Backend, BackendBuffer, BufferKind, Error, HardwareCapabilities, Result};
