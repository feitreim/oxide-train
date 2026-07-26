//! Shared-memory tiles with the swizzle in the type.
//!
//! The layout scheme is the one flash-attn and gemm validated on B200: a
//! SWIZZLE_128B tile is stored as stacked 128-byte-row *subtiles* (64 bf16
//! columns each), so the swizzle phase inside each subtile equals the row
//! index — the coincidence a 64-wide panel gives for free, kept by
//! construction at every width. A `[R, 128]` bf16 operand is two stacked
//! `[R, 64]` subtiles a subtile-stride apart; `[R, 64]` operands (P/dS) are a
//! single subtile. Widths that are not a whole number of subtiles are a
//! compile error, not a differently-swizzled layout (issue #61, risk 3:
//! restrict honestly rather than pretend generality).
//!
//! Two facts of that layout the types keep straight:
//! - **Absolute phase.** SWIZZLE_128B XORs *physical* address bits [9:7]
//!   into the 16-byte-chunk index, so manual swizzled stores must fold in
//!   the tile base's own 128-byte row phase ([`SharedTile::swizzle_phase`]);
//!   `swizzle_probe` in flash-attn verifies the pattern on hardware.
//! - **TMA loads a subtile per box.** The tensor maps built by [`crate::global`]
//!   describe 64-column boxes, so [`SharedTile::tma_load`] issues one
//!   `cp.async.bulk.tensor` per subtile, lifting the leading coordinate by 64
//!   per stack level ([`load_panel` in flash-attn's tcgen05.rs, generalized]).

use core::marker::PhantomData;

use cuda_device::tma::{
    TmaDescriptor, cp_async_bulk_tensor_2d_g2s, cp_async_bulk_tensor_2d_g2s_multicast_cg2,
    cp_async_bulk_tensor_3d_g2s,
};

use crate::sync::Semaphore;

/// Element marker for tile types. Only carries the byte width — device code
/// moves tile data as packed words through intrinsics, never as `T` values.
pub trait Element {
    const BYTES: usize;
}

/// bf16 — the only staged-operand element the tcgen05 kernels use.
pub struct Bf16;
impl Element for Bf16 {
    const BYTES: usize = 2;
}

/// Swizzle mode marker. Only `SWIZZLE_128B` is implemented: it is the only
/// mode the validated kernels use, and the subtile scheme depends on its
/// 128-byte atom.
pub trait Swizzle {
    /// Bytes of one swizzle atom — the physical row width of a subtile.
    const ATOM_BYTES: usize;
    /// The mode's encoding in a tcgen05 shared-memory operand descriptor.
    const DESCRIPTOR_MODE: u8;
}

/// 128-byte swizzle: 16-byte chunks XOR physical address bits [9:7].
pub struct Swizzle128B;
impl Swizzle for Swizzle128B {
    const ATOM_BYTES: usize = 128;
    const DESCRIPTOR_MODE: u8 = 2;
}

/// A `[R, C]` shared-memory tile of `E` elements under swizzle `S`, stored
/// as `C / (ATOM_BYTES / E::BYTES)` stacked `[R, subtile]` panels. The handle
/// is a base pointer plus compile-time shape — Copy, register-resident, and
/// free once inlined.
pub struct SharedTile<E: Element, const R: usize, const C: usize, S: Swizzle> {
    base: *mut u8,
    _marker: PhantomData<(E, S)>,
}

impl<E: Element, const R: usize, const C: usize, S: Swizzle> Clone for SharedTile<E, R, C, S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Element, const R: usize, const C: usize, S: Swizzle> Copy for SharedTile<E, R, C, S> {}

impl<E: Element, const R: usize, const C: usize, S: Swizzle> SharedTile<E, R, C, S> {
    /// Columns of one subtile (64 for bf16 under SWIZZLE_128B).
    pub const SUBTILE_COLS: usize = S::ATOM_BYTES / E::BYTES;
    /// Stacked subtiles in this tile.
    pub const SUBTILES: usize = C / Self::SUBTILE_COLS;
    /// Bytes of one `[R, SUBTILE_COLS]` subtile.
    pub const SUBTILE_BYTES: usize = R * S::ATOM_BYTES;
    /// Bytes of the whole tile — what a TMA load of it must charge via
    /// [`Semaphore::expect_tx`].
    pub const BYTES: usize = R * C * E::BYTES;

    const WIDTH_OK: () = assert!(
        C.is_multiple_of(S::ATOM_BYTES / E::BYTES),
        "tile width must be a whole number of swizzle subtiles"
    );

    /// Wrap a raw shared-memory base (a `DynamicSharedArray` offset or a
    /// `SharedArray` static).
    ///
    /// # Safety
    ///
    /// `base` must point to at least `Self::BYTES` bytes of shared memory,
    /// 128-byte aligned, living as long as every use of the tile.
    #[inline(always)]
    pub const unsafe fn from_raw(base: *mut u8) -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::WIDTH_OK;
        Self {
            base,
            _marker: PhantomData,
        }
    }

    /// The tile's base address.
    #[inline(always)]
    pub const fn base(self) -> *mut u8 {
        self.base
    }

    /// Base address of stacked subtile `i`.
    ///
    /// # Safety
    ///
    /// `i < Self::SUBTILES`.
    #[inline(always)]
    pub unsafe fn subtile(self, i: usize) -> *mut u8 {
        unsafe { self.base.add(i * Self::SUBTILE_BYTES) }
    }

    /// TMA the tile from a [`crate::global`] panel map: one box per subtile,
    /// the leading (column) coordinate lifted by `SUBTILE_COLS` per stack
    /// level, `row` selecting the global row range and `plane` the panel.
    /// Completion lands on `sem`; the caller charges [`Self::BYTES`] (once
    /// per tile, however many boxes) via [`Semaphore::expect_tx`].
    ///
    /// # Safety
    ///
    /// `map` must describe a live global buffer whose box shape matches
    /// `[R, SUBTILE_COLS]`, and `sem` must be an initialized TMA barrier.
    #[inline(always)]
    pub unsafe fn tma_load(self, map: *const TmaDescriptor, row: i32, plane: i32, sem: Semaphore) {
        unsafe { self.tma_load_at(map, 0, row, plane, sem) }
    }

    /// [`Self::tma_load`] landing at subtile row `dst_row` instead of the top —
    /// how a tile taller than the map's box gets built out of several global
    /// row ranges (the backward kernels stack two adjacent 64-row tiles into
    /// one 128-row operand, `dst_row = 0` then `dst_row = 64`).
    ///
    /// The stacking is layout-free precisely because a box height is a whole
    /// number of swizzle periods (8 rows of 128 bytes): landing a box at row
    /// 64 of a subtile reproduces exactly the swizzle rows 64.. of one tall
    /// tile would have had. The caller charges the bytes actually in flight —
    /// `box_rows * C * E::BYTES` per call, not [`Self::BYTES`].
    ///
    /// # Safety
    ///
    /// As [`Self::tma_load`], plus `dst_row + box_rows <= R` and `dst_row` a
    /// multiple of the 8-row swizzle period.
    #[inline(always)]
    pub unsafe fn tma_load_at(
        self,
        map: *const TmaDescriptor,
        dst_row: usize,
        row: i32,
        plane: i32,
        sem: Semaphore,
    ) {
        unsafe {
            let mut i = 0usize;
            while i < Self::SUBTILES {
                cp_async_bulk_tensor_3d_g2s(
                    self.subtile(i).add(dst_row * S::ATOM_BYTES),
                    map,
                    (i * Self::SUBTILE_COLS) as i32,
                    row,
                    plane,
                    sem.raw(),
                );
                i += 1;
            }
        }
    }

    /// TMA the tile from a 2d tensor map: one box per subtile, the leading
    /// coordinate lifted by `SUBTILE_COLS` per stack level. A K-major operand
    /// (`[R, K]`, one subtile per swizzle atom of K) is a single box at
    /// `(k, row)`; an MN-major operand (`[K, MN]`) is one box per 64-wide MN
    /// subtile at `(mn + 64 * i, k)` — the coordinate order the map's fast
    /// axis dictates, which is why both coordinates are the caller's.
    ///
    /// # Safety
    ///
    /// `map` must describe a live global buffer whose box shape matches
    /// `[R, SUBTILE_COLS]`, and `sem` must be an initialized TMA barrier.
    #[inline(always)]
    pub unsafe fn tma_load_2d(
        self,
        map: *const TmaDescriptor,
        leading: i32,
        minor: i32,
        sem: Semaphore,
    ) {
        unsafe {
            let mut i = 0usize;
            while i < Self::SUBTILES {
                cp_async_bulk_tensor_2d_g2s(
                    self.subtile(i),
                    map,
                    leading + (i * Self::SUBTILE_COLS) as i32,
                    minor,
                    sem.raw(),
                );
                i += 1;
            }
        }
    }

    /// [`Self::tma_load_2d`] as a cta_group::2 multicast: every box lands in
    /// the CTAs of `cta_mask`, completing on each CTA's own copy of the
    /// barrier behind `sem` — pass [`Semaphore::multicast_alias`], not the
    /// local handle.
    ///
    /// # Safety
    ///
    /// As [`Self::tma_load_2d`], and the block must run as a cluster whose
    /// every masked CTA has an initialized barrier at the aliased address.
    #[inline(always)]
    pub unsafe fn tma_load_2d_multicast_cg2(
        self,
        map: *const TmaDescriptor,
        leading: i32,
        minor: i32,
        sem: Semaphore,
        cta_mask: u16,
    ) {
        unsafe {
            let mut i = 0usize;
            while i < Self::SUBTILES {
                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                    self.subtile(i),
                    map,
                    leading + (i * Self::SUBTILE_COLS) as i32,
                    minor,
                    sem.raw(),
                    cta_mask,
                );
                i += 1;
            }
        }
    }

    /// The tile base's absolute position in the 8-row swizzle period.
    /// SWIZZLE_128B XORs *physical* address bits [9:7] into the chunk index,
    /// so a tile whose base is not 1024-byte aligned starts mid-period and
    /// every manual swizzled store must fold this phase in.
    #[inline(always)]
    pub fn swizzle_phase(self) -> usize {
        (self.base as usize >> 7) & 7
    }

    /// Address of 16-byte chunk `chunk` in subtile row `row`, with the
    /// swizzle applied exactly as the TMA engine would have — the store-side
    /// twin of a swizzled TMA load, valid for single-subtile tiles (P/dS)
    /// where the eight chunks index the whole row. For store loops, hoist
    /// the phase once with [`Self::chunk_writer`].
    ///
    /// # Safety
    ///
    /// `row < R`, `chunk < 8`, and the tile must be a single subtile wide.
    #[inline(always)]
    pub unsafe fn swizzled_chunk(self, row: usize, chunk: usize) -> *mut u8 {
        unsafe { self.chunk_writer().at(row, chunk) }
    }

    /// The tile's swizzled-store handle with the base's absolute phase
    /// captured once — hoist it outside a fragment-store loop exactly like
    /// the hand-written kernels hoisted their `p_phase` variables.
    #[inline(always)]
    pub fn chunk_writer(self) -> SwizzledChunks {
        const {
            assert!(
                C * E::BYTES == S::ATOM_BYTES,
                "swizzled chunk stores need a one-subtile tile"
            )
        };
        SwizzledChunks {
            base: self.base,
            phase: self.swizzle_phase(),
        }
    }

    /// tcgen05 shared-memory operand descriptor for the K-major operand at
    /// `byte_offset` into the tile — same encoding as gemm's: 16-byte leading
    /// offset (the second core matrix sits eight bf16 columns along the row),
    /// 1024-byte stride, the swizzle mode in bits [63:61]. Pure bit math on
    /// the address; the MMA that consumes it carries the safety obligations.
    #[inline(always)]
    pub fn operand_descriptor(self, byte_offset: usize) -> u64 {
        self.descriptor(byte_offset, 16)
    }

    #[inline(always)]
    fn descriptor(self, byte_offset: usize, leading_bytes: u32) -> u64 {
        encode_descriptor(
            self.base as u64 + byte_offset as u64,
            leading_bytes,
            S::DESCRIPTOR_MODE,
        )
    }

    /// This tile as a K-major [`OperandWalk`]: one K=16 chunk every 32 bytes
    /// along the swizzled rows, 16-byte leading offset. Restricted to tiles
    /// whose K spans exactly one swizzle atom per row (gemm's `[128, 64]`
    /// stage) — a linear step cannot cross stacked subtiles.
    #[inline(always)]
    pub fn k_walk(self) -> OperandWalk {
        const {
            assert!(
                C * E::BYTES == S::ATOM_BYTES,
                "a linear K-major walk needs K to span exactly one swizzle atom"
            )
        };
        OperandWalk {
            base: self.base,
            chunk_step: 16 * E::BYTES,
            leading_bytes: 16,
            mode: S::DESCRIPTOR_MODE,
        }
    }

    /// This tile as an MN-major [`OperandWalk`] (the instruction carries the
    /// transpose bits): one K=16 chunk every 16 rows, and the leading offset
    /// jumps to the stacked subtile holding MN columns 64..128
    /// ([`Self::SUBTILE_BYTES`] — not a step along the row).
    #[inline(always)]
    pub fn mn_walk(self) -> OperandWalk {
        const { assert!(R.is_multiple_of(16), "MN-major chunks are 16 rows each") };
        OperandWalk {
            base: self.base,
            chunk_step: 16 * S::ATOM_BYTES,
            leading_bytes: Self::SUBTILE_BYTES as u32,
            mode: S::DESCRIPTOR_MODE,
        }
    }
}

#[inline(always)]
fn encode_descriptor(address: u64, leading_bytes: u32, mode: u8) -> u64 {
    const STRIDE_BYTES: u32 = 1024;
    ((address >> 4) & 0x3fff)
        | ((((leading_bytes >> 4) & 0x3fff) as u64) << 16)
        | ((((STRIDE_BYTES >> 4) & 0x3fff) as u64) << 32)
        | (1u64 << 46)
        | ((mode as u64) << 61)
}

/// One MMA operand's chunk walk with the layout erased to values: the chunk
/// step and descriptor leading offset that distinguish a K-major from an
/// MN-major operand, as runtime data instead of a type. This exists for
/// kernels that select the layout at runtime (gemm's `transposed` launch
/// parameter): a value-level walk keeps the issue loop *single* — one
/// select feeding one chain — matching the hand-written kernel's schedule,
/// where a typed two-arm branch would duplicate the MMA chain per layout
/// and hand ptxas a different instruction stream to allocate against.
#[derive(Clone, Copy)]
pub struct OperandWalk {
    base: *mut u8,
    chunk_step: usize,
    leading_bytes: u32,
    mode: u8,
}

impl OperandWalk {
    /// Operand descriptor for K chunk `chunk`.
    #[inline(always)]
    pub fn chunk_descriptor(self, chunk: usize) -> u64 {
        encode_descriptor(
            self.base as u64 + (chunk * self.chunk_step) as u64,
            self.leading_bytes,
            self.mode,
        )
    }
}

/// A single-subtile tile's swizzled-store cursor: 128-byte rows, eight
/// 16-byte chunks per row, chunk index XORed with `(row + phase) & 7` where
/// `phase` is the tile base's absolute position in the swizzle period
/// (captured once at construction).
#[derive(Clone, Copy)]
pub struct SwizzledChunks {
    base: *mut u8,
    phase: usize,
}

impl SwizzledChunks {
    /// Address of chunk `chunk` in row `row`.
    ///
    /// # Safety
    ///
    /// `row` inside the tile, `chunk < 8`.
    #[inline(always)]
    pub unsafe fn at(self, row: usize, chunk: usize) -> *mut u8 {
        unsafe {
            self.base
                .add(row * 128 + (chunk ^ ((row + self.phase) & 7)) * 16)
        }
    }
}

/// `N` same-shaped tiles backing a pipeline ring: tile `i` lives in stage
/// `i % N`. The parity arithmetic for the matching barriers lives in
/// [`crate::sync::SemaphoreRing`].
pub struct SharedTileRing<E: Element, const R: usize, const C: usize, S: Swizzle, const N: usize> {
    base: *mut u8,
    _marker: PhantomData<(E, S)>,
}

impl<E: Element, const R: usize, const C: usize, S: Swizzle, const N: usize> Clone
    for SharedTileRing<E, R, C, S, N>
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Element, const R: usize, const C: usize, S: Swizzle, const N: usize> Copy
    for SharedTileRing<E, R, C, S, N>
{
}

impl<E: Element, const R: usize, const C: usize, S: Swizzle, const N: usize>
    SharedTileRing<E, R, C, S, N>
{
    /// Bytes of the whole ring.
    pub const BYTES: usize = N * SharedTile::<E, R, C, S>::BYTES;

    /// Wrap `N` consecutive tiles' worth of shared memory.
    ///
    /// # Safety
    ///
    /// Same contract as [`SharedTile::from_raw`], for `Self::BYTES` bytes.
    #[inline(always)]
    pub const unsafe fn attach(base: *mut u8) -> Self {
        Self {
            base,
            _marker: PhantomData,
        }
    }

    /// The tile of stage `index % N`.
    #[inline(always)]
    pub fn tile(self, index: u32) -> SharedTile<E, R, C, S> {
        unsafe {
            SharedTile::from_raw(
                self.base
                    .add((index as usize % N) * SharedTile::<E, R, C, S>::BYTES),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Panel = SharedTile<Bf16, 64, 128, Swizzle128B>;
    type PTile = SharedTile<Bf16, 64, 64, Swizzle128B>;
    type Paired = SharedTile<Bf16, 128, 128, Swizzle128B>;

    #[test]
    fn shape_math_matches_the_flash_layout() {
        // The constants tcgen05.rs derives by hand: TILE_BYTES / SUBTILE_BYTES.
        assert_eq!(Panel::SUBTILES, 2);
        assert_eq!(Panel::SUBTILE_BYTES, 64 * 128);
        assert_eq!(Panel::BYTES, 64 * 128 * 2);
        assert_eq!(PTile::SUBTILES, 1);
        assert_eq!(PTile::BYTES, 64 * 64 * 2);
        // The paired backward operand: [128, 64] subtiles a TILE_BYTES apart.
        assert_eq!(Paired::SUBTILE_BYTES, 128 * 128);
    }

    #[test]
    fn swizzled_chunk_folds_the_absolute_base_phase() {
        // Base at an odd 128-byte row phase, like a P subtile mid-plan:
        // phase = (base >> 7) & 7. Pointer math only — never dereferenced.
        let base = 0x1080usize;
        let tile = unsafe { PTile::from_raw(base as *mut u8) };
        assert_eq!(tile.swizzle_phase(), (base >> 7) & 7);
        // tcgen05.rs formula: base + row*128 + (chunk ^ ((row + phase) & 7))*16.
        let phase = tile.swizzle_phase();
        for row in [0usize, 2, 7, 63] {
            for chunk in 0usize..8 {
                let expected = base + row * 128 + ((chunk ^ ((row + phase) & 7)) * 16);
                assert_eq!(
                    unsafe { tile.swizzled_chunk(row, chunk) } as usize,
                    expected
                );
            }
        }
    }

    #[test]
    fn operand_descriptor_encodes_like_gemm() {
        // Same encoding smem_descriptor() produced: address bits, 16-byte
        // leading offset, 1024-byte stride, mode 2 in bits [63:61].
        let base = 0x4000usize;
        let tile = unsafe { PTile::from_raw(base as *mut u8) };
        let descriptor = tile.operand_descriptor(32);
        let address = ((base as u64 + 32) >> 4) & 0x3fff;
        let expected = address | (1u64 << 16) | (64u64 << 32) | (1u64 << 46) | (2u64 << 61);
        assert_eq!(descriptor, expected);
    }

    #[test]
    fn walks_reproduce_gemm_consume_stage_descriptors() {
        // gemm's consume_stage built build_smem_descriptor(smem + offset,
        // leading, 1024, 2) with (offset, leading) = (chunk * 32, 16) for
        // K-major and (chunk * 16 * 128, SUBTILE_BYTES = 8192) for MN-major.
        type KStage = SharedTile<Bf16, 128, 64, Swizzle128B>;
        type MnStage = SharedTile<Bf16, 64, 128, Swizzle128B>;
        assert_eq!(MnStage::SUBTILE_BYTES, 8192);
        let base = 0x4000u64;
        let expected = |offset: u64, leading: u64| {
            (((base + offset) >> 4) & 0x3fff)
                | (((leading >> 4) & 0x3fff) << 16)
                | (64u64 << 32)
                | (1u64 << 46)
                | (2u64 << 61)
        };
        let k = unsafe { KStage::from_raw(base as *mut u8) }.k_walk();
        let mn = unsafe { MnStage::from_raw(base as *mut u8) }.mn_walk();
        for chunk in 0..4usize {
            assert_eq!(k.chunk_descriptor(chunk), expected(chunk as u64 * 32, 16));
            assert_eq!(
                mn.chunk_descriptor(chunk),
                expected(chunk as u64 * 2048, 8192)
            );
        }
    }

    #[test]
    fn ring_stages_wrap() {
        let base = 0x2000usize as *mut u8;
        let ring = unsafe { SharedTileRing::<Bf16, 64, 128, Swizzle128B, 3>::attach(base) };
        assert_eq!(ring.tile(0).base(), base);
        assert_eq!(ring.tile(4).base() as usize, 0x2000 + Panel::BYTES);
        assert_eq!(ring.tile(5).base() as usize, 0x2000 + 2 * Panel::BYTES);
        assert_eq!(ring.tile(6).base(), base);
    }
}
