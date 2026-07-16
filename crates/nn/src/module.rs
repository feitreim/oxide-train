//! The [`Module`] trait and composition combinators.

/// A differentiable computation with statically-typed input/output.
///
/// This is the whole "autograd": no tape, no graph object. Because shapes are
/// const generics, `Input`/`Output`/`Ctx` are concrete types per module
/// instantiation, and a composition that doesn't type-check is a model that
/// doesn't chain-rule.
pub trait Module {
    type Input;
    type Output;
    /// Whatever `backward` needs, saved by `forward`: typically the input
    /// itself and/or intermediate activations. Moved, never cloned.
    type Ctx;

    /// Compute the output and save what backward will need.
    fn forward(&self, x: Self::Input) -> (Self::Output, Self::Ctx);

    /// Consume the saved context and the output-gradient; *accumulate*
    /// parameter gradients into the module's own grad buffers and return the
    /// input-gradient (same type as the input).
    ///
    /// Accumulation (`+=`) rather than assignment is the contract, so shared
    /// parameters and gradient accumulation over micro-batches come for free.
    fn backward(&mut self, ctx: Self::Ctx, dy: Self::Output) -> Self::Input;

    /// Reset parameter gradient buffers. Default: no parameters, no-op.
    fn zero_grad(&mut self) {}
}

/// Sequential composition: `Chain(a, b)` computes `b(a(x))`.
///
/// The `B: Module<Input = A::Output>` bound *is* the chain rule's shape
/// agreement; backward runs `b` then `a`, threading each saved `Ctx` back to
/// its owner.
pub struct Chain<A, B> {
    pub a: A,
    pub b: B,
}

impl<A, B> Chain<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A, B> Module for Chain<A, B>
where
    A: Module,
    B: Module<Input = A::Output>,
{
    type Input = A::Input;
    type Output = B::Output;
    type Ctx = (A::Ctx, B::Ctx);

    fn forward(&self, x: Self::Input) -> (Self::Output, Self::Ctx) {
        let (y, ctx_a) = self.a.forward(x);
        let (z, ctx_b) = self.b.forward(y);
        (z, (ctx_a, ctx_b))
    }

    fn backward(&mut self, (ctx_a, ctx_b): Self::Ctx, dz: Self::Output) -> Self::Input {
        let dy = self.b.backward(ctx_b, dz);
        self.a.backward(ctx_a, dy)
    }

    fn zero_grad(&mut self) {
        self.a.zero_grad();
        self.b.zero_grad();
    }
}

/// `chain!(a, b, c, ...)` — right-nested [`Chain`] without the parens pyramid.
#[macro_export]
macro_rules! chain {
    ($a:expr, $b:expr $(,)?) => {
        $crate::Chain::new($a, $b)
    };
    ($a:expr, $b:expr, $($rest:expr),+ $(,)?) => {
        $crate::Chain::new($a, $crate::chain!($b, $($rest),+))
    };
}
