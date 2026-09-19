//! Backends. Each is a cargo feature; an unbuilt backend is *absent* from
//! [`crate::backend::factories`], which is how "this binary lacks the backend" is
//! detected without a second list to keep in sync.

#[cfg(feature = "backend-llamacpp")]
pub mod llamacpp;
