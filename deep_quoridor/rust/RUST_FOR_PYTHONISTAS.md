# Rust for People Who Know Python (and maybe C++/Java)

This guide is tailored to the actual code in `deep_quoridor/rust/`. Every
concept is illustrated with real lines from this repository. After reading it
you should be able to follow roughly 80–90% of the code here without looking
things up constantly.

---

## Table of Contents

1. [The Big Idea: Safety Without a GC](#1-the-big-idea)
2. [Variables and Mutability](#2-variables-and-mutability)
3. [Types](#3-types)
4. [Functions](#4-functions)
5. [Structs](#5-structs)
6. [Enums](#6-enums)
7. [Pattern Matching](#7-pattern-matching)
8. [Option — No More None Crashes](#8-option--no-more-none-crashes)
9. [Result — Errors Without Exceptions](#9-result--errors-without-exceptions)
10. [Ownership and Borrowing](#10-ownership-and-borrowing)
11. [References: `&T` and `&mut T`](#11-references-t-and-mut-t)
12. [Traits — Rust's Interfaces](#12-traits--rusts-interfaces)
13. [Trait Objects: `dyn Trait`](#13-trait-objects-dyn-trait)
14. [Generics](#14-generics)
15. [Closures and Iterators](#15-closures-and-iterators)
16. [Modules and Visibility](#16-modules-and-visibility)
17. [Attributes and Macros](#17-attributes-and-macros)
18. [Conditional Compilation — Features](#18-conditional-compilation--features)
19. [Lifetimes (Brief)](#19-lifetimes-brief)
20. [Common Patterns in This Codebase](#20-common-patterns-in-this-codebase)

---

## 1. The Big Idea

Rust gives you the performance of C++ without a garbage collector, while
statically preventing the two biggest sources of C++ bugs:

- **Use-after-free / dangling pointers** — the compiler rejects these at
  compile time via the *ownership* system.
- **Data races** — the compiler rejects concurrent mutation without
  synchronisation.

Python analogy: imagine Python where the interpreter checks at *import time*
(not at runtime) that you never access a variable after it's been deleted or
that two threads never write to the same list simultaneously. That checking is
free at runtime because it already happened.

The trade-off: you have to learn a few compiler rules up-front (ownership,
borrowing). The compiler errors are famous for being helpful.

---

## 2. Variables and Mutability

```rust
// Immutable by default — like a Python variable you promise not to rebind
let x = 5;

// Mutable — you must say so explicitly
let mut score = 0u32;
score += 1;
```

In Python everything is mutable. In Rust, `let` is immutable by default; you
opt into mutability with `let mut`. This shows up constantly in this codebase:

```rust
// From selfplay.rs
let mut wins = [0u32; 2];   // mutable array
let mut draws = 0u32;
```

If you forget `mut` and try to assign, the compiler tells you exactly which
line to add it to.

---

## 3. Types

### Primitive numeric types

| Rust     | Python / C++ equivalent          |
|----------|----------------------------------|
| `i8`     | signed 8-bit int (-128..127)     |
| `i32`    | `int` / `int32_t`                |
| `i64`    | `int` / `int64_t`                |
| `usize`  | `int` (always ≥ 0, pointer-sized)|
| `f32`    | `float` (32-bit)                 |
| `f64`    | `float` (64-bit, Python default) |
| `bool`   | `bool`                           |

The grid cells in this repo are `i8`, player counts are `i32`, array indices
are `usize`.

```rust
// From grid.rs — plain integer constants
pub const CELL_FREE: i8 = -1;
pub const CELL_WALL: i8 = 10;
```

### String types

`String` is an owned, heap-allocated string (like Python `str`).
`&str` is a borrowed string slice (like a reference into one).

```rust
let s: String = "hello".to_string();
let slice: &str = "world";        // string literal is &str
```

In practice you'll see both: function arguments usually take `&str`, return
values and struct fields usually hold `String`.

### Arrays and slices

```rust
let arr = [0u32; 2];     // fixed-size array of 2 u32s, all zero
let slice: &[bool] = &arr_of_bools;   // reference to a contiguous sequence
```

Arrays `[T; N]` have a compile-time fixed size. Slices `&[T]` are references
to a contiguous sequence of unknown (runtime) length — you'll see them as
function parameters: `action_mask: &[bool]` in `ActionSelector::select_action`.

### Tuples

```rust
let pair: (usize, Vec<f32>) = (3, vec![0.1, 0.9]);
let (idx, probs) = pair;   // destructuring — like Python tuple unpacking
```

Functions that return multiple values use tuples. You'll see this everywhere:
```rust
// From agents/mod.rs
fn select_action(...) -> anyhow::Result<(usize, Vec<f32>)>;
```

### Vec — the growable list

`Vec<T>` is Python's `list`. It lives on the heap and can grow.

```rust
let mut v: Vec<f32> = Vec::new();
v.push(0.5);
let also_a_vec = vec![1.0, 2.0, 3.0];   // vec! macro for literals
```

---

## 4. Functions

```rust
// From onnx_agent.rs
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp_values.iter().sum();
    exp_values.iter().map(|&x| x / sum).collect()
}
```

Key points:
- `pub` means public (visible outside this module). Without it, private.
- Parameter types are mandatory: `logits: &[f32]`.
- Return type after `->`.
- **No `return` needed** — the last expression without a semicolon is the
  return value. Adding a semicolon turns it into a statement (returns nothing).

```rust
fn add(a: i32, b: i32) -> i32 {
    a + b        // returned — no semicolon
}

fn greet(name: &str) {
    println!("Hello, {}", name);   // returns () (unit, like Python None)
}
```

---

## 5. Structs

Rust has no classes with methods baked in. Instead: a `struct` holds data, and
an `impl` block adds methods separately.

```rust
// From replay_writer.rs
pub struct GameMetadata {
    pub model_version: u32,
    pub game_length: usize,
    pub creator: String,
}

// From game_runner.rs
impl GameMetadata {
    fn some_method(&self) -> usize {
        self.game_length
    }
}
```

`&self` is like Python's `self` — it's the receiver.
`&mut self` would be needed if the method modifies the struct.

Constructing a struct:
```rust
let metadata = GameMetadata {
    model_version: 0,
    game_length: result.replay_items.len(),
    creator: format!("{}", pid),
};
```

There is no class hierarchy / inheritance. Code reuse is done via *traits*
(section 12).

---

## 6. Enums

Rust enums are much more powerful than Python or Java enums — each variant can
carry data.

```rust
// The standard library Option type is defined roughly like this:
enum Option<T> {
    Some(T),   // has a value of type T
    None,      // no value
}

// And Result:
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

You can define your own:
```rust
enum Direction {
    North,
    South,
    East,
    West,
}

enum Action {
    Move(i32, i32),          // carries coordinates
    PlaceWall { row: i32, col: i32, orientation: i32 },  // named fields
}
```

Enums are almost always used with `match` (next section).

---

## 7. Pattern Matching

`match` is like a supercharged `switch`/`if-elif` that also *destructures*
values. The compiler checks that all cases are covered.

```rust
// From selfplay.rs
match result.winner {
    Some(0) => wins[0] += 1,
    Some(1) => wins[1] += 1,
    _ => draws += 1,          // _ is a wildcard — matches anything else
}
```

```rust
// Also from selfplay.rs — matching a tuple of two Options
match (&mut onnx_p2, &mut random_p2) {
    (Some(ref mut a), _) => a,   // onnx_p2 is Some → use it
    (_, Some(ref mut a)) => a,   // random_p2 is Some → use it
    _ => unreachable!(),          // logically impossible
}
```

You can match on literal values, enum variants, ranges, tuples, and structs.
Each arm can bind variables:

```rust
match cli.p2.as_deref() {
    Some("random") => { random_p2 = Some(RandomAgent::new()); }
    Some(other)    => { /* `other` is bound to the string value */ }
    None           => { onnx_p2 = Some(OnnxAgent::new(&cli.model_path)?); }
}
```

`if let` is a shorthand when you only care about one variant:
```rust
if let Some(winner) = result.winner {
    println!("Player {} won", winner);
}
```

---

## 8. Option — No More None Crashes

There is no `null`/`None` in Rust (unless you use unsafe code). Whenever a
value might be absent, you use `Option<T>`:

```rust
let maybe: Option<String> = Some("hello".to_string());
let nothing: Option<String> = None;
```

The compiler *forces* you to handle the None case before you can use the inner
value. This eliminates an entire class of runtime errors.

Common ways to use it:

```rust
// unwrap() — crashes (panics) if None. Use only when you're certain.
let val = maybe.unwrap();

// unwrap_or() — provides a fallback
let desc = cli.p2.as_deref().unwrap_or("onnx (same model)");

// if let — handle the Some case
if let Some(v) = maybe {
    println!("{}", v);
}

// match — handle both cases
match maybe {
    Some(v) => println!("{}", v),
    None    => println!("nothing"),
}

// ? operator inside functions that return Option
fn find_something(items: &[i32]) -> Option<i32> {
    let first = items.first()?;   // returns None early if slice is empty
    Some(*first * 2)
}
```

`as_deref()` converts `Option<String>` → `Option<&str>` (a borrowed view of
the string), which is handy for matching on string literals.

---

## 9. Result — Errors Without Exceptions

Rust has no exceptions. Functions that can fail return `Result<T, E>`:

```rust
pub fn new(model_path: &str) -> Result<Self> {  // Self means OnnxAgent
    let session = Session::builder()
        .context("Failed to create ONNX session builder")?
        .commit_from_file(model_path)
        .context("Failed to load ONNX model")?;
    Ok(Self { session })
}
```

`Ok(value)` wraps a success value; `Err(error)` wraps a failure.

### The `?` operator

`?` is the key to ergonomic error handling. On a `Result`:
- If `Ok(v)`, unwraps to `v` and continues.
- If `Err(e)`, **returns early** from the current function with `Err(e)`.

It's equivalent to:
```python
# Python equivalent
try:
    v = might_fail()
except Exception as e:
    return Err(e)
# continue with v
```

You can chain it:
```rust
let config = load_config(&cli.config)?;   // returns early on failure
let q = &config.quoridor;                 // only reached if Ok
```

### `anyhow`

This codebase uses the `anyhow` crate for error handling. `anyhow::Result<T>`
is shorthand for `Result<T, anyhow::Error>`, which can hold any error type.
`.context("message")` adds a human-readable description to an error.

`anyhow::bail!("message")` is a macro that returns an error immediately:
```rust
anyhow::bail!("Unknown --p2 agent type: '{}'", other);
```

---

## 10. Ownership and Borrowing

This is the concept most unique to Rust. It replaces both garbage collection
and manual `free()`.

### The three rules

1. Every value has exactly **one owner**.
2. When the owner goes out of scope, the value is **dropped** (freed).
3. You can have either **one mutable reference** OR **any number of immutable
   references** at a time — never both simultaneously.

### Move semantics

```rust
let a = String::from("hello");
let b = a;          // ownership moves to b; a is no longer valid
// println!("{}", a);  // ERROR: a was moved
println!("{}", b);  // fine
```

This is different from Python (where both `a` and `b` would point to the same
object) and C++ (where this is a copy by default, or a move if you use
`std::move`).

Primitive types like `i32`, `f32`, `bool` implement `Copy`, so they are
duplicated on assignment instead of moved:
```rust
let x: i32 = 5;
let y = x;   // x is still valid — i32 is Copy
```

### Cloning

If you need to duplicate a non-Copy type:
```rust
let a = String::from("hello");
let b = a.clone();   // explicit copy; both a and b are valid
```

### Why this matters for reading the code

Every time you see `&` or `&mut`, it means "borrowing" — temporarily giving
access without transferring ownership (section 11).

---

## 11. References: `&T` and `&mut T`

A reference is a pointer that the compiler guarantees is valid. There are two
kinds:

| Rust        | Meaning                              | Python analogy           |
|-------------|--------------------------------------|--------------------------|
| `&T`        | Shared (immutable) reference         | Reading a variable       |
| `&mut T`    | Exclusive (mutable) reference        | Writing to a variable    |

```rust
// From pathfinding.rs style
pub fn distance_to_row(
    grid: &ArrayView2<i8>,   // shared reference — we only read it
    start_row: i32,
    target_row: i32,
) -> i32 { ... }
```

```rust
// Mutable reference — the function can modify the array
pub fn compute_move_action_mask(
    grid: &ArrayView2<i8>,
    action_mask: &mut ArrayViewMut1<bool>,   // we write into this
) { ... }
```

The compiler enforces: while a `&mut` reference to a value exists, no other
reference (mutable or not) may exist to the same value. This prevents data
races at compile time.

### `ref mut` inside patterns

Inside a `match` arm, `ref mut` means "give me a mutable reference to the
matched value rather than moving it":

```rust
match (&mut onnx_p2, &mut random_p2) {
    (Some(ref mut a), _) => a,   // a: &mut OnnxAgent
    ...
}
```

Without `ref mut`, the match would try to *move* the value out, which isn't
always possible.

---

## 12. Traits — Rust's Interfaces

A `trait` defines a set of methods that types must implement. It's like a Java
interface or Python abstract base class.

```rust
// From agents/mod.rs
pub trait ActionSelector {
    fn select_action(
        &mut self,
        grid: &ArrayView2<i8>,
        player_positions: &ArrayView2<i32>,
        walls_remaining: &ArrayView1<i32>,
        goal_rows: &ArrayView1<i32>,
        current_player: i32,
        action_mask: &[bool],
    ) -> anyhow::Result<(usize, Vec<f32>)>;
}
```

Implementing the trait:
```rust
// OnnxAgent implements ActionSelector
impl ActionSelector for OnnxAgent {
    fn select_action(&mut self, ...) -> anyhow::Result<(usize, Vec<f32>)> {
        // actual neural-network inference
    }
}

// RandomAgent also implements ActionSelector
impl ActionSelector for RandomAgent {
    fn select_action(&mut self, ...) -> anyhow::Result<(usize, Vec<f32>)> {
        // random sampling from valid moves
    }
}
```

Traits can also have *default* method implementations (like Python mixin
classes). The most important built-in traits:

| Trait      | What it does                                     |
|------------|--------------------------------------------------|
| `Clone`    | `obj.clone()` — explicit deep copy               |
| `Copy`     | implicit copy on assignment (for small types)    |
| `Debug`    | `{:?}` formatting (like Python `__repr__`)       |
| `Display`  | `{}` formatting (like Python `__str__`)          |
| `Iterator` | enables `for` loops and `.map()/.filter()` etc.  |
| `Default`  | `Type::default()` — zero/empty value             |

These are commonly derived automatically with `#[derive(...)]` (section 17).

---

## 13. Trait Objects: `dyn Trait`

When you need to store or pass "something that implements a trait" without
knowing the concrete type at compile time, you use a *trait object*: `&dyn
Trait` or `Box<dyn Trait>`.

```rust
// From selfplay.rs — both OnnxAgent and RandomAgent are possible here
let agent_p2: &mut dyn ActionSelector = match (&mut onnx_p2, &mut random_p2) {
    (Some(ref mut a), _) => a,
    (_, Some(ref mut a)) => a,
    _ => unreachable!(),
};
```

`agent_p2` can point to either type at runtime. The method call
`agent_p2.select_action(...)` is resolved at runtime via a vtable — exactly
like virtual functions in C++ or interface calls in Java.

The alternative (when you know the type at compile time) is *static dispatch*
with generics — no runtime overhead:
```rust
fn run<A: ActionSelector>(agent: &mut A, ...) { ... }
```

---

## 14. Generics

Generics let you write code that works for many types. The syntax is angle
brackets.

```rust
// T is a type parameter — like templates in C++ or generics in Java
fn largest<T: PartialOrd>(list: &[T]) -> T { ... }
```

In this codebase, the most common generics come from the `ndarray` crate:

```rust
// ArrayView2<i8>  →  a 2D array of i8 values (read-only view)
// ArrayView1<i32> →  a 1D array of i32 values (read-only view)
// Array4<f32>     →  an owned 4D array of f32 values
pub fn grid_game_state_to_resnet_input(
    grid: &ArrayView2<i8>,
    walls_remaining: &ArrayView1<i32>,
    current_player: i32,
) -> ndarray::Array4<f32> { ... }
```

`ndarray` is the Rust equivalent of NumPy. `Array2<f32>` ≈ `np.ndarray` with
`dtype=float32` and 2 dimensions.

---

## 15. Closures and Iterators

### Closures

Closures are anonymous functions that can capture their environment:

```rust
let threshold = 0.5;
let big: Vec<f32> = probs.iter()
    .filter(|&&p| p > threshold)   // closure captures `threshold`
    .copied()
    .collect();
```

Closure syntax: `|args| body`. For multi-line: `|args| { ... }`.

Compared to Python lambdas, Rust closures:
- Can be multi-line
- Are strongly typed (the compiler infers types)
- Explicitly capture by reference (`&x`) or by value (move)

### Iterators

Rust iterators are lazy (like Python generators) and chain with methods:

```rust
// From onnx_agent.rs
let exp_values: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
let sum: f32 = exp_values.iter().sum();
```

Common iterator methods:

| Method         | Python equivalent          |
|----------------|---------------------------|
| `.map(f)`      | `map(f, iter)`            |
| `.filter(f)`   | `filter(f, iter)`         |
| `.collect()`   | `list(...)`               |
| `.sum()`       | `sum(...)`                |
| `.enumerate()` | `enumerate(...)`          |
| `.zip(other)`  | `zip(iter1, iter2)`       |
| `.flat_map(f)` | `chain(*map(f, iter))`    |
| `.any(f)`      | `any(f(x) for x in iter)` |
| `.all(f)`      | `all(f(x) for x in iter)` |

```rust
// From selfplay.rs
for (i, (&p, &valid)) in probs.iter().zip(action_mask.iter()).enumerate() {
    if valid && p > best_prob {
        best_prob = p;
        best_idx = i;
    }
}
```

### Rayon — parallel iterators

The `rayon` crate adds `.par_iter()` which runs the pipeline in parallel across
threads, with no other code changes needed:

```rust
// From minimax.rs — parallel evaluation
actions.par_iter().map(|action| {
    // each action evaluated on a different thread
    evaluate_one(action)
}).collect()
```

---

## 16. Modules and Visibility

Modules are Rust's namespacing mechanism, roughly equivalent to Python packages.

```rust
// From lib.rs — declaring submodules
pub mod actions;       // public — usable from outside this crate
pub mod game_state;
mod validation;        // private — only usable within this crate
mod pathfinding;
```

Each module lives in a file with the same name (`actions.rs`) or a directory
(`compact/mod.rs` for the `compact` module).

Using items from another module:
```rust
use crate::agents::ActionSelector;          // from within the same crate
use quoridor_rs::game_runner::play_game;    // from the crate's public API
use std::time::Instant;                     // from the standard library
```

`crate` means the root of the current library. `super` means the parent module.

---

## 17. Attributes and Macros

### Attributes

Attributes look like `#[something]` and are metadata applied to the next item.
Common ones in this codebase:

```rust
#[derive(Debug, Clone)]   // auto-implement Debug and Clone traits
struct MyStruct { ... }

#[test]                   // marks a function as a unit test
fn test_softmax() { ... }

#[allow(dead_code)]       // suppress "unused" compiler warning
fn helper() { ... }

#[inline]                 // hint to inline this function at call sites
pub fn hot_path() { ... }

#[cfg(feature = "binary")]  // conditional compilation (section 18)
pub mod game_runner;
```

`#[derive(...)]` is particularly useful — it generates boilerplate
implementations automatically. `#[derive(Debug)]` is like Python's `__repr__`;
`#[derive(Clone)]` adds `.clone()`.

### Macros

Macros look like function calls but end with `!`. They expand at compile time.

| Macro            | Purpose                                   |
|------------------|-------------------------------------------|
| `println!(...)`  | print to stdout (like Python `print`)     |
| `format!(...)`   | build a String (like Python f-strings)    |
| `vec![...]`      | create a Vec literal                      |
| `assert_eq!(a,b)`| test that a == b, panic if not            |
| `unreachable!()` | panic with "should never reach here"      |
| `todo!()`        | placeholder: panics with "not implemented"|
| `params![...]`   | from rusqlite: SQL parameter list         |

Format strings use `{}` for Display and `{:?}` for Debug:

```rust
println!("[{}/{}] P1 wins: {}", game_idx + 1, cli.num_games, wins[0]);
```

`#[derive(Parser)]` from the `clap` crate auto-generates argument parsing from
the struct definition — you get `--config`, `--model-path`, etc. for free.

---

## 18. Conditional Compilation — Features

The `Cargo.toml` defines *features* — optional capabilities:

```toml
[features]
default = ["python"]
python = ["pyo3", "numpy"]
binary = ["clap", "ort", "serde_yaml", "ndarray-npy", "zip"]
```

Code is included or excluded based on active features:

```rust
#[cfg(feature = "python")]
use pyo3::prelude::*;         // only compiled when python feature is on

#[cfg(feature = "binary")]
pub mod game_runner;          // only compiled for the binary targets
```

This is how the same codebase compiles both as a Python extension (`.so` file)
and as a standalone binary (`selfplay`), with different dependencies for each.

When building, you activate features with `--features`:
```bash
cargo build --bin selfplay --features binary --no-default-features
```

---

## 19. Lifetimes (Brief)

Lifetimes are annotations that tell the compiler how long a reference is valid.
They appear as `'name`:

```rust
// From lib.rs — Python FFI
fn get_valid_move_actions<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray2<i8>,
    ...
) -> Bound<'py, PyArray2<i32>> { ... }
```

`'py` here means "the returned array lives as long as the Python GIL is held".
The compiler uses this to ensure you don't return a reference to something that
has already been freed.

In most ordinary Rust code you don't write lifetime annotations — the compiler
infers them. You'll mainly see them at FFI boundaries (like PyO3) or in
advanced generic code. Don't worry about writing them; just recognise the `'`
syntax.

---

## 20. Common Patterns in This Codebase

### ndarray indexing

```rust
// 2D indexing — like numpy arr[i, j]
grid[[i, j]]

// Slicing — like numpy arr[0, 3, :, :]
input.slice_mut(ndarray::s![0, 3, .., ..]).fill(my_walls);

// Shape access
let grid_size = grid.ncols();
assert_eq!(input.shape(), &[1, 5, 13, 13]);
```

### Iterating with index

```rust
for (i, action) in actions.iter().enumerate() { ... }
```

### Collecting results of a transform

```rust
// Build a Vec from an iterator (like Python list comprehension)
let probs: Vec<f32> = logits.iter().map(|&x| x.exp()).collect();
```

### Casting between numeric types

```rust
let n: i32 = 5;
let m: usize = n as usize;   // explicit cast — no implicit coercions
let f: f32 = n as f32;
```

### `.to_string()` and `format!()`

```rust
let s: String = "hello".to_string();
let s2: String = format!("game_{:04}", idx);  // {:04} pads to 4 digits
```

### `Box<dyn Error>` — generic error return

```rust
// From lib.rs — function can return any error type
) -> Result<usize, Box<dyn std::error::Error>> {
```

This is the "I don't care what the error type is" escape hatch, similar to
returning `Exception` in Python.

### `Arc<Mutex<T>>` — shared mutable state across threads

```rust
// From q_minimax.rs
use std::sync::{Arc, Mutex};
let logs: Arc<Mutex<Vec<MinimaxLogEntry>>> = Arc::new(Mutex::new(Vec::new()));

// In a thread:
logs.lock().unwrap().push(entry);
```

`Arc` is a reference-counted pointer safe to share across threads (like
Python's objects, but explicit). `Mutex` adds a lock so only one thread
modifies the inner data at a time. `lock().unwrap()` acquires the lock (panics
if the mutex is poisoned by a previous panic).

### `impl Trait` return types

```rust
fn make_iter() -> impl Iterator<Item = i32> { ... }
```

Means "returns something that implements `Iterator`" without naming the concrete
type. The compiler resolves it at compile time (no runtime overhead).

---

## Quick Reference Cheat Sheet

```
let x = 5;             // immutable binding
let mut x = 5;         // mutable binding
x as f32               // explicit cast
&x                     // shared reference (borrow)
&mut x                 // mutable reference (exclusive borrow)

Option<T>              // Some(v) or None
Result<T, E>           // Ok(v)  or Err(e)
expr?                  // return Err early if Err, else unwrap Ok

match expr { ... }     // exhaustive pattern matching
if let Some(v) = opt { ... }  // one-arm match

struct Foo { x: i32 }         // data struct
impl Foo { fn bar(&self) {} } // methods
trait Bar { fn baz(&self); }  // interface/ABC
impl Bar for Foo { ... }      // implement trait

Vec<T>                 // growable list (Python list)
[T; N]                 // fixed-size array
&[T]                   // slice (reference to contiguous elements)

#[derive(Debug,Clone)] // auto-implement traits
#[test]                // unit test
#[cfg(feature="...")]  // conditional compilation

println!("{}", x)      // print
format!("{:?}", x)     // format to String
vec![1, 2, 3]          // Vec literal
assert_eq!(a, b)       // test equality
```

---

## What to Look Up Next

Once you're comfortable with the above, the things that will help most for this
codebase specifically:

- **ndarray book** — `https://docs.rs/ndarray` — for array shapes, slicing,
  views vs owned arrays.
- **`?` operator** in detail — how it interacts with different error types.
- **Rayon docs** — `par_iter()` and other parallel primitives.
- **PyO3 guide** — only needed if you're modifying the Python FFI layer in
  `lib.rs`.
