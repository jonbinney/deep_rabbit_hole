/// Minimax for building a policy database.
///
/// Uses i8 values (1 = P0 wins, 0 = tie, -1 = P1 wins)
/// with a transposition table.
///
/// Storage: Parquet file with two columns (`state`, `value`), sorted by
/// `state` bytes (lex order). The footer holds key/value metadata
/// (board_size, max_walls, max_steps, num_states). Row-group min/max
/// statistics on `state` are used to prune lookups; we never load the
/// entire DB into memory.
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, FixedSizeBinaryArray, Int8Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use dashmap::DashMap;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder};
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use parquet::file::statistics::Statistics;
use parquet::format::KeyValue;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use super::q_bit_repr::CompactState;
use super::q_game_mechanics::QGameMechanics;

/// Width of the on-disk / in-memory byte representation of a state.
pub const STATE_BYTES: usize = 24;

/// Transposition table type alias.
pub type TranspositionTable = DashMap<CompactState, i8>;

/// Number of rows per Parquet row group. Larger means fewer groups (less
/// per-group overhead) but larger decode units. 1M is a good general default.
const ROW_GROUP_SIZE: usize = 1_000_000;

/// Per-batch chunk size when streaming rows into the writer. Each chunk
/// becomes one Arrow `RecordBatch`; the writer buffers them up to a row
/// group and then flushes.
const WRITE_CHUNK_SIZE: usize = 65_536;

/// Per-row-group statistics cached at open time, used to prune lookups.
/// State values are stored as `FixedSizeBinary(24)`; comparisons use
/// lexicographic byte order, matching the Parquet sort/statistics ordering.
#[derive(Clone, Debug)]
struct RowGroupStats {
    idx: usize,
    min: [u8; STATE_BYTES],
    max: [u8; STATE_BYTES],
    num_rows: u64,
}

/// Decoded contents of one row group, kept around when we want to reuse
/// it across nearby lookups in the same call.
struct DecodedRowGroup {
    states: Vec<CompactState>,
    values: Vec<i8>,
}

/// Backing storage for an open PolicyDb. The lazy variant defers row-group
/// decode until each lookup; the eager variant loads everything into RAM
/// up front for O(1) state lookups and O(1) rowid access.
enum Storage {
    Lazy {
        path: String,
        metadata: ArrowReaderMetadata,
        row_groups: Vec<RowGroupStats>,
        /// `cum_rows[i]` = sum of `num_rows` over row groups `0..i`.
        /// Length `row_groups.len() + 1`. `cum_rows.last()` = total rows.
        cum_rows: Vec<u64>,
    },
    Eager {
        /// File order (= sorted by state bytes lex). Indexable by `rowid - 1`.
        ordered: Vec<(CompactState, i8)>,
        /// O(1) point lookups for `lookup_values_by_state`.
        table: HashMap<CompactState, i8>,
    },
}

/// Parquet-backed policy database for storing and querying pre-computed
/// minimax values. Single file per DB; sorted by `state` bytes for efficient
/// row-group pruning on point lookups.
pub struct PolicyDb {
    storage: Storage,
    mechanics: QGameMechanics,
    board_size: usize,
    max_walls: usize,
    max_steps: usize,
    num_states: usize,
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(
            "state",
            DataType::FixedSizeBinary(STATE_BYTES as i32),
            false,
        ),
        Field::new("value", DataType::Int8, false),
    ]))
}

fn parse_meta(kv: Option<&Vec<KeyValue>>, key: &str) -> Option<usize> {
    kv?.iter()
        .find(|e| e.key == key)
        .and_then(|e| e.value.as_ref())
        .and_then(|v| v.parse().ok())
}

fn bytes_to_state(b: &[u8]) -> Result<CompactState, Box<dyn std::error::Error>> {
    if b.len() != STATE_BYTES {
        return Err(format!(
            "state column row width is {}, expected {STATE_BYTES}",
            b.len()
        )
        .into());
    }
    let mut a = [0u8; STATE_BYTES];
    a.copy_from_slice(b);
    Ok(CompactState::from_bytes(a))
}

fn slice_to_arr(b: &[u8]) -> Result<[u8; STATE_BYTES], Box<dyn std::error::Error>> {
    if b.len() != STATE_BYTES {
        return Err(format!("stat byte width is {}, expected {STATE_BYTES}", b.len()).into());
    }
    let mut a = [0u8; STATE_BYTES];
    a.copy_from_slice(b);
    Ok(a)
}

impl PolicyDb {
    /// Open an existing policy database for reading.
    ///
    /// When `lazy` is `false` (the default for callers), the entire dataset
    /// is decoded into a `HashMap<CompactState, i8>` at open time so subsequent
    /// state lookups are O(1). When `lazy` is `true`, lookups walk Parquet
    /// row groups on demand using the file's min/max statistics; useful for
    /// DBs too large to fit in memory.
    pub fn open(path: &str, lazy: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(Path::new(path))?;
        let metadata = ArrowReaderMetadata::load(&file, Default::default())?;

        let kv = metadata.metadata().file_metadata().key_value_metadata();
        let board_size =
            parse_meta(kv, "board_size").ok_or("missing board_size in parquet metadata")?;
        let max_walls =
            parse_meta(kv, "max_walls").ok_or("missing max_walls in parquet metadata")?;
        let max_steps =
            parse_meta(kv, "max_steps").ok_or("missing max_steps in parquet metadata")?;
        let num_states_meta = parse_meta(kv, "num_states");

        let mechanics = QGameMechanics::new(board_size, max_walls, max_steps);

        let pq_meta = metadata.metadata();
        let num_row_groups = pq_meta.num_row_groups();
        let mut row_groups = Vec::with_capacity(num_row_groups);
        let mut cum_rows = Vec::with_capacity(num_row_groups + 1);
        cum_rows.push(0u64);
        let mut total = 0u64;
        for i in 0..num_row_groups {
            let rg = pq_meta.row_group(i);
            let num_rows = rg.num_rows() as u64;
            let stats = rg
                .column(0)
                .statistics()
                .ok_or_else(|| format!("row group {i} missing statistics on state column"))?;
            let (min, max) = match stats {
                Statistics::FixedLenByteArray(s) => (
                    slice_to_arr(s.min_opt().ok_or("missing min stat on state")?.as_ref())?,
                    slice_to_arr(s.max_opt().ok_or("missing max stat on state")?.as_ref())?,
                ),
                _ => {
                    return Err(
                        format!("unexpected stats variant for state column: {stats:?}").into(),
                    );
                }
            };
            row_groups.push(RowGroupStats {
                idx: i,
                min,
                max,
                num_rows,
            });
            total += num_rows;
            cum_rows.push(total);
        }

        // Prefer the cached count from metadata, fall back to summed row counts.
        let num_states = num_states_meta.unwrap_or(total as usize);

        let storage = if lazy {
            Storage::Lazy {
                path: path.to_string(),
                metadata,
                row_groups,
                cum_rows,
            }
        } else {
            // Eager load: walk all row groups in one pass, decoding each
            // RecordBatch into the running ordered Vec, then build the
            // O(1) lookup HashMap from it.
            let file = File::open(Path::new(path))?;
            let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata.clone())
                .build()?;

            let mut state_buf: Vec<CompactState> = Vec::with_capacity(num_states);
            let mut value_buf: Vec<i8> = Vec::with_capacity(num_states);
            for batch in reader {
                let batch = batch?;
                append_batch(&batch, &mut state_buf, &mut value_buf)?;
            }

            let mut ordered: Vec<(CompactState, i8)> = Vec::with_capacity(state_buf.len());
            for (s, v) in state_buf.into_iter().zip(value_buf.into_iter()) {
                ordered.push((s, v));
            }
            let mut table = HashMap::with_capacity(ordered.len());
            for &(s, v) in &ordered {
                table.insert(s, v);
            }
            Storage::Eager { ordered, table }
        };

        Ok(Self {
            storage,
            mechanics,
            board_size,
            max_walls,
            max_steps,
            num_states,
        })
    }

    /// Get a reference to the game mechanics.
    pub fn mechanics(&self) -> &QGameMechanics {
        &self.mechanics
    }

    /// Read metadata: `(board_size, max_walls, max_steps, num_states)`.
    pub fn read_metadata(
        &self,
    ) -> Result<(usize, usize, usize, Option<usize>), Box<dyn std::error::Error>> {
        Ok((
            self.board_size,
            self.max_walls,
            self.max_steps,
            Some(self.num_states),
        ))
    }

    /// Count the total number of states.
    pub fn count_states(&self) -> Result<usize, Box<dyn std::error::Error>> {
        Ok(self.num_states)
    }

    /// Decode a single row group into `(states, values)` columns.
    /// Reopens the file so this is safe to call from a `&self` context
    /// without interior mutability or sync. Lazy mode only.
    fn decode_row_group(
        path: &str,
        metadata: &ArrowReaderMetadata,
        rg: &RowGroupStats,
    ) -> Result<DecodedRowGroup, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata.clone())
            .with_row_groups(vec![rg.idx])
            .with_batch_size(rg.num_rows as usize)
            .build()?;

        let mut states = Vec::with_capacity(rg.num_rows as usize);
        let mut values = Vec::with_capacity(rg.num_rows as usize);
        while let Some(batch) = reader.next() {
            let batch = batch?;
            append_batch(&batch, &mut states, &mut values)?;
        }
        Ok(DecodedRowGroup { states, values })
    }

    /// Look up values for all actions reachable from the given state.
    ///
    /// Returns `None` if there are no valid actions or no DB entries were found.
    /// Values are returned from the acting player's perspective.
    pub fn lookup_action_values(
        &self,
        data: CompactState,
    ) -> Result<Option<(Vec<(u8, u8, u8)>, Vec<i32>)>, Box<dyn std::error::Error>> {
        let mechanics = &self.mechanics;
        let cp = mechanics.repr().get_current_player(data);

        let mut data_mut = data;
        let moves = mechanics.get_valid_moves(data_mut);
        let walls = mechanics.get_valid_wall_placements(&mut data_mut);

        let mut actions: Vec<(u8, u8, u8)> = moves
            .into_iter()
            .map(|(r, c)| (r as u8, c as u8, 2))
            .collect();
        actions.extend(
            walls
                .into_iter()
                .map(|(r, c, t)| (r as u8, c as u8, t as u8)),
        );

        if actions.is_empty() {
            return Ok(None);
        }

        // Compute child states and classify each as terminal-or-DB-lookup.
        // P0-perspective values: 1 = P0 wins, 0 = tie, -1 = P1 wins.
        let mut child_states = Vec::with_capacity(actions.len());
        let mut terminal_p0: Vec<Option<i32>> = Vec::with_capacity(actions.len());

        for &(row, col, action_type) in &actions {
            let mut child_data = data;
            let (r, c, t) = (row as usize, col as usize, action_type as usize);
            if action_type == 2 {
                mechanics.execute_move(&mut child_data, cp, r, c);
            } else {
                mechanics.execute_wall_placement(&mut child_data, cp, r, c, t);
            }
            mechanics.switch_player(&mut child_data);

            let child_cp = mechanics.repr().get_current_player(child_data);
            let child_opp = 1 - child_cp;

            let term = if mechanics.check_win(child_data, child_opp) {
                Some(if child_opp == 0 { 1 } else { -1 })
            } else if mechanics.repr().get_completed_steps(child_data)
                >= mechanics.repr().max_steps()
            {
                Some(0)
            } else {
                None
            };

            child_states.push(child_data);
            terminal_p0.push(term);
        }

        // Batch-look up all non-terminal children in one sorted sweep.
        let need_lookup: Vec<CompactState> = child_states
            .iter()
            .zip(terminal_p0.iter())
            .filter(|(_, t)| t.is_none())
            .map(|(s, _)| *s)
            .collect();

        let lookup_pairs = self.lookup_values_by_state(&need_lookup)?;
        let lookup_map: HashMap<CompactState, i32> = lookup_pairs.into_iter().collect();

        let mut values = Vec::with_capacity(actions.len());
        let mut any_found = false;
        for (child, term) in child_states.iter().zip(terminal_p0.iter()) {
            let value_p0: i32 = match term {
                Some(v) => {
                    any_found = true;
                    *v
                }
                None => match lookup_map.get(child) {
                    Some(v) => {
                        any_found = true;
                        *v
                    }
                    None => panic!("No DB entry found for child state"),
                },
            };
            // Convert P0-perspective to acting-player perspective.
            values.push(if cp == 0 { value_p0 } else { -value_p0 });
        }

        if !any_found {
            return Ok(None);
        }

        Ok(Some((actions, values)))
    }

    /// Fetch states and values by 1-based rowid (matches old SQLite ROWID
    /// semantics: row N is the Nth entry in sorted-state order).
    ///
    /// Returns one `(state, value)` per requested rowid, in ascending rowid
    /// order. Rowids out of range are silently dropped.
    pub fn fetch_states_by_rowid(
        &self,
        rowids: &[i64],
    ) -> Result<Vec<(CompactState, i32)>, Box<dyn std::error::Error>> {
        if rowids.is_empty() {
            return Ok(Vec::new());
        }

        match &self.storage {
            Storage::Eager { ordered, .. } => {
                let total_rows = ordered.len() as u64;
                let mut sorted: Vec<u64> = rowids
                    .iter()
                    .filter_map(|&r| {
                        if r >= 1 && (r as u64) <= total_rows {
                            Some((r - 1) as u64)
                        } else {
                            None
                        }
                    })
                    .collect();
                sorted.sort_unstable();

                let mut results = Vec::with_capacity(sorted.len());
                for r0 in sorted {
                    let (s, v) = ordered[r0 as usize];
                    results.push((s, v as i32));
                }
                Ok(results)
            }
            Storage::Lazy {
                path,
                metadata,
                row_groups,
                cum_rows,
            } => {
                let total_rows = *cum_rows.last().unwrap_or(&0);
                let mut sorted: Vec<u64> = rowids
                    .iter()
                    .filter_map(|&r| {
                        if r >= 1 && (r as u64) <= total_rows {
                            Some((r - 1) as u64)
                        } else {
                            None
                        }
                    })
                    .collect();
                sorted.sort_unstable();

                let mut results = Vec::with_capacity(sorted.len());
                let mut current_rg: Option<usize> = None;
                let mut decoded: Option<DecodedRowGroup> = None;

                for r0 in sorted {
                    // cum_rows is sorted ascending; find the row group whose
                    // range [cum_rows[i], cum_rows[i+1]) contains r0.
                    let rg_idx = match cum_rows.binary_search(&r0) {
                        Ok(i) => i,      // r0 is the first row of group i
                        Err(i) => i - 1, // r0 falls in group i-1
                    };
                    let row_in_group = (r0 - cum_rows[rg_idx]) as usize;

                    if current_rg != Some(rg_idx) {
                        decoded =
                            Some(Self::decode_row_group(path, metadata, &row_groups[rg_idx])?);
                        current_rg = Some(rg_idx);
                    }
                    let dec = decoded.as_ref().unwrap();
                    let s = dec.states[row_in_group];
                    let v = dec.values[row_in_group] as i32;
                    results.push((s, v));
                }

                Ok(results)
            }
        }
    }

    /// Look up values for the given states.
    ///
    /// Returns `(state, value)` for each state found in the DB. States
    /// not present are omitted. In eager mode this is an O(N) HashMap
    /// scan; in lazy mode it walks Parquet row groups in sorted order,
    /// decoding each at most once.
    pub fn lookup_values_by_state(
        &self,
        states: &[CompactState],
    ) -> Result<Vec<(CompactState, i32)>, Box<dyn std::error::Error>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }

        match &self.storage {
            Storage::Eager { table, .. } => Ok(states
                .iter()
                .filter_map(|s| table.get(s).map(|v| (*s, *v as i32)))
                .collect()),
            Storage::Lazy {
                path,
                metadata,
                row_groups,
                ..
            } => {
                let mut sorted: Vec<[u8; STATE_BYTES]> =
                    states.iter().map(|s| s.to_bytes()).collect();
                sorted.sort_unstable();
                sorted.dedup();

                let mut results = Vec::new();
                let mut q_idx = 0usize;
                let mut rg_idx = 0usize;

                while q_idx < sorted.len() && rg_idx < row_groups.len() {
                    let q = sorted[q_idx];
                    let rg = &row_groups[rg_idx];

                    if q < rg.min {
                        // Sorted file: q can't appear in any later group either.
                        q_idx += 1;
                    } else if q > rg.max {
                        rg_idx += 1;
                    } else {
                        // q is in [min, max]; decode this row group once and
                        // resolve every query whose value falls in the range.
                        let decoded = Self::decode_row_group(path, metadata, rg)?;
                        // Build a sorted bytes view of the decoded states for binary search.
                        let decoded_bytes: Vec<[u8; STATE_BYTES]> =
                            decoded.states.iter().map(|s| s.to_bytes()).collect();
                        while q_idx < sorted.len() && sorted[q_idx] <= rg.max {
                            let q2 = sorted[q_idx];
                            if q2 >= rg.min {
                                if let Ok(pos) = decoded_bytes.binary_search(&q2) {
                                    results.push((
                                        CompactState::from_bytes(q2),
                                        decoded.values[pos] as i32,
                                    ));
                                }
                            }
                            q_idx += 1;
                        }
                        rg_idx += 1;
                    }
                }

                Ok(results)
            }
        }
    }

    /// Create a new policy database from a transposition table.
    ///
    /// Only states where `completed_steps % step_interval == 0` are saved.
    /// Returns the number of entries written (after filtering).
    pub fn write(
        mechanics: &QGameMechanics,
        entries: TranspositionTable,
        path: &str,
        board_size: usize,
        max_steps: usize,
        max_walls: usize,
        step_interval: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        // Drain DashMap, apply step_interval filter, sort by state bytes
        // (matches Parquet FixedSizeBinary statistics ordering).
        let mut rows: Vec<([u8; STATE_BYTES], i8)> = entries
            .into_iter()
            .filter_map(|(s, v)| {
                let steps = mechanics.repr().get_completed_steps(s);
                if steps % step_interval == 0 {
                    Some((s.to_bytes(), v))
                } else {
                    None
                }
            })
            .collect();
        rows.sort_unstable_by_key(|(s, _)| *s);

        let num_rows = rows.len();
        let kv_metadata = vec![
            KeyValue {
                key: "board_size".to_string(),
                value: Some(board_size.to_string()),
            },
            KeyValue {
                key: "max_walls".to_string(),
                value: Some(max_walls.to_string()),
            },
            KeyValue {
                key: "max_steps".to_string(),
                value: Some(max_steps.to_string()),
            },
            KeyValue {
                key: "num_states".to_string(),
                value: Some(num_rows.to_string()),
            },
        ];

        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
            .set_max_row_group_size(ROW_GROUP_SIZE)
            .set_key_value_metadata(Some(kv_metadata))
            .build();

        let schema = schema();
        let file = File::create(Path::new(path))?;
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

        // Stream rows in moderate chunks so we don't allocate one giant batch.
        for chunk in rows.chunks(WRITE_CHUNK_SIZE) {
            let states_arr = Arc::new(FixedSizeBinaryArray::try_from_iter(
                chunk.iter().map(|(s, _)| s.to_vec()),
            )?);
            let values: Vec<i8> = chunk.iter().map(|(_, v)| *v).collect();
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![states_arr, Arc::new(Int8Array::from(values))],
            )?;
            writer.write(&batch)?;
        }
        writer.close()?;

        Ok(num_rows)
    }
}

/// Append a record batch's two columns to the running state/value vectors.
fn append_batch(
    batch: &RecordBatch,
    states: &mut Vec<CompactState>,
    values: &mut Vec<i8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let s_col = batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .ok_or("state column is not FixedSizeBinary")?;
    let v_col = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int8Array>()
        .ok_or("value column is not Int8")?;
    if s_col.value_length() != STATE_BYTES as i32 {
        return Err(format!(
            "state column width {} != expected {}",
            s_col.value_length(),
            STATE_BYTES
        )
        .into());
    }
    for i in 0..s_col.len() {
        states.push(bytes_to_state(s_col.value(i))?);
    }
    values.extend_from_slice(v_col.values());
    Ok(())
}

/// Get all valid actions (moves + wall placements) for the current player.
fn get_all_actions(mechanics: &QGameMechanics, data: &mut CompactState) -> Vec<(u8, u8, u8)> {
    let moves = mechanics.get_valid_moves(*data);
    let mut actions: Vec<(u8, u8, u8)> = moves
        .into_iter()
        .map(|(r, c)| (r as u8, c as u8, 2))
        .collect();
    let walls = mechanics.get_valid_wall_placements(data);
    actions.extend(
        walls
            .into_iter()
            .map(|(r, c, t)| (r as u8, c as u8, t as u8)),
    );
    actions
}

/// Minimax evaluation with transposition table.
///
/// Returns the value from player 0's (absolute) perspective:
/// - `1` = player 0 wins with best play
/// - `0` = game is a tie with best play
/// - `-1` = player 1 wins with best play
///
/// All reachable non-terminal states and their values are stored in the
/// transposition table for later export to a policy database.
pub fn minimax(
    mechanics: &QGameMechanics,
    data: &mut CompactState,
    transposition_table: &TranspositionTable,
) -> i8 {
    minimax_inner(mechanics, data, transposition_table, None)
}

fn minimax_inner(
    mechanics: &QGameMechanics,
    data: &mut CompactState,
    transposition_table: &TranspositionTable,
    mut rng: Option<&mut StdRng>,
) -> i8 {
    if let Some(entry) = transposition_table.get(data) {
        return *entry;
    }

    let current_player = mechanics.repr().get_current_player(*data);
    let opponent = 1 - current_player;

    // Terminal states: return value directly without storing in transposition table.
    if mechanics.check_win(*data, opponent) {
        return match opponent {
            0 => 1,
            1 => -1,
            _ => panic!("Bad player number ({})", opponent),
        };
    }
    if mechanics.repr().get_completed_steps(*data) >= mechanics.repr().max_steps() {
        return 0;
    }

    // Not a terminal state. Recurse.

    let mut actions = get_all_actions(mechanics, data);
    assert!(
        !actions.is_empty(),
        "No valid actions - should never happen"
    );

    // Shuffle actions if an RNG is provided (for Lazy SMP)
    if let Some(ref mut r) = rng {
        actions.shuffle(*r);
    }

    let is_maximizing = current_player == 0;
    let mut best_value: i8 = if is_maximizing { -1 } else { 1 };

    for &(row, col, action_type) in &actions {
        let mut new_data = *data;
        let (r, c, t) = (row as usize, col as usize, action_type as usize);
        if action_type == 2 {
            mechanics.execute_move(
                &mut new_data,
                mechanics.repr().get_current_player(*data),
                r,
                c,
            );
        } else {
            mechanics.execute_wall_placement(&mut new_data, current_player, r, c, t);
        }

        mechanics.switch_player(&mut new_data);

        let child_value = minimax_inner(
            mechanics,
            &mut new_data,
            transposition_table,
            rng.as_deref_mut(),
        );

        if is_maximizing {
            if child_value > best_value {
                best_value = child_value;
            }
        } else {
            if child_value < best_value {
                best_value = child_value;
            }
        }
    }

    transposition_table.insert(*data, best_value);

    best_value
}

/// Lazy SMP parallel minimax. Spawns `num_threads` threads each running
/// the full minimax with randomized move ordering, all sharing the same
/// transposition table. Returns the root value.
pub fn minimax_lazy_smp(
    mechanics: &QGameMechanics,
    data: &mut CompactState,
    transposition_table: &TranspositionTable,
    num_threads: usize,
) -> i8 {
    let data_snapshot = *data;
    let result = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);
        for i in 0..num_threads {
            let mut thread_data = data_snapshot;
            let tt = &transposition_table;
            let mech = &mechanics;
            handles.push(s.spawn(move || {
                let mut rng = StdRng::seed_from_u64(i as u64);
                minimax_inner(mech, &mut thread_data, tt, Some(&mut rng))
            }));
        }
        let mut results = Vec::with_capacity(num_threads);
        for h in handles {
            results.push(h.join().expect("minimax thread panicked"));
        }
        results[0]
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_minimax_initial_state_3x3() {
        let mechanics = QGameMechanics::new(3, 0, 10);
        let mut data = mechanics.create_initial_state();
        let table = TranspositionTable::new();

        let value = minimax(&mechanics, &mut data, &table);

        assert!(value == -1, "P1 should always lose 3x3 with no walls");
    }

    #[test]
    fn test_minimax_win_in_one() {
        let mechanics = QGameMechanics::new(3, 0, 10);
        let mut data = mechanics.create_initial_state();

        mechanics.repr().set_player_position(&mut data, 0, 1, 1);
        mechanics.repr().set_player_position(&mut data, 1, 2, 0);
        mechanics.repr().set_current_player(&mut data, 0);
        mechanics.repr().set_completed_steps(&mut data, 2);

        let table = TranspositionTable::new();
        let value = minimax(&mechanics, &mut data, &table);

        assert_eq!(value, 1, "Player 0 should be able to win in one move");
    }

    #[test]
    fn test_minimax_tie_at_max_steps() {
        let mechanics = QGameMechanics::new(3, 0, 2);
        let mut data = mechanics.create_initial_state();
        mechanics.repr().set_completed_steps(&mut data, 2);

        let table = TranspositionTable::new();
        let value = minimax(&mechanics, &mut data, &table);

        assert_eq!(value, 0, "Max steps reached should be a tie");
    }

    #[test]
    fn test_transposition_table_populated() {
        let mechanics = QGameMechanics::new(3, 0, 4);
        let mut data = mechanics.create_initial_state();
        let table = TranspositionTable::new();

        minimax(&mechanics, &mut data, &table);

        assert!(!table.is_empty(), "Transposition table should have entries");
    }

    /// End-to-end: write a small DB, reopen it, verify metadata, count, and
    /// every state we wrote can be looked up by both rowid and state value.
    #[test]
    fn test_write_then_read_roundtrip() {
        let mechanics = QGameMechanics::new(3, 0, 8);
        let mut root = mechanics.create_initial_state();
        let table = TranspositionTable::new();
        minimax(&mechanics, &mut root, &table);
        assert!(!table.is_empty());

        // Snapshot expected entries before write() drains the DashMap.
        let expected: Vec<(CompactState, i8)> =
            table.iter().map(|kv| (*kv.key(), *kv.value())).collect();

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        let path_str = path.to_str().unwrap();
        let written = PolicyDb::write(
            &mechanics, table, path_str, 3, // board_size
            8, // max_steps
            0, // max_walls
            1, // step_interval (keep all)
        )
        .unwrap();
        assert_eq!(written, expected.len());

        let db = PolicyDb::open(path_str, false).unwrap();
        assert_eq!(db.read_metadata().unwrap(), (3, 0, 8, Some(expected.len())));
        assert_eq!(db.count_states().unwrap(), expected.len());

        // Round-trip every entry by state lookup.
        let states: Vec<CompactState> = expected.iter().map(|(s, _)| *s).collect();
        let pairs = db.lookup_values_by_state(&states).unwrap();
        assert_eq!(pairs.len(), expected.len());
        let got_map: HashMap<CompactState, i32> = pairs.into_iter().collect();
        for (s, v) in &expected {
            assert_eq!(got_map.get(s), Some(&(*v as i32)));
        }

        // Round-trip every entry by sequential rowid scan.
        let all_ids: Vec<i64> = (1..=expected.len() as i64).collect();
        let by_id = db.fetch_states_by_rowid(&all_ids).unwrap();
        assert_eq!(by_id.len(), expected.len());
        let by_id_map: HashMap<CompactState, i32> = by_id.into_iter().collect();
        for (s, v) in &expected {
            assert_eq!(by_id_map.get(s), Some(&(*v as i32)));
        }
    }

    /// `lookup_action_values` should resolve terminal child states inline
    /// (without DB lookup) and DB-resident children via the Parquet sweep.
    #[test]
    fn test_lookup_action_values_terminal() {
        let mechanics = QGameMechanics::new(3, 0, 8);
        let mut root = mechanics.create_initial_state();
        let table = TranspositionTable::new();
        minimax(&mechanics, &mut root, &table);

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        let path_str = path.to_str().unwrap();
        PolicyDb::write(&mechanics, table, path_str, 3, 8, 0, 1).unwrap();

        let db = PolicyDb::open(path_str, false).unwrap();

        // Construct a state where P0 is one row from goal — at least one
        // child action wins immediately (terminal child, no DB lookup).
        let mut state = mechanics.create_initial_state();
        mechanics.repr().set_player_position(&mut state, 0, 1, 1);
        mechanics.repr().set_player_position(&mut state, 1, 2, 0);
        mechanics.repr().set_current_player(&mut state, 0);
        mechanics.repr().set_completed_steps(&mut state, 2);

        let result = db.lookup_action_values(state).unwrap();
        let (actions, values) = result.expect("should have valid actions");
        assert_eq!(actions.len(), values.len());
        assert!(!actions.is_empty());
        // At least one action should yield a winning value (1) from the
        // acting player's perspective.
        assert!(
            values.iter().any(|&v| v == 1),
            "expected at least one winning action value, got {values:?}"
        );
    }

    /// Eager and lazy modes must return identical results across all
    /// public query methods.
    #[test]
    fn test_eager_matches_lazy() {
        let mechanics = QGameMechanics::new(3, 0, 8);
        let mut root = mechanics.create_initial_state();
        let table = TranspositionTable::new();
        minimax(&mechanics, &mut root, &table);
        let expected: Vec<(CompactState, i8)> =
            table.iter().map(|kv| (*kv.key(), *kv.value())).collect();

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        let path_str = path.to_str().unwrap();
        PolicyDb::write(&mechanics, table, path_str, 3, 8, 0, 1).unwrap();

        let eager = PolicyDb::open(path_str, false).unwrap();
        let lazy = PolicyDb::open(path_str, true).unwrap();

        // Same metadata.
        assert_eq!(
            eager.read_metadata().unwrap(),
            lazy.read_metadata().unwrap()
        );
        assert_eq!(eager.count_states().unwrap(), lazy.count_states().unwrap());

        // Same lookup_values_by_state results (sort both since order isn't
        // guaranteed across modes).
        let states: Vec<CompactState> = expected.iter().map(|(s, _)| *s).collect();
        let mut e_pairs = eager.lookup_values_by_state(&states).unwrap();
        let mut l_pairs = lazy.lookup_values_by_state(&states).unwrap();
        e_pairs.sort_unstable_by_key(|(s, _)| s.to_bytes());
        l_pairs.sort_unstable_by_key(|(s, _)| s.to_bytes());
        assert_eq!(e_pairs, l_pairs);
        assert_eq!(e_pairs.len(), expected.len());

        // Same fetch_states_by_rowid results in sequential order. Both
        // modes return rows in ascending rowid order, which is by
        // sorted-state order on disk.
        let all_ids: Vec<i64> = (1..=expected.len() as i64).collect();
        let e_by_id = eager.fetch_states_by_rowid(&all_ids).unwrap();
        let l_by_id = lazy.fetch_states_by_rowid(&all_ids).unwrap();
        assert_eq!(e_by_id, l_by_id);

        // Same lookup_action_values: pick a non-terminal state from the
        // table and compare the (actions, values) tuple. Use sorting
        // since action enumeration order is identical but we sort to be
        // robust against any future reordering.
        let probe_state = expected
            .iter()
            .find(|(s, _)| {
                !mechanics.check_win(*s, 0)
                    && !mechanics.check_win(*s, 1)
                    && mechanics.repr().get_completed_steps(*s) < mechanics.repr().max_steps()
            })
            .map(|(s, _)| *s)
            .expect("expected at least one non-terminal state");

        let (e_actions, e_values) = eager
            .lookup_action_values(probe_state)
            .unwrap()
            .expect("eager: action values");
        let (l_actions, l_values) = lazy
            .lookup_action_values(probe_state)
            .unwrap()
            .expect("lazy: action values");

        let mut e_pairs: Vec<((u8, u8, u8), i32)> = e_actions.into_iter().zip(e_values).collect();
        let mut l_pairs: Vec<((u8, u8, u8), i32)> = l_actions.into_iter().zip(l_values).collect();
        e_pairs.sort_unstable_by_key(|(a, _)| *a);
        l_pairs.sort_unstable_by_key(|(a, _)| *a);
        assert_eq!(e_pairs, l_pairs);
    }
}
