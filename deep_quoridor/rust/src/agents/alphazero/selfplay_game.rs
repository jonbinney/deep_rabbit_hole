//! Async self-play game runner.
//!
//! Plays a complete game between two `LeafParallelMCTS` agents (or one MCTS +
//! one "random" baseline). The replay buffer mirrors `game_runner::play_game`'s
//! current-player-downward storage so downstream training is unchanged.

use std::collections::HashSet;

use anyhow::Result;
use ndarray::Axis;

use crate::agents::alphazero::selfplay_mcts::{LeafParallelMCTS, SearchStats};
use crate::compact::q_bit_repr::CompactState;
use crate::compact::q_game_mechanics::QGameMechanics;
use crate::game_runner::{GameResult, ReplayBufferItem};
use crate::grid_helpers::compact_state_to_resnet_input;
use crate::rotation::{create_rotation_mapping, remap_mask, remap_policy, rotate_compact_state};

/// P2 agent variants. Today only AlphaZero (same as P1) or `Random` are wired up.
pub enum P2 {
    AlphaZero(LeafParallelMCTS),
    Random,
}

/// Apply temperature-and-sample to visit counts. Same semantics as
/// `agent::apply_temperature_and_sample`; duplicated here so we don't depend on
/// the sync `AlphaZeroAgent` plumbing.
fn sample_action(
    visit_counts: &[u32],
    action_indices: &[usize],
    temperature: f32,
    deterministic: bool,
) -> usize {
    use rand::Rng;
    assert!(!visit_counts.is_empty() && !action_indices.is_empty());
    if temperature == 0.0 {
        let max_v = visit_counts.iter().max().copied().unwrap_or(0);
        let tied: Vec<usize> = visit_counts
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v == max_v { Some(i) } else { None })
            .collect();
        if deterministic {
            return action_indices[tied[0]];
        }
        let mut rng = rand::thread_rng();
        return action_indices[tied[rng.gen_range(0..tied.len())]];
    }
    let total: f64 = visit_counts
        .iter()
        .map(|&v| (v as f64).powf(1.0 / temperature as f64))
        .sum();
    if total <= 0.0 {
        return action_indices[0];
    }
    let mut rng = rand::thread_rng();
    let r: f64 = rng.r#gen();
    let mut cum = 0.0;
    for (i, &v) in visit_counts.iter().enumerate() {
        cum += (v as f64).powf(1.0 / temperature as f64) / total;
        if r < cum {
            return action_indices[i];
        }
    }
    action_indices[action_indices.len() - 1]
}

/// Number of leading plies that define a game's "opening" for uniqueness.
pub const OPENING_PLIES: usize = 8;

/// Per-game MCTS diagnostics, summed over the game's searches, plus move-sequence hashes.
#[derive(Debug, Clone, Default)]
pub struct GameMetrics {
    pub sims: u64,
    pub terminal_wins: u64,
    pub truncations: u64,
    pub max_depth: u32,
    pub sum_depth: u64,
    pub moves: u64,
    pub sum_root_entropy: f64,
    pub sum_top_move_frac: f64,
    pub sum_nodes: u64,
    pub sum_internal_nodes: u64,
    pub full_hash: u64,
    pub opening_hash: u64,
}

/// Deterministic (within-process) hash of a move-index sequence.
fn hash_actions(actions: &[usize]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    actions.hash(&mut h);
    h.finish()
}

/// Per-game settings for action selection.
#[derive(Debug, Clone, Copy)]
pub struct GameSettings {
    pub temperature: f32,
    pub drop_t_on_step: Option<usize>,
    pub deterministic_tie_break: bool,
}

pub async fn play_game_async(
    p1: &mut LeafParallelMCTS,
    p2: &mut P2,
    settings: GameSettings,
    board_size: i32,
    max_walls: i32,
    max_steps: i32,
) -> Result<(GameResult, GameMetrics)> {
    let mechanics =
        QGameMechanics::new(board_size as usize, max_walls as usize, max_steps as usize);
    let mut data = mechanics.create_initial_state();
    let (orig_to_rot, _) = create_rotation_mapping(board_size);
    let mut replay_items: Vec<ReplayBufferItem> = Vec::new();
    let visited = HashSet::new();
    let mut winner: Option<i32> = None;
    let mut gm = GameMetrics::default();
    let mut actions: Vec<usize> = Vec::new();

    for step in 0..max_steps {
        let current_player = mechanics.repr().get_current_player(data) as i32;
        let mask = mechanics.get_action_mask_immut(data);
        if !mask.iter().any(|&m| m) {
            break;
        }

        let resnet_input = compact_state_to_resnet_input(&mechanics, data);

        let (action_idx, policy, stats) = if current_player == 0 {
            let (a, p, s) =
                run_az_select(p1, data, &mechanics, &visited, settings, step as usize).await?;
            (a, p, Some(s))
        } else {
            match p2 {
                P2::AlphaZero(m) => {
                    let (a, p, s) =
                        run_az_select(m, data, &mechanics, &visited, settings, step as usize)
                            .await?;
                    (a, p, Some(s))
                }
                P2::Random => {
                    let (a, p) = random_select(&mask);
                    (a, p, None)
                }
            }
        };
        if let Some(s) = stats {
            gm.moves += 1;
            gm.sims += s.sims as u64;
            gm.terminal_wins += s.terminal_wins as u64;
            gm.truncations += s.truncations as u64;
            gm.max_depth = gm.max_depth.max(s.max_depth);
            gm.sum_depth += s.sum_depth;
            gm.sum_root_entropy += s.root_visit_entropy;
            gm.sum_top_move_frac += s.top_move_visit_frac;
            gm.sum_nodes += s.nodes as u64;
            gm.sum_internal_nodes += s.internal_nodes as u64;
        }
        actions.push(action_idx);

        // Replay capture (current-player-downward frame).
        let (stored_input_3d, stored_policy, stored_mask) = if current_player == 1 {
            let rotated_data = rotate_compact_state(&mechanics, data);
            let rotated_input = compact_state_to_resnet_input(&mechanics, rotated_data)
                .index_axis(Axis(0), 0)
                .to_owned();
            let rotated_policy = remap_policy(&policy, &orig_to_rot);
            let rotated_mask = remap_mask(&mask, &orig_to_rot);
            (rotated_input, rotated_policy, rotated_mask)
        } else {
            (
                resnet_input.index_axis(Axis(0), 0).to_owned(),
                policy.clone(),
                mask.clone(),
            )
        };
        replay_items.push(ReplayBufferItem {
            input_array: stored_input_3d,
            policy: stored_policy,
            action_mask: stored_mask,
            value: 0.0,
            player: current_player,
        });

        // Apply action, then advance roots for tree reuse on the next move.
        mechanics.apply_action_index(&mut data, action_idx);
        p1.advance_root(action_idx);
        if let P2::AlphaZero(m) = p2 {
            m.advance_root(action_idx);
        }

        if mechanics.check_win(data, current_player as usize) {
            winner = Some(current_player);
            break;
        }
    }
    if let Some(w) = winner {
        for item in replay_items.iter_mut() {
            item.value = if item.player == w { 1.0 } else { -1.0 };
        }
    }
    gm.full_hash = hash_actions(&actions);
    gm.opening_hash = hash_actions(&actions[..actions.len().min(OPENING_PLIES)]);
    let num_turns = if winner.is_some() {
        actions.len() as i32
    } else {
        max_steps
    };
    Ok((
        GameResult {
            winner,
            num_turns,
            replay_items,
        },
        gm,
    ))
}

async fn run_az_select(
    mcts: &mut LeafParallelMCTS,
    data: CompactState,
    mechanics: &QGameMechanics,
    visited: &HashSet<CompactState>,
    settings: GameSettings,
    step: usize,
) -> Result<(usize, Vec<f32>, SearchStats)> {
    let (children, _root_value, stats) = mcts.search(data, mechanics, visited).await?;
    let visit_counts: Vec<u32> = children.iter().map(|c| c.visit_count).collect();
    let action_indices: Vec<usize> = children.iter().map(|c| c.action_index).collect();
    let temperature = match settings.drop_t_on_step {
        Some(t) if step >= t => 0.0,
        _ => settings.temperature,
    };
    let action_idx = sample_action(
        &visit_counts,
        &action_indices,
        temperature,
        settings.deterministic_tie_break,
    );

    // Build full policy vector from visit counts.
    let total_visits: u32 = visit_counts.iter().sum();
    let mask = mechanics.get_action_mask_immut(data);
    let mut policy = vec![0.0f32; mask.len()];
    if total_visits > 0 {
        for c in &children {
            policy[c.action_index] = c.visit_count as f32 / total_visits as f32;
        }
    }
    Ok((action_idx, policy, stats))
}

fn random_select(mask: &[bool]) -> (usize, Vec<f32>) {
    use rand::Rng;
    let valid: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v { Some(i) } else { None })
        .collect();
    let mut rng = rand::thread_rng();
    let idx = valid[rng.gen_range(0..valid.len())];
    let mut p = vec![0.0f32; mask.len()];
    p[idx] = 1.0;
    (idx, p)
}

#[cfg(test)]
mod tests {
    use super::{OPENING_PLIES, hash_actions};

    #[test]
    fn identical_sequences_hash_equal_different_differ() {
        let a = vec![3usize, 7, 1, 9, 2, 4, 8, 0, 5, 6];
        let b = a.clone();
        let mut c = a.clone();
        c[9] = 99; // differs only after the opening

        assert_eq!(
            hash_actions(&a),
            hash_actions(&b),
            "identical games hash equal"
        );
        assert_ne!(
            hash_actions(&a),
            hash_actions(&c),
            "different full games differ"
        );

        let open_a = hash_actions(&a[..a.len().min(OPENING_PLIES)]);
        let open_c = hash_actions(&c[..c.len().min(OPENING_PLIES)]);
        assert_eq!(open_a, open_c, "same opening hashes equal");
    }
}
