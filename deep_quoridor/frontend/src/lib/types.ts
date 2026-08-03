export type Orientation = "h" | "v";

export type EnrichedAction =
  | { kind: "move"; index: number; to: [number, number] }
  | { kind: "wall"; index: number; row: number; col: number; orientation: Orientation };

export interface WallEntry { row: number; col: number; orientation: Orientation }

export interface StateView {
  board_size: number;
  max_walls: number;
  max_steps: number;
  current_player: number;
  p1_pos: [number, number];
  p2_pos: [number, number];
  p1_walls: number;
  p2_walls: number;
  walls: WallEntry[];
  legal_actions: EnrichedAction[];
  completed_steps: number;
  winner: number | null;
  human_player: number;
  last_action: EnrichedAction | null;
  move_history: number[];
}

export interface SearchResult {
  action: number;
  rootValue: number;
  children: { actionIndex: number; visitCount: number }[];
}
