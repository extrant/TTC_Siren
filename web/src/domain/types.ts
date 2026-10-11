// Types of the data the server sends (docs/design/GAME_LOG_SCHEMA.md, EXPLANATION_FACTS.md, API.md section 5). Only what the UI reads is typed; unknown fields are ignored.

export type Lang = 'zh-CN' | 'en'
export type Owner = 'r' | 'b' | null

/** A card as the engine writes it: [id|null, type|null, owner, modifier, base u r d l, display u r d l, canUse]. */
export type WireCard = [number | null, string | null, Owner, number, number, number, number, number, number, number, number, number, number]

export interface WireState {
  cur: number
  board: (WireCard | null)[]
  hands: WireCard[][]
  generated?: number[][]
}

export type Cause = 'basic' | 'reverse' | 'fallen_ace' | 'same' | 'plus' | 'combo'
export const CAUSES: readonly Cause[] = ['basic', 'reverse', 'fallen_ace', 'same', 'plus', 'combo']

export interface FlipFact {
  row: number
  col: number
  card_id: number
  cause: Cause
  my_side: string
  my_value: number
  their_value: number
  sum?: number
  via_row?: number
  via_col?: number
  combo_kind?: string
}

export interface MoveFact {
  card_id: number
  row: number
  col: number
  slot: number
  edges: [number, number, number, number]
  type: string | null
  assumed: boolean
}

export interface ImmediateEffectFact {
  type: 'immediate_effect'
  move: MoveFact
  flips: FlipFact[]
  flip_count: number
  cards_before: { mine: number; theirs: number }
  cards_after: { mine: number; theirs: number }
  modifier_changes: { type: string; from: number; to: number; board_cards: number; hand_cards: number }[]
  active_rules: string[]
}

export interface PvPly {
  ply: number
  side: 'self' | 'opponent'
  card_id: number
  row: number
  col: number
  flips: number
}

export interface OutcomeFact {
  type: 'outcome'
  kind: 'exact' | 'heuristic'
  result: string
  pv: PvPly[]
  depth?: number
  score?: number | null
  difference?: number | null
  basis?: string
  empty_cells?: number
}

export interface RobustFact {
  type: 'robust_endgame'
  scenarios: number
  safety_ratio: number
  safety_percent: number
  average_difference: number | null
  worst_difference: number | null
  overridden: boolean
  override_rule: string | null
}

export interface UnknownFact {
  type: 'unknown_cards'
  opponent_unknown: number
  own_unknown?: number
  strategy: string
  primary_rule?: string | null
  npc_name?: string | null
  npc_pool_size?: number
  assumed: { slot: number; card_id: number; star: number; type: string | null }[]
}

export interface ConfidenceFact {
  type: 'confidence'
  solved_to_end: boolean
  depth_reached: number
  depth_requested: number
  empty_cells: number
  nodes: number
  search_ms: number
  stopped_early: boolean
  robust_used: boolean
}

export interface ExplanationText {
  brief: string
  full: string
}

export interface Explanation {
  version: number
  level: string
  language?: Lang
  facts: Record<string, unknown>[]
  text: ExplanationText
  texts?: Partial<Record<Lang, ExplanationText>>
}

export interface MoveRef {
  slot: number
  row: number
  col: number
  card_id: number
}

export interface DepthRow {
  depth: number
  score: number
  move: MoveRef | null
  nodes: number
  tt_hits?: number
  cutoffs?: number
  elapsed_ms: number
}

export interface BaseRecord {
  v: number
  seq: number
  type: string
  t: string
  elapsed_ms: number
}

export interface GameStartRecord extends BaseRecord {
  type: 'game_start'
  game_id: string
  id_source: string
  rules: string[]
  rules_text: string
  open_mode: string
  my_owner: 'red' | 'blue'
  npc_name: string | null
  npc_ids: number[]
}

export interface RequestResult {
  move: MoveRef | null
  standard_move: MoveRef | null
  standard_score: number | null
  depth_reached: number
  nodes: number
  tt_hits: number
  cutoffs: number
  root_moves: number
  search_ms: number
  depths: { depth: number; score: number; move: MoveRef | null; nodes_total: number; elapsed_ms: number }[]
  robust: { scenarios: number; overridden: boolean; elapsed_ms: number } | null
}

export interface RequestRecord extends BaseRecord {
  type: 'request'
  request_no: number
  seed: string
  settings: Record<string, unknown>
  request_raw?: unknown
  rules: string[]
  state: WireState | null
  id_inference: string | null
  http_status: number
  error: unknown
  result?: RequestResult
  explanation?: Explanation
}

export interface OpponentFlip {
  row: number
  col: number
  card_id: number
  cause: Cause | null
}

export interface OpponentMoveRecord extends BaseRecord {
  type: 'opponent_move'
  between: [number, number]
  consistent: boolean
  card_id: number | null
  card_edges: number[] | null
  ambiguous?: boolean
  candidate_ids?: number[]
  row: number | null
  col: number | null
  owner: 'red' | 'blue'
  flips: OpponentFlip[]
  my_move: { row: number; col: number; as_recommended: boolean } | null
  new_cells: number[]
  note: string | null
}

export interface GameEndRecord extends BaseRecord {
  type: 'game_end'
  source: string
  final: { mine: number; theirs: number }
  winner: 'mine' | 'theirs' | 'draw'
  board: (WireCard | null)[]
}

export type LogRecord = GameStartRecord | RequestRecord | OpponentMoveRecord | GameEndRecord

export interface GameSummary {
  game_id: string
  started: string
  last_write: string
  records: number
  requests: number
  ended: boolean
  winner: string | null
  my_owner: string | null
  rules: string | null
  id_source: string | null
}
