// The card database as numbers only (generated at build time from data/幻卡数据库.csv by scripts/build-cards.mjs): edges, star rating and type of every card id.
// The UI needs it for the ghost cards of a principal variation and for the star pips; the log itself names cards by id and edges.
import generated from '../generated/cards.json'

export type CardType = '兽人' | '拂晓' | '帝国' | '蛮神'
export const CARD_TYPES: readonly CardType[] = ['兽人', '拂晓', '帝国', '蛮神']

export interface CardInfo {
  id: number
  edges: [number, number, number, number]
  star: number
  type: CardType | null
}

const table = generated.cards as unknown as Record<string, [number, number, number, number, number, number]>

export function cardInfo(id: number | null): CardInfo | null {
  if (id === null) return null
  const row = table[String(id)]
  if (!row) return null
  return { id, edges: [row[0], row[1], row[2], row[3]], star: row[4], type: row[5] === 0 ? null : CARD_TYPES[row[5] - 1] }
}

export function starOf(id: number | null): number {
  return cardInfo(id)?.star ?? 0
}
