// Reads the card database of the project (data/幻卡数据库.csv: edges, star, type; the names are not used) and writes src/generated/cards.json:
// { "cards": { "<id>": [top, right, bottom, left, star, typeIndex] } } with typeIndex 0 = none, 1..4 = 兽人 拂晓 帝国 蛮神.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const csv = join(here, '..', '..', 'data', '幻卡数据库.csv')
const text = readFileSync(csv, 'utf8').replace(/^﻿/, '')
const lines = text.split(/\r?\n/).filter((l) => l.length > 0)
const header = lines[0].split(',')
const col = (name) => {
  const i = header.indexOf(name)
  if (i < 0) throw new Error(`column ${name} not found in ${csv}`)
  return i
}
const [cId, cTop, cBottom, cLeft, cRight, cStar, cType] = ['序号', 'Top', 'Bottom', 'Left', 'Right', '星级', 'TripleTriadCardType'].map(col)
const types = ['', '兽人', '拂晓', '帝国', '蛮神']
const cards = {}
for (const line of lines.slice(1)) {
  const f = line.split(',')
  const id = Number(f[cId])
  if (!Number.isInteger(id)) continue
  cards[id] = [Number(f[cTop]), Number(f[cRight]), Number(f[cBottom]), Number(f[cLeft]), Number(f[cStar]), Math.max(0, types.indexOf(f[cType] ?? ''))]
}
const out = join(here, '..', 'src', 'generated', 'cards.json')
mkdirSync(dirname(out), { recursive: true })
writeFileSync(out, JSON.stringify({ cards }) + '\n')
console.log(`cards.json: ${Object.keys(cards).length} cards`)
