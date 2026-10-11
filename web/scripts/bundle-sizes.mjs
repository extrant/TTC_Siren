// Sizes of the built files, raw and gzip (level 9), for the report: node scripts/bundle-sizes.mjs > ../docs/reports/phase8b/bundle_sizes.txt
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { gzipSync } from 'node:zlib'

const dist = new URL('../dist/', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')
const rows = []
const walk = (dir) => {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name)
    if (statSync(p).isDirectory()) walk(p)
    else {
      const data = readFileSync(p)
      rows.push({ file: p.slice(dist.length).split(/[\\/]/).join('/'), raw: data.length, gzip: /\.(js|css|html|svg|json)$/.test(name) ? gzipSync(data, { level: 9 }).length : null })
    }
  }
}
walk(dist)
rows.sort((a, b) => a.file.localeCompare(b.file))
const kb = (n) => (n / 1024).toFixed(1).padStart(8) + ' KiB'
const group = (re) => rows.filter((r) => re.test(r.file))
const sum = (rs, k) => rs.reduce((s, r) => s + (r[k] ?? r.raw), 0)
console.log('file'.padEnd(52), 'raw'.padStart(12), 'gzip'.padStart(12))
for (const r of rows) console.log(r.file.padEnd(52), kb(r.raw), r.gzip === null ? '           -' : kb(r.gzip))
const first = group(/^(index\.html|assets\/index-.*\.(js|css))$/)
const three = group(/^assets\/Board3D-.*\.js$/)
console.log('')
console.log(`first page (index.html + main js + css):  raw ${kb(sum(first, 'raw'))}  gzip ${kb(sum(first, 'gzip'))}`)
console.log(`3D table chunk (three.js, loaded only when a table is shown in 3D):  raw ${kb(sum(three, 'raw'))}  gzip ${kb(sum(three, 'gzip'))}`)
console.log(`mascot images (WebP, loaded as needed):  raw ${kb(sum(group(/\.webp$/), 'raw'))}`)
console.log(`all of dist:  raw ${kb(sum(rows, 'raw'))}`)
