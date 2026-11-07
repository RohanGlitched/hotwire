// Display helpers in desktop/plugin.js: units, titles, picker labels, drawing titles, board counts.
//
//   node --test tests-js/ui-text.test.mjs
//
// The plugin file imports the Hermes SDK, so the pure helpers are lifted out of its text by name
// into a temporary module (the shipped file itself stays one ESM file with no eval).
import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { test } from 'node:test'
import { fileURLToPath, pathToFileURL } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const src = readFileSync(join(here, '..', 'desktop', 'plugin.js'), 'utf8')

function lift(name) {
  const start = src.indexOf(`\nfunction ${name}(`)
  if (start < 0) {
    const m = src.match(new RegExp(`\\nconst ${name} = [^\\n]+`))
    if (!m) throw new Error(`${name} not found in plugin.js`)
    return m[0]
  }
  let i = src.indexOf('{', start)
  let depth = 0
  for (; i < src.length; i++) {
    if (src[i] === '{') depth++
    else if (src[i] === '}' && --depth === 0) break
  }
  return src.slice(start, i + 1)
}

const names = ['tidy', 'sentence', 'plural', 'shortDate', 'designTitle', 'pickerLabels', 'tidyNote', 'xmlEsc', 'reEsc', 'stripSvgTitle', 'boardCounts', 'specLabel', 'specTarget', 'fmtNum', 'specMeasured']
const dir = mkdtempSync(join(tmpdir(), 'hotwire-ui-'))
const file = join(dir, 'ui.mjs')
writeFileSync(file, names.map(lift).join('\n') + `\nexport { ${names.join(', ')} }\n`)
const ui = await import(pathToFileURL(file).href)

test('one unit style: a space before units, a real micro sign, % hugs its number', () => {
  assert.equal(ui.tidy('3.5V to 6V'), '3.5 V to 6 V')
  assert.equal(ui.tidy('below 2mA'), 'below 2 mA')
  assert.equal(ui.tidy('2.2uF electrolytic'), '2.2 µF electrolytic')
  assert.equal(ui.tidy('above 20 %'), 'above 20%')
  assert.equal(ui.tidy('gain at 1000Hz is 61.4 dB'), 'gain at 1000 Hz is 61.4 dB')
  assert.equal(ui.tidy('2N3904 at C4'), '2N3904 at C4')
  assert.equal(ui.sentence('draws 1.02mA'), 'Draws 1.02 mA')
  assert.equal(ui.plural(1, 'link'), '1 link')
  assert.equal(ui.plural(4, 'track cut'), '4 track cuts')
  assert.equal(ui.fmtNum(0.00102, 'A'), '1.02 mA')
  assert.equal(ui.fmtNum(2.2e-6, 'F'), '2.2 µF')
})

test('agent notes read as text', () => {
  assert.equal(ui.tidyNote('  R3 10k -> 4.7k: v(c2) inside 3.5-6 V '), 'R3 10k → 4.7k: v(c2) inside 3.5–6 V')
})

test('spec rows lead with plain English and fall back gracefully', () => {
  assert.equal(ui.specLabel({ spec: 'gain_db(1000) > 25 dB', label: 'Gain at 1 kHz' }), 'Gain at 1 kHz')
  assert.equal(ui.specLabel({ spec: 'gain_db(1000) > 25 dB' }), 'gain_db(1000) > 25 dB')
  assert.equal(ui.specTarget({ target: '3.5V to 6V' }), '3.5 V to 6 V')
  assert.equal(ui.specMeasured({ measured: 0.00102, unit: 'A' }), '1.02 mA')
  assert.equal(ui.specMeasured({ measured_text: '4.93 V', measured: 4.93 }), '4.93 V')
  assert.equal(ui.specMeasured({ outdated: true, measured: null }), '—')
})

test('design titles: a human name, else the circuit title, else a tidy slug', () => {
  assert.equal(ui.designTitle({ id: 'silicon-fuzz-for-rohan', name: 'Silicon fuzz for Rohan', title: 'Two-transistor silicon fuzz' }), 'Silicon fuzz for Rohan')
  assert.equal(ui.designTitle({ id: 'quarter-clean-fuzz', name: 'quarter-clean-fuzz', title: 'Two-transistor silicon fuzz' }), 'Two-transistor silicon fuzz')
  assert.equal(ui.designTitle({ id: 'qcf-a', name: 'qcf-a', title: 'v' }), 'Qcf a')
  assert.equal(ui.designTitle({ id: 'x', name: 'x', title: 'Untitled circuit' }), 'X')
})

test('picker tells duplicates apart without glyphs', () => {
  const rows = [
    { id: 'a', name: 'a', title: 'Fuzz', latest: 3, updated: 1793995047 },
    { id: 'b', name: 'b', title: 'Fuzz', latest: 1, updated: 1793995047 },
    { id: 'c', name: 'c', title: 'Boost', latest: 1, updated: 1793995047 },
    { id: 'd', name: 'd', title: 'Wah', latest: 1, updated: 1793995047 },
    { id: 'e', name: 'e', title: 'Wah', latest: 1, updated: 1793995047 }
  ]
  const l = ui.pickerLabels(rows)
  assert.deepEqual(l.get('a'), { title: 'Fuzz', suffix: 'v3' })
  assert.deepEqual(l.get('b'), { title: 'Fuzz', suffix: 'v1' })
  assert.deepEqual(l.get('c'), { title: 'Boost', suffix: '' })
  assert.deepEqual([l.get('d').suffix, l.get('e').suffix], ['d', 'e'])
  for (const v of l.values()) assert.ok(!/[✓✗~]/.test(v.title + v.suffix))
})

test('drawing titles come off on the bench, and the schematic band with them', () => {
  const svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="-52 -319 581 476" width="581" height="476" class="hotwire-schematic"><title>Fuzz &amp; co (schematic)</title><g class="hw-ink"><text x="-34" y="-297" font-size="14" font-weight="600" fill="currentColor">Fuzz &amp; co</text><path d="M0 0"/></g></svg>'
  const out = ui.stripSvgTitle(svg, 'Fuzz & co')
  assert.ok(!out.includes('font-weight="600"'))
  assert.ok(out.includes('<title>Fuzz &amp; co (schematic)</title>'))
  assert.ok(out.includes('viewBox="-52 -291 581 448"') && out.includes('height="448"'))
  assert.equal(ui.stripSvgTitle(svg, 'Another title'), svg)
})

test('the chat board card counts what the drawing prints', () => {
  const svg = '<svg><text>Stripboard 14 × 7 holes (1.4 x 0.7 in) · 10 parts on board · 4 track cuts · 1 links · 8 off-board pads</text></svg>'
  const r = { steps: new Array(15).fill('x'), board: { cuts: 4, links: 1 }, offboard: [1, 2, 3, 4, 5] }
  assert.equal(ui.boardCounts(r, svg), '15 build steps · 4 track cuts · 1 wire link · 8 off-board pads')
  assert.equal(ui.boardCounts({ board: { cuts: 2, links: 0 } }, null), '2 track cuts · 0 wire links')
})
