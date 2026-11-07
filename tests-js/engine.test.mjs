// JS live engine (desktop/plugin.js) vs the Python simulator (engine.analyses.process_audio).
//
//   python tests-js/export_vectors.py     # once, or after an engine change
//   node --test tests-js/
//
// Every vector is the same plucked-string riff run through one circuit in Python;
// the JS engine (the same code the AudioWorklet runs) must reproduce it within 1% RMS.
import assert from 'node:assert/strict'
import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { test } from 'node:test'

import { loadEngine } from './load-engine.mjs'

const HW = await loadEngine()
const dir = new URL('./vectors/', import.meta.url)
const files = existsSync(dir) ? readdirSync(dir).filter(f => f.endsWith('.json')).sort() : []

function relRms(a, b) {
  let num = 0
  let den = 0
  for (let k = 0; k < b.length; k++) {
    const d = a[k] - b[k]
    num += d * d
    den += b[k] * b[k]
  }
  return Math.sqrt(num / Math.max(den, 1e-30))
}

test('vectors exist (run python tests-js/export_vectors.py)', () => {
  assert.ok(files.length >= 6, `only ${files.length} vectors in tests-js/vectors`)
})

const results = []

for (const f of files) {
  const v = JSON.parse(readFileSync(new URL(f, dir), 'utf8'))
  test(`${v.name}: JS matches Python within 1% RMS`, () => {
    const t0 = performance.now()
    const lc = new HW.LiveCircuit(v.compiled, v.sr)
    const y = lc.process(Float64Array.from(v.input))
    const secs = (performance.now() - t0) / 1000
    assert.equal(y.length, v.output.length)
    assert.ok(lc.converged, 'DC operating point did not converge')
    const err = relRms(y, v.output)
    results.push({ name: v.name, err, rt: secs / (y.length / v.sr) })
    assert.ok(err < 0.01, `relative RMS error ${(err * 100).toFixed(3)}%`)
  })
}

test('pot taper law matches the spec', () => {
  assert.equal(HW.potFraction(0.5, 'lin'), 0.5)
  assert.ok(Math.abs(HW.potFraction(0.5, 'log') - (Math.sqrt(11) - 1) / 10) < 1e-12)
  assert.ok(Math.abs(HW.potFraction(0.5, 'revlog') - (1 - (Math.sqrt(11) - 1) / 10)) < 1e-12)
  assert.deepEqual(HW.potHalves(100, 0, 'lin'), [1, 100])
})

test('moving a pot mid-stream changes the sound and stays finite', () => {
  const f = files.find(n => n.startsWith('fuzz'))
  assert.ok(f, 'fuzz vector missing')
  const v = JSON.parse(readFileSync(new URL(f, dir), 'utf8'))
  const lc = new HW.LiveCircuit(v.compiled, v.sr)
  const pot = v.compiled.pots.find(p => /vol/i.test(p.label)) || v.compiled.pots[0]
  const x = v.input
  let before = 0
  let after = 0
  const half = Math.floor(x.length / 2)
  for (let k = 0; k < half; k++) before += lc.tick(x[k]) ** 2
  lc.setPot(pot.name, 0.05)
  for (let k = half; k < x.length; k++) {
    const y = lc.tick(x[k])
    assert.ok(Number.isFinite(y))
    after += y ** 2
  }
  assert.ok(after < before * 0.5, `turning ${pot.label} down should make it quieter (${after} vs ${before})`)
})

test('setting a pot to its compiled position reproduces the compiled values', () => {
  const v = JSON.parse(readFileSync(new URL(files.find(n => n.startsWith('fuzz-knobs')), dir), 'utf8'))
  const net = new HW.Network(v.compiled)
  for (const p of v.compiled.pots) {
    const [ra, rb] = HW.potHalves(p.value, p.pos, p.taper)
    const a = v.compiled.elements.find(e => e.name === p.halves[0]).value
    const b = v.compiled.elements.find(e => e.name === p.halves[1]).value
    assert.ok(Math.abs(ra - a) / a < 1e-9 && Math.abs(rb - b) / b < 1e-9)
    assert.ok(net.setPot(p.name, p.pos))
  }
})

test('summary', () => {
  if (!results.length) return
  const worst = results.reduce((a, b) => (a.err > b.err ? a : b))
  console.log(`\n  ${results.length} circuits; worst ${worst.name} ${(worst.err * 100).toExponential(2)}% RMS`)
  for (const r of results) console.log(`  ${r.name.padEnd(30)} ${(r.err * 100).toExponential(2)}%  x${r.rt.toFixed(2)} real time`)
})
