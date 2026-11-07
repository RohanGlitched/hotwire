// Hotwire for Hermes Desktop: the Bench.
//
// A right pane and a full page that follow the agent's circuit designs live
// (schematic, verdicts, Bode plot, stress, parts, stripboard), pedal-style knobs
// that re-check the design as you turn them, an A/B player for rendered
// auditions, and Live mode: the circuit itself runs in an AudioWorklet so a knob
// changes the sound while it plays. Also tool media renderers for
// circuit_audition / circuit_check / circuit_build and a ::hotwire-knobs directive.
//
// One ESM file, loaded uncompiled: imports only the SDK and React, no JSX.
//
// Transcript directive (Desktop only): a paragraph that is exactly
//   ::hotwire-knobs{design="silicon-fuzz"}
// renders that design's knobs inline in the chat, with live re-check verdict
// chips, Play live, and Open in Bench. The design id is the `design` field the
// circuit_* tools return. Anything else in the attributes is ignored.
import * as sdk from '@hermes/plugin-sdk'
import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

// ===== HOTWIRE LIVE ENGINE BEGIN =====
// A port of engine/mna.py + engine/devices.py (same device equations, same
// nodal-DK trapezoidal transient). It is a plain function so its source text can
// also seed the AudioWorklet module (no eval: the text goes into a Blob).
function hotwireLiveEngine() {
  const BOLTZ = 1.38064852e-23
  const CHARGE = 1.6021766208e-19
  const TNOM = 300.15
  const VT = (BOLTZ * TNOM) / CHARGE
  const GMIN = 1e-12
  const EXP_MAX = 80.0
  const E_MAX = Math.exp(EXP_MAX)
  const OPAMP_RP = 1e3
  const OPAMP_HALF_MIN = 0.01
  const OPAMP_CLAMP_K = 4.0 // anti-windup: pole node held within mid +/- K*half (devices.py)
  const OPAMP_CLAMP_EPS = 0.01
  const VT3 = -3 * ((1.38064852e-23 * 300.15) / 1.6021766208e-19)
  const GSHUNT = 1e-12
  const INF = Infinity
  const SQRT2 = Math.SQRT2

  const DEFAULTS = {
    D: { IS: 1e-14, N: 1.0, RS: 0.0, CJO: 0.0, VJ: 1.0, M: 0.5, FC: 0.5, TT: 0.0, BV: INF, IBV: 1e-3, NBV: 0.0 },
    NPN: { IS: 1e-16, BF: 100.0, NF: 1.0, VAF: INF, IKF: INF, ISE: 0.0, NE: 1.5, BR: 1.0, NR: 1.0, VAR: INF, IKR: INF,
      ISC: 0.0, NC: 2.0, RB: 0.0, RE: 0.0, RC: 0.0, CJE: 0.0, VJE: 0.75, MJE: 0.33, TF: 0.0, CJC: 0.0, VJC: 0.75,
      MJC: 0.33, TR: 0.0, FC: 0.5 },
    NJF: { VTO: -2.0, BETA: 1e-4, LAMBDA: 0.0, B: 1.0, IS: 1e-14, N: 1.0, RD: 0.0, RS: 0.0, CGS: 0.0, CGD: 0.0, PB: 1.0,
      M: 0.5, FC: 0.5 },
    OPAMP: { A0: 2e5, GBW: 1e6, RIN: 1e12, ROUT: 75.0, VSAT_HI: 1.5, VSAT_LO: 1.5, IQ: 1e-3 },
    NMOS: { VTO: 0.0, KP: 2e-5, GAMMA: 0.0, PHI: 0.6, LAMBDA: 0.0, RD: 0.0, RS: 0.0, CBD: 0.0, CBS: 0.0, IS: 1e-14,
      PB: 0.8, MJ: 0.5, FC: 0.5, CGSO: 0.0, CGDO: 0.0, CGBO: 0.0, CGS: 0.0, CGD: 0.0, LD: 0.0, W: 1e-4, L: 1e-4,
      MULT: 1.0 }
  }
  DEFAULTS.PNP = DEFAULTS.NPN
  DEFAULTS.PJF = DEFAULTS.NJF
  DEFAULTS.PMOS = DEFAULTS.NMOS
  const ALIASES = {
    D: { CJ0: 'CJO', CJ: 'CJO', PB: 'VJ', MJ: 'M' },
    NPN: { VA: 'VAF', VB: 'VAR', IK: 'IKF', C2: 'ISE', C4: 'ISC', PE: 'VJE', ME: 'MJE', PC: 'VJC', MC: 'MJC' },
    NJF: { VT0: 'VTO' },
    OPAMP: { AOL: 'A0', GAIN: 'A0', GB: 'GBW', RI: 'RIN', RO: 'ROUT' },
    NMOS: { VT0: 'VTO', U0: 'UO' }
  }
  ALIASES.PNP = ALIASES.NPN
  ALIASES.PJF = ALIASES.NJF
  ALIASES.PMOS = ALIASES.NMOS
  const ZERO_IS_INF = { VAF: 1, VAR: 1, IKF: 1, IKR: 1, BV: 1 }

  function modelParams(type, raw) {
    const base = Object.assign({}, DEFAULTS[type])
    const alias = ALIASES[type] || {}
    for (const key of Object.keys(raw || {})) {
      const up = key.toUpperCase()
      const k = alias[up] || up
      let v = raw[key]
      if (v === null) v = INF // the REST layer sends non-finite numbers as null
      v = Number(v)
      if (Number.isNaN(v)) continue
      if (ZERO_IS_INF[k] && v === 0) v = INF
      base[k] = v
    }
    return base
  }

  // exp continued linearly above EXP_MAX, and its derivative
  function xexp(x) {
    return x > EXP_MAX ? E_MAX * (1 + x - EXP_MAX) : Math.exp(x)
  }
  function dxexp(x, e) {
    return x > EXP_MAX ? E_MAX : e
  }
  function vcritOf(vte, isat) {
    return vte * Math.log(vte / (SQRT2 * Math.max(isat, 1e-300)))
  }
  // SPICE depletion charge/capacitance with FC linearisation -> [q, c]
  function depletion(v, cj, vj, m, fc, out) {
    const fcv = fc * vj
    if (v < fcv) {
      const arg = 1 - v / vj
      const sarg = Math.exp(-m * Math.log(arg))
      out[0] = (vj * cj * (1 - arg * sarg)) / (1 - m)
      out[1] = cj * sarg
    } else {
      const f1 = (vj * (1 - Math.pow(1 - fc, 1 - m))) / (1 - m)
      const f2 = Math.pow(1 - fc, 1 + m)
      const f3 = 1 - fc * (1 + m)
      out[0] = cj * f1 + (cj / f2) * (f3 * (v - fcv) + (m / (2 * vj)) * (v * v - fcv * fcv))
      out[1] = (cj / f2) * (f3 + (m * v) / vj)
    }
    return out
  }
  function pnjlim(vnew, vold, vt, vcrit) {
    if (vnew > vcrit && Math.abs(vnew - vold) > 2 * vt) {
      if (vold > 0) {
        const arg = 1 + (vnew - vold) / vt
        return arg > 0 ? vold + vt * Math.log(Math.max(arg, 1e-300)) : vcrit
      }
      return vt * Math.log(Math.max(vnew / vt, 1e-300))
    }
    if (vnew < 0) {
      const floor = vold > 0 ? -vold - 1 : 2 * vold - 1
      return Math.max(vnew, floor)
    }
    return vnew
  }
  function fetlim(vnew, vold, vto) {
    const vtsthi = Math.abs(2 * (vold - vto)) + 2
    const vtstlo = Math.abs(vold - vto) + 1
    const vtox = vto + 3.5
    const delv = vnew - vold
    if (vold >= vto) {
      if (vold >= vtox) {
        if (delv <= 0) {
          if (vnew >= vtox) {
            if (-delv > vtstlo) vnew = vold - vtstlo
          } else {
            vnew = Math.max(vnew, vto + 2)
          }
        } else if (delv >= vtsthi) {
          vnew = vold + vtsthi
        }
      } else if (delv <= 0) {
        vnew = Math.max(vnew, vto - 0.5)
      } else {
        vnew = Math.min(vnew, vto + 4)
      }
    } else if (delv <= 0) {
      if (-delv > vtsthi) vnew = vold - vtsthi
    } else {
      const vtemp = vto + 0.5
      if (vnew <= vtemp) {
        if (delv > vtstlo) vnew = vold + vtstlo
      } else {
        vnew = vtemp
      }
    }
    return vnew
  }

  function limvds(vnew, vold) {
    if (vold >= 3.5) {
      if (vnew > vold) vnew = Math.min(vnew, 3 * vold + 2)
      else if (vnew < 3.5) vnew = Math.max(vnew, 2)
    } else if (vnew > vold) vnew = Math.min(vnew, 4)
    else vnew = Math.max(vnew, -0.5)
    return vnew
  }
  // softplus(x) = EPS ln(1 + e^(x/EPS)) and its slope -> out[0], out[1]
  function softplus(x, out) {
    const z = x / OPAMP_CLAMP_EPS
    if (z > 0) {
      const e = Math.exp(-z)
      out[0] = x + OPAMP_CLAMP_EPS * Math.log1p(e)
      out[1] = 1 / (1 + e)
    } else {
      const e = Math.exp(z)
      out[0] = OPAMP_CLAMP_EPS * Math.log1p(e)
      out[1] = e / (1 + e)
    }
    return out
  }
  function opampClampG(a0) {
    return 1e3 + (10 * Math.max(a0, 1)) / OPAMP_RP
  }
  // level-1 MOSFET instance constants (devices.mos_geometry)
  function mosGeometry(p) {
    const m = p.MULT
    const leff = p.L - 2 * p.LD
    if (!(m > 0 && p.W > 0 && leff > 0)) throw new Error('MOSFET W, L - 2*LD and M must be positive')
    return { beta: (p.KP * m * p.W) / leff, isat: p.IS * m, cbd: p.CBD * m, cbs: p.CBS * m, rd: p.RD / m, rs: p.RS / m,
      cgs: (p.CGSO * p.W + p.CGS) * m, cgd: (p.CGDO * p.W + p.CGD) * m, cgb: p.CGBO * leff * m }
  }

  // ---------------------------------------------------------------- dense linear algebra
  // Solve A x = b in place (A n*n row-major, destroyed). Returns false when singular.
  function luSolve(A, b, n, piv) {
    for (let k = 0; k < n; k++) {
      let p = k
      let best = Math.abs(A[k * n + k])
      for (let r = k + 1; r < n; r++) {
        const a = Math.abs(A[r * n + k])
        if (a > best) {
          best = a
          p = r
        }
      }
      if (!(best > 0) || !Number.isFinite(best)) return false
      if (p !== k) {
        for (let c = 0; c < n; c++) {
          const t = A[k * n + c]
          A[k * n + c] = A[p * n + c]
          A[p * n + c] = t
        }
        const t = b[k]
        b[k] = b[p]
        b[p] = t
      }
      const d = A[k * n + k]
      for (let r = k + 1; r < n; r++) {
        const f = A[r * n + k] / d
        if (f === 0) continue
        A[r * n + k] = 0
        for (let c = k + 1; c < n; c++) A[r * n + c] -= f * A[k * n + c]
        b[r] -= f * b[k]
      }
    }
    for (let k = n - 1; k >= 0; k--) {
      let s = b[k]
      for (let c = k + 1; c < n; c++) s -= A[k * n + c] * b[c]
      b[k] = s / A[k * n + k]
    }
    return true
  }
  // Inverse by Gauss-Jordan with partial pivoting (n*n row-major) or null.
  function invert(M, n) {
    const A = Float64Array.from(M)
    const I = new Float64Array(n * n)
    for (let i = 0; i < n; i++) I[i * n + i] = 1
    for (let k = 0; k < n; k++) {
      let p = k
      let best = Math.abs(A[k * n + k])
      for (let r = k + 1; r < n; r++) {
        const a = Math.abs(A[r * n + k])
        if (a > best) {
          best = a
          p = r
        }
      }
      if (!(best > 0) || !Number.isFinite(best)) return null
      if (p !== k) {
        for (let c = 0; c < n; c++) {
          let t = A[k * n + c]
          A[k * n + c] = A[p * n + c]
          A[p * n + c] = t
          t = I[k * n + c]
          I[k * n + c] = I[p * n + c]
          I[p * n + c] = t
        }
      }
      const d = 1 / A[k * n + k]
      for (let c = 0; c < n; c++) {
        A[k * n + c] *= d
        I[k * n + c] *= d
      }
      for (let r = 0; r < n; r++) {
        if (r === k) continue
        const f = A[r * n + k]
        if (f === 0) continue
        for (let c = 0; c < n; c++) {
          A[r * n + c] -= f * A[k * n + c]
          I[r * n + c] -= f * I[k * n + c]
        }
      }
    }
    return I
  }

  function potFraction(pos, taper) {
    pos = Math.min(1, Math.max(0, Number(pos)))
    const t = String(taper || 'lin').toLowerCase()
    if (t === 'log' || t === 'a' || t === 'audio') return (Math.pow(11, pos) - 1) / 10
    if (t === 'revlog' || t === 'rev' || t === 'c') return 1 - (Math.pow(11, 1 - pos) - 1) / 10
    return pos
  }
  function potHalves(value, pos, taper) {
    const f = potFraction(pos, taper)
    return [Math.max(1, value * f), Math.max(1, value * (1 - f))]
  }

  // ---------------------------------------------------------------- network
  class Network {
    constructor(cj, opts) {
      opts = opts || {}
      if (!cj || !Array.isArray(cj.elements) || !Array.isArray(cj.nodes)) throw new Error('compiled circuit is malformed')
      this.cj = cj
      const nn = cj.nodes.length
      this.nn = nn
      const node = k => (k > 0 ? k - 1 : -1)
      // pass 1: models, internal nodes, branches
      const plans = []
      let nInt = 0
      let nBr = 0
      for (const e of cj.elements) {
        let mp = null
        let type = null
        if (e.kind === 'D') type = 'D'
        else if (e.kind === 'Q') type = String(e.type || 'npn').toUpperCase()
        else if (e.kind === 'J') type = String(e.type || 'njf').toUpperCase()
        else if (e.kind === 'X') type = 'OPAMP'
        else if (e.kind === 'M') type = String(e.type || 'nmos').toUpperCase()
        if (type) {
          if (!DEFAULTS[type]) throw new Error(`${e.name}: unsupported model type ${type}`)
          mp = modelParams(type, e.p)
        }
        let need = 0
        if (e.kind === 'D') need = mp.RS > 0 ? 1 : 0
        else if (e.kind === 'Q') need = (mp.RB > 0) + (mp.RC > 0) + (mp.RE > 0)
        else if (e.kind === 'J') need = (mp.RD > 0) + (mp.RS > 0)
        else if (e.kind === 'X') need = 1
        else if (e.kind === 'M') {
          const geo = mosGeometry(mp)
          need = (geo.rd > 0) + (geo.rs > 0)
        }
        if (e.kind === 'V' || e.kind === 'L' || e.kind === 'E') nBr++
        nInt += need
        plans.push({ e, mp, type, ns: (e.n || []).map(node) })
      }
      this.nNodes = nn + nInt
      const n = this.nNodes + nBr
      this.n = n
      this.plans = plans
      let nextInt = nn
      let nextBr = this.nNodes
      const reactive = []
      const sources = []
      this.inputSource = -1
      this.linear = [] // conductance stamps that depend on element values: {a, b, name}
      const dio = []
      const bjt = []
      const jft = []
      const opa = []
      const mos = []
      const extraG = [] // fixed entries [row, col, val]
      const newInt = () => nextInt++
      for (const pl of plans) {
        const { e, mp, ns } = pl
        const k = e.kind
        if (k === 'R') {
          if (!e.value) throw new Error(`${e.name} has zero resistance`)
          pl.g = 1 / Number(e.value)
        } else if (k === 'S') {
          pl.g = 1 / (e.on ? 0.1 : 1e9)
        } else if (k === 'C') {
          reactive.push([ns[0], ns[1], Number(e.value)])
        } else if (k === 'L') {
          pl.br = nextBr++
          reactive.push([pl.br, -2, -Number(e.value)])
        } else if (k === 'V') {
          pl.br = nextBr++
          const s = { name: e.name, dc: Number(e.dc || 0), col: [[pl.br, 1]] }
          if (e.input) this.inputSource = sources.length
          sources.push(s)
        } else if (k === 'I') {
          const col = []
          if (ns[0] >= 0) col.push([ns[0], -1])
          if (ns[1] >= 0) col.push([ns[1], 1])
          if (e.input) this.inputSource = sources.length
          sources.push({ name: e.name, dc: Number(e.dc || 0), col })
        } else if (k === 'E') {
          pl.br = nextBr++
        } else if (k === 'D') {
          let ai = ns[0]
          if (mp.RS > 0) {
            ai = newInt()
            pl.int = [[ns[0], ai, 1 / mp.RS]]
          }
          pl.ports = [ai, ns[1]]
          dio.push(pl)
        } else if (k === 'Q') {
          let [ci, bi, ei] = ns
          pl.int = []
          if (mp.RC > 0) {
            ci = newInt()
            pl.int.push([ns[0], ci, 1 / mp.RC])
          }
          if (mp.RB > 0) {
            bi = newInt()
            pl.int.push([ns[1], bi, 1 / mp.RB])
          }
          if (mp.RE > 0) {
            ei = newInt()
            pl.int.push([ns[2], ei, 1 / mp.RE])
          }
          pl.ports = [ci, bi, ei]
          pl.pol = pl.type === 'NPN' ? 1 : -1
          bjt.push(pl)
        } else if (k === 'J') {
          let [di, g, si] = ns
          pl.int = []
          if (mp.RD > 0) {
            di = newInt()
            pl.int.push([ns[0], di, 1 / mp.RD])
          }
          if (mp.RS > 0) {
            si = newInt()
            pl.int.push([ns[2], si, 1 / mp.RS])
          }
          pl.ports = [di, g, si]
          pl.pol = pl.type === 'NJF' ? 1 : -1
          jft.push(pl)
        } else if (k === 'M') {
          const [dd, g, ss, bb] = ns
          const geo = mosGeometry(mp)
          let di = dd
          let si = ss
          pl.int = []
          if (geo.rd > 0) {
            di = newInt()
            pl.int.push([dd, di, 1 / geo.rd])
          }
          if (geo.rs > 0) {
            si = newInt()
            pl.int.push([ss, si, 1 / geo.rs])
          }
          for (const [a, b, c] of [[g, si, geo.cgs], [g, di, geo.cgd], [g, bb, geo.cgb]]) {
            if (c > 0 && a !== b) reactive.push([a, b, c])
          }
          pl.ports = [di, g, si, bb]
          pl.tied = bb === si
          pl.pol = pl.type === 'NMOS' ? 1 : -1
          pl.geo = geo
          mos.push(pl)
        } else if (k === 'X') {
          const p = newInt()
          pl.p = p
          const a0 = Math.max(mp.A0, 1)
          const fp = Math.max(mp.GBW, 1) / a0
          reactive.push([p, -1, 1 / (2 * Math.PI * fp * OPAMP_RP)])
          pl.gout = 1 / Math.max(mp.ROUT, 1e-3)
          if (mp.IQ !== 0) {
            const col = []
            if (ns[2] >= 0) col.push([ns[2], -1])
            if (ns[3] >= 0) col.push([ns[3], 1])
            sources.push({ name: e.name + '#IQ', dc: mp.IQ, col })
          }
          opa.push(pl)
        } else if (k === 'G' || k === 'I') {
          // stamped below
        } else if (k !== 'G') {
          throw new Error(`${e.name}: element kind ${k} is not supported live`)
        }
      }
      this.sources = sources
      this.ns = sources.length
      if (this.inputSource < 0) throw new Error('the circuit has no input source (*@ in)')
      // reactive incidence S (nr x n) and values
      const nr = reactive.length
      this.nr = nr
      this.reactive = reactive
      this.cval = new Float64Array(nr)
      reactive.forEach((r, i) => {
        this.cval[i] = r[2]
      })
      // B (n x ns)
      const B = new Float64Array(n * this.ns)
      sources.forEach((s, j) => {
        for (const [row, val] of s.col) B[row * this.ns + j] += val
      })
      this.B = B
      // ports
      const vports = []
      const iports = []
      this.groups = []
      if (dio.length) {
        const vs = vports.length
        for (const pl of dio) vports.push(pl.ports)
        for (const pl of dio) iports.push(pl.ports)
        this.groups.push(makeDiodes(dio, vs))
      }
      if (bjt.length) {
        const vs = vports.length
        const k = bjt.length
        for (const pl of bjt) vports.push([pl.ports[1], pl.ports[2]])
        for (const pl of bjt) vports.push([pl.ports[1], pl.ports[0]])
        for (let j = 0; j < 2 * k; j++) iports.push(vports[vs + j])
        this.groups.push(makeBJTs(bjt, vs))
      }
      if (jft.length) {
        const vs = vports.length
        const k = jft.length
        for (const pl of jft) vports.push([pl.ports[1], pl.ports[2]])
        for (const pl of jft) vports.push([pl.ports[1], pl.ports[0]])
        for (let j = 0; j < 2 * k; j++) iports.push(vports[vs + j])
        this.groups.push(makeJFETs(jft, vs))
      }
      // MOSFETs: bulk-on-source devices need 2 voltage ports and 1 current port each,
      // the rest 3 and 3 (devices.MOSFETGroup); current ports are indexed separately
      for (const tied of [true, false]) {
        const sel = mos.filter(pl => pl.tied === tied)
        if (!sel.length) continue
        const vs = vports.length
        const is = iports.length
        for (const pl of sel) vports.push([pl.ports[1], pl.ports[2]])
        for (const pl of sel) vports.push([pl.ports[0], pl.ports[2]])
        for (const pl of sel) iports.push([pl.ports[0], pl.ports[2]])
        if (!tied) {
          for (const pl of sel) vports.push([pl.ports[3], pl.ports[2]])
          for (const pl of sel) iports.push([pl.ports[3], pl.ports[2]])
          for (const pl of sel) iports.push([pl.ports[3], pl.ports[0]])
        }
        this.groups.push(makeMOSFETs(sel, vs, is, tied))
      }
      if (opa.length) {
        const vs = vports.length
        const is = iports.length
        for (const pl of opa) vports.push([pl.p, -1])
        for (const pl of opa) vports.push([pl.ns[2], -1])
        for (const pl of opa) vports.push([pl.ns[3], -1])
        for (const pl of opa) iports.push([-1, pl.ns[4]])
        for (const pl of opa) iports.push([pl.p, -1]) // anti-windup clamp current out of the pole node
        this.groups.push(makeOpAmps(opa, vs, is))
      }
      this.vports = vports
      this.iports = iports
      this.mv = vports.length
      this.mi = iports.length
      this.hasCharge = this.groups.some(g => g.hasCharge)
      // structural nonzeros of the device Jacobian (shared by the charge Jacobian)
      const pat = []
      for (const g of this.groups) for (const pr of g.pattern) pat.push(pr)
      this.patQ = Int32Array.from(pat.map(pr => pr[0]))
      this.patC = Int32Array.from(pat.map(pr => pr[1]))
      this.patI = Int32Array.from(pat.map(pr => pr[0] * vports.length + pr[1]))
      this.extraG = extraG
      this.buildG()
      // scratch
      this.v = new Float64Array(this.mv)
      this.i = new Float64Array(this.mi)
      this.J = new Float64Array(this.mi * this.mv)
      this.q = new Float64Array(this.mi)
      this.C = new Float64Array(this.mi * this.mv)
      this.pots = {}
      for (const p of cj.pots || []) this.pots[String(p.name).toUpperCase()] = Object.assign({}, p)
      void opts
    }

    // dense G including every linear stamp (rebuilt when a pot moves)
    buildG() {
      const n = this.n
      const G = new Float64Array(n * n)
      const sg = (a, b, g) => {
        if (a >= 0) G[a * n + a] += g
        if (b >= 0) G[b * n + b] += g
        if (a >= 0 && b >= 0) {
          G[a * n + b] -= g
          G[b * n + a] -= g
        }
      }
      const sb = (a, b, br) => {
        if (a >= 0) {
          G[a * n + br] += 1
          G[br * n + a] += 1
        }
        if (b >= 0) {
          G[b * n + br] -= 1
          G[br * n + b] -= 1
        }
      }
      for (const pl of this.plans) {
        const { e, ns, mp } = pl
        const k = e.kind
        if (k === 'R' || k === 'S') sg(ns[0], ns[1], pl.g)
        else if (k === 'L' || k === 'V') sb(ns[0], ns[1], pl.br)
        else if (k === 'E') {
          sb(ns[0], ns[1], pl.br)
          const gain = Number(e.value)
          if (ns[2] >= 0) G[pl.br * n + ns[2]] -= gain
          if (ns[3] >= 0) G[pl.br * n + ns[3]] += gain
        } else if (k === 'G') {
          const gm = Number(e.value)
          for (const [row, s] of [[ns[0], 1], [ns[1], -1]]) {
            if (row < 0) continue
            if (ns[2] >= 0) G[row * n + ns[2]] += s * gm
            if (ns[3] >= 0) G[row * n + ns[3]] -= s * gm
          }
        } else if (k === 'D' || k === 'Q' || k === 'J' || k === 'M') {
          for (const [a, b, g] of pl.int || []) sg(a, b, g)
        } else if (k === 'X') {
          const [inp, inn, , , out] = ns
          if (mp.RIN > 0 && Number.isFinite(mp.RIN)) sg(inp, inn, 1 / mp.RIN)
          const a0 = Math.max(mp.A0, 1)
          const gm = a0 / OPAMP_RP
          if (inp >= 0) G[pl.p * n + inp] -= gm
          if (inn >= 0) G[pl.p * n + inn] += gm
          sg(pl.p, -1, 1 / OPAMP_RP)
          sg(out, -1, pl.gout)
        }
      }
      this.G = G
    }

    setResistor(name, ohms) {
      const pl = this.plans.find(p => p.e.name === name && p.e.kind === 'R')
      if (!pl) return false
      pl.g = 1 / Math.max(ohms, 1e-6)
      return true
    }

    // move a pot: both halves change, G is rebuilt; returns true when something changed
    setPot(name, pos) {
      const p = this.pots[String(name).toUpperCase()]
      if (!p) return false
      p.pos = Math.min(1, Math.max(0, Number(pos)))
      const [ra, rb] = potHalves(Number(p.value), p.pos, p.taper)
      const halves = p.halves || [p.name + '_A', p.name + '_B']
      const a = this.setResistor(halves[0], ra)
      const b = this.setResistor(halves[1], rb)
      if (a || b) this.buildG()
      return a || b
    }

    portV(x, out) {
      for (let r = 0; r < this.mv; r++) {
        const [a, b] = this.vports[r]
        out[r] = (a >= 0 ? x[a] : 0) - (b >= 0 ? x[b] : 0)
      }
      return out
    }

    // device currents and dense Jacobian (mi x mv) at port voltages v
    evalDevices(v) {
      this.J.fill(0)
      for (const g of this.groups) g.evaluate(v, this.i, this.J, this.mv)
    }
    chargeDevices(v) {
      this.q.fill(0)
      this.C.fill(0)
      for (const g of this.groups) if (g.hasCharge) g.charges(v, this.q, this.C, this.mv)
    }
    limit(vnew, vold) {
      for (const g of this.groups) g.limit(vnew, vold)
    }
    initialPorts(out) {
      out.fill(0)
      for (const g of this.groups) g.initial(out)
      return out
    }

    u(inputValue) {
      const u = new Float64Array(this.ns)
      this.sources.forEach((s, j) => {
        u[j] = s.dc
      })
      if (inputValue !== undefined) u[this.inputSource] = this.sources[this.inputSource].dc + inputValue
      return u
    }

    // ---- DC Newton on the full MNA system (mna.Network.newton)
    newton(x0, u, gextra, maxiter, initJunctions) {
      const n = this.n
      const mv = this.mv
      const mi = this.mi
      const reltol = 1e-6
      const vntol = 1e-9
      const abstol = 1e-14
      const b = new Float64Array(n)
      for (let r = 0; r < n; r++) {
        let s = 0
        for (let j = 0; j < this.ns; j++) s += this.B[r * this.ns + j] * u[j]
        b[r] = s
      }
      const Gb = Float64Array.from(this.G)
      for (let r = 0; r < this.nNodes; r++) Gb[r * n + r] += GSHUNT + gextra
      let x = Float64Array.from(x0)
      let vOld = new Float64Array(mv)
      if (initJunctions) this.initialPorts(vOld)
      else this.portV(x, vOld)
      const vRaw = new Float64Array(mv)
      const M = new Float64Array(n * n)
      const rhs = new Float64Array(n)
      const piv = null
      for (let it = 1; it <= maxiter; it++) {
        let limited = false
        M.set(Gb)
        rhs.set(b)
        if (mv) {
          let v
          if (it === 1 && initJunctions) v = Float64Array.from(vOld)
          else {
            this.portV(x, vRaw)
            v = Float64Array.from(vRaw)
            this.limit(v, vOld)
            for (let r = 0; r < mv; r++) {
              if (Math.abs(v[r] - vRaw[r]) > 1e-12 * (1 + Math.abs(vRaw[r]))) {
                limited = true
                break
              }
            }
          }
          this.evalDevices(v)
          const J = this.J
          // ft = i - J v ; M += Ni^T J Nv ; rhs -= Ni^T ft
          for (let ip = 0; ip < mi; ip++) {
            let ft = this.i[ip]
            for (let c = 0; c < mv; c++) ft -= J[ip * mv + c] * v[c]
            const [ia, ib] = this.iports[ip]
            if (ia >= 0) rhs[ia] -= ft
            if (ib >= 0) rhs[ib] += ft
            for (let c = 0; c < mv; c++) {
              const jv = J[ip * mv + c]
              if (jv === 0) continue
              const [va, vb] = this.vports[c]
              if (ia >= 0) {
                if (va >= 0) M[ia * n + va] += jv
                if (vb >= 0) M[ia * n + vb] -= jv
              }
              if (ib >= 0) {
                if (va >= 0) M[ib * n + va] -= jv
                if (vb >= 0) M[ib * n + vb] += jv
              }
            }
          }
          vOld = v
        }
        if (!luSolve(M, rhs, n, piv)) return [x, false, it]
        for (let r = 0; r < n; r++) if (!Number.isFinite(rhs[r])) return [x, false, it]
        let ok = true
        for (let r = 0; r < n; r++) {
          const tol = r < this.nNodes ? vntol : abstol
          if (Math.abs(rhs[r] - x[r]) > reltol * Math.max(Math.abs(rhs[r]), Math.abs(x[r])) + tol) {
            ok = false
            break
          }
        }
        x = Float64Array.from(rhs)
        if (!mv) return [x, true, it]
        if (ok && !limited && it > 1) return [x, true, it]
      }
      return [x, false, maxiter]
    }

    solveDC(u, maxiter) {
      maxiter = maxiter || 150
      const n = this.n
      let [x, ok] = this.newton(new Float64Array(n), u, 0, maxiter, true)
      if (ok) return { x, ok: true, method: 'newton' }
      x = new Float64Array(n)
      let g = 1e-2
      let good = null
      let factor = 10
      let first = true
      while (g >= 1e-12) {
        const r = this.newton(x, u, g, maxiter, first)
        first = false
        if (r[1]) {
          x = r[0]
          good = g
          g /= factor
        } else {
          if (good === null) break
          factor = Math.sqrt(factor)
          if (factor < 1.05) break
          g = good / factor
        }
      }
      if (good !== null) {
        const r = this.newton(x, u, 0, maxiter, false)
        if (r[1]) return { x: r[0], ok: true, method: 'gmin' }
      }
      x = new Float64Array(n)
      let alpha = 0
      let step = 0.1
      let okAny = true
      while (alpha < 1) {
        const aTry = Math.min(1, alpha + step)
        const ua = u.map(val => val * aTry)
        const r = this.newton(x, ua, 0, maxiter, alpha === 0)
        if (r[1]) {
          x = r[0]
          alpha = aTry
          step = Math.min(step * 1.5, 0.5)
        } else {
          step /= 4
          if (step < 1e-5) {
            okAny = false
            break
          }
        }
      }
      if (okAny && alpha >= 1) return { x, ok: true, method: 'source' }
      return { x, ok: false, method: 'failed' }
    }
  }

  // ---------------------------------------------------------------- device groups
  function makeDiodes(list, vs) {
    const d = list.map(pl => {
      const p = pl.mp
      const vte = p.N * VT
      const o = { IS: p.IS, vte, vcrit: vcritOf(vte, p.IS), TT: p.TT, CJO: p.CJO, VJ: p.VJ, M: Math.min(p.M, 0.9),
        FC: Math.min(p.FC, 0.95), bv: Number.isFinite(p.BV) }
      if (o.bv) {
        o.vtbv = (p.NBV > 0 ? p.NBV : p.N) * VT
        o.tbv = p.BV - o.vtbv * Math.log1p(p.IBV / p.IS)
      }
      return o
    })
    const tmp = [0, 0]
    return {
      pattern: d.map((o, j) => [vs + j, vs + j]),
      hasCharge: d.some(o => o.CJO > 0 || o.TT > 0),
      evaluate(v, I, J, mv) {
        for (let j = 0; j < d.length; j++) {
          const o = d[j]
          const r = vs + j
          const x = v[r]
          let cd
          let gd
          if (o.bv && x < -o.tbv) {
            const xb = -(o.tbv + x) / o.vtbv
            const eb = xexp(xb)
            cd = -o.IS * eb + GMIN * x
            gd = (o.IS * dxexp(xb, eb)) / o.vtbv + GMIN
          } else {
            const xx = x / o.vte
            const e = xexp(xx)
            cd = o.IS * (e - 1) + GMIN * x
            gd = (o.IS * dxexp(xx, e)) / o.vte + GMIN
          }
          I[r] = cd
          J[r * mv + r] = gd
        }
      },
      charges(v, Q, C, mv) {
        for (let j = 0; j < d.length; j++) {
          const o = d[j]
          const r = vs + j
          const x = v[r]
          const xx = x / o.vte
          const e = xexp(xx)
          const cd = o.IS * (e - 1)
          const gd = (o.IS * dxexp(xx, e)) / o.vte
          depletion(x, o.CJO, o.VJ, o.M, o.FC, tmp)
          Q[r] = o.TT * cd + tmp[0]
          C[r * mv + r] = o.TT * gd + tmp[1]
        }
      },
      limit(vnew, vold) {
        for (let j = 0; j < d.length; j++) {
          const r = vs + j
          vnew[r] = pnjlim(vnew[r], vold[r], d[j].vte, d[j].vcrit)
        }
      },
      initial(out) {
        for (let j = 0; j < d.length; j++) out[vs + j] = d[j].vcrit
      }
    }
  }

  function makeBJTs(list, vs) {
    const k = list.length
    const d = list.map(pl => {
      const p = pl.mp
      const vtf = p.NF * VT
      const vtr = p.NR * VT
      return { pol: pl.pol, IS: p.IS, BF: p.BF, BR: p.BR, vtf, vtr, ivaf: 1 / p.VAF, ivar: 1 / p.VAR, iikf: 1 / p.IKF,
        iikr: 1 / p.IKR, ISE: p.ISE, ISC: p.ISC, vte: p.NE * VT, vtc: p.NC * VT, vcritBe: vcritOf(vtf, p.IS),
        vcritBc: vcritOf(vtr, p.IS), CJE: p.CJE, VJE: p.VJE, MJE: Math.min(p.MJE, 0.9), CJC: p.CJC, VJC: p.VJC,
        MJC: Math.min(p.MJC, 0.9), TF: p.TF, TR: p.TR, FC: Math.min(p.FC, 0.95) }
    })
    const hasIk = d.some(o => o.iikf > 0 || o.iikr > 0)
    const hasLeak = d.some(o => o.ISE > 0 || o.ISC > 0)
    const c = {}
    function core(o, vbeP, vbcP) {
      const vbe = o.pol * vbeP
      const vbc = o.pol * vbcP
      const xe = vbe / o.vtf
      const ee = xexp(xe)
      const cbe = o.IS * (ee - 1)
      const gbe = (o.IS * dxexp(xe, ee)) / o.vtf
      const xc = vbc / o.vtr
      const ec = xexp(xc)
      const cbc = o.IS * (ec - 1)
      const gbc = (o.IS * dxexp(xc, ec)) / o.vtr
      let cben
      let gben
      let cbcn
      let gbcn
      if (hasLeak) {
        const xle = vbe / o.vte
        const xlc = vbc / o.vtc
        const ele = xexp(xle)
        const elc = xexp(xlc)
        cben = o.ISE * (ele - 1) + GMIN * vbe
        gben = (o.ISE * dxexp(xle, ele)) / o.vte + GMIN
        cbcn = o.ISC * (elc - 1) + GMIN * vbc
        gbcn = (o.ISC * dxexp(xlc, elc)) / o.vtc + GMIN
      } else {
        cben = GMIN * vbe
        gben = GMIN
        cbcn = GMIN * vbc
        gbcn = GMIN
      }
      const q1 = 1 / (1 - o.ivaf * vbc - o.ivar * vbe)
      let qb
      let dqbdve
      let dqbdvc
      if (hasIk) {
        const q2 = o.iikf * cbe + o.iikr * cbc
        const sq = Math.sqrt(Math.max(1 + 4 * q2, 1e-30))
        qb = (q1 * (1 + sq)) / 2
        dqbdve = q1 * (qb * o.ivar + (o.iikf * gbe) / sq)
        dqbdvc = q1 * (qb * o.ivaf + (o.iikr * gbc) / sq)
      } else {
        qb = q1
        dqbdve = q1 * qb * o.ivar
        dqbdvc = q1 * qb * o.ivaf
      }
      c.vbe = vbe
      c.vbc = vbc
      c.cbe = cbe
      c.gbe = gbe
      c.cbc = cbc
      c.gbc = gbc
      c.cben = cben
      c.gben = gben
      c.cbcn = cbcn
      c.gbcn = gbcn
      c.qb = qb
      c.dqbdve = dqbdve
      c.dqbdvc = dqbdvc
      return c
    }
    const t1 = [0, 0]
    const t2 = [0, 0]
    return {
      pattern: [].concat(...Array.from({ length: k }, (_, j) => [[vs + j, vs + j], [vs + j, vs + k + j], [vs + k + j, vs + j], [vs + k + j, vs + k + j]])),
      hasCharge: d.some(o => o.CJE > 0 || o.CJC > 0 || o.TF > 0 || o.TR > 0),
      evaluate(v, I, J, mv) {
        for (let j = 0; j < k; j++) {
          const o = d[j]
          const re = vs + j
          const rc = vs + k + j
          const s = core(o, v[re], v[rc])
          const it = (s.cbe - s.cbc) / s.qb
          const ic = it - s.cbc / o.BR - s.cbcn
          const ib = s.cbe / o.BF + s.cben + s.cbc / o.BR + s.cbcn
          const dicE = (s.gbe - it * s.dqbdve) / s.qb
          const dicC = (-s.gbc - it * s.dqbdvc) / s.qb - s.gbc / o.BR - s.gbcn
          const dibE = s.gbe / o.BF + s.gben
          const dibC = s.gbc / o.BR + s.gbcn
          I[re] = o.pol * (ic + ib)
          I[rc] = -o.pol * ic
          J[re * mv + re] = dicE + dibE
          J[re * mv + rc] = dicC + dibC
          J[rc * mv + re] = -dicE
          J[rc * mv + rc] = -dicC
        }
      },
      charges(v, Q, C, mv) {
        for (let j = 0; j < k; j++) {
          const o = d[j]
          const re = vs + j
          const rc = vs + k + j
          const s = core(o, v[re], v[rc])
          const fwd = s.vbe > 0
          const cbeM = fwd ? s.cbe / s.qb : s.cbe
          const gbeM = fwd ? (s.gbe - cbeM * s.dqbdve) / s.qb : s.gbe
          const geqcb = fwd ? (-o.TF * cbeM * s.dqbdvc) / s.qb : 0
          depletion(s.vbe, o.CJE, o.VJE, o.MJE, o.FC, t1)
          depletion(s.vbc, o.CJC, o.VJC, o.MJC, o.FC, t2)
          Q[re] = o.pol * (o.TF * cbeM + t1[0])
          Q[rc] = o.pol * (o.TR * s.cbc + t2[0])
          C[re * mv + re] = o.TF * gbeM + t1[1]
          C[re * mv + rc] = geqcb
          C[rc * mv + re] = 0
          C[rc * mv + rc] = o.TR * s.gbc + t2[1]
        }
      },
      limit(vnew, vold) {
        for (let j = 0; j < k; j++) {
          const o = d[j]
          const re = vs + j
          const rc = vs + k + j
          vnew[re] = o.pol * pnjlim(o.pol * vnew[re], o.pol * vold[re], o.vtf, o.vcritBe)
          vnew[rc] = o.pol * pnjlim(o.pol * vnew[rc], o.pol * vold[rc], o.vtr, o.vcritBc)
        }
      },
      initial(out) {
        for (let j = 0; j < k; j++) {
          out[vs + j] = d[j].pol * d[j].vcritBe
          out[vs + k + j] = 0
        }
      }
    }
  }

  function makeJFETs(list, vs) {
    const k = list.length
    const d = list.map(pl => {
      const p = pl.mp
      const vtn = p.N * VT
      return { pol: pl.pol, VTO: p.VTO, BETA: p.BETA, LAMBDA: p.LAMBDA, B: p.B, bfac: (1 - p.B) / (p.PB - p.VTO),
        IS: p.IS, vtn, vcrit: vcritOf(vtn, p.IS), CGS: p.CGS, CGD: p.CGD, PB: p.PB, M: Math.min(p.M, 0.9),
        FC: Math.min(p.FC, 0.95) }
    })
    const ch = [0, 0, 0]
    function channel(o, vgst, vds) {
      if (!(vgst > 0)) {
        ch[0] = 0
        ch[1] = 0
        ch[2] = 0
        return ch
      }
      const beta = o.BETA
      const lam = o.LAMBDA
      const b = o.B
      const betap = beta * (1 + lam * vds)
      if (vgst >= vds) {
        const apart = 2 * b + 3 * o.bfac * (vgst - vds)
        const cpart = vds * (vds * (o.bfac * vds - b) + vgst * apart)
        ch[0] = betap * cpart
        ch[1] = betap * vds * (apart + 3 * o.bfac * vgst)
        ch[2] = betap * (vgst - vds) * apart + beta * lam * cpart
      } else {
        const bf = vgst * o.bfac
        const cpart = vgst * vgst * (b + bf)
        ch[0] = betap * cpart
        ch[1] = betap * vgst * (2 * b + 3 * bf)
        ch[2] = lam * beta * cpart
      }
      return ch
    }
    const ids = [0, 0, 0]
    function idsOf(o, vgs, vgd) {
      const vds = vgs - vgd
      const normal = vds >= 0
      const vctl = (normal ? vgs : vgd) - o.VTO
      const r = channel(o, vctl, Math.abs(vds))
      if (normal) {
        ids[0] = r[0]
        ids[1] = r[1] + r[2]
        ids[2] = -r[2]
      } else {
        ids[0] = -r[0]
        ids[1] = r[2]
        ids[2] = -(r[1] + r[2])
      }
      return ids
    }
    const t1 = [0, 0]
    const t2 = [0, 0]
    return {
      pattern: [].concat(...Array.from({ length: k }, (_, j) => [[vs + j, vs + j], [vs + j, vs + k + j], [vs + k + j, vs + j], [vs + k + j, vs + k + j]])),
      hasCharge: d.some(o => o.CGS > 0 || o.CGD > 0),
      evaluate(v, I, J, mv) {
        for (let j = 0; j < k; j++) {
          const o = d[j]
          const rs = vs + j
          const rd = vs + k + j
          const vgs = o.pol * v[rs]
          const vgd = o.pol * v[rd]
          const xs = vgs / o.vtn
          const xd = vgd / o.vtn
          const es = xexp(xs)
          const ed = xexp(xd)
          const igs = o.IS * (es - 1) + GMIN * vgs
          const ggs = (o.IS * dxexp(xs, es)) / o.vtn + GMIN
          const igd = o.IS * (ed - 1) + GMIN * vgd
          const ggd = (o.IS * dxexp(xd, ed)) / o.vtn + GMIN
          const r = idsOf(o, vgs, vgd)
          I[rs] = o.pol * (igs + r[0])
          I[rd] = o.pol * (igd - r[0])
          J[rs * mv + rs] = ggs + r[1]
          J[rs * mv + rd] = r[2]
          J[rd * mv + rs] = -r[1]
          J[rd * mv + rd] = ggd - r[2]
        }
      },
      charges(v, Q, C, mv) {
        for (let j = 0; j < k; j++) {
          const o = d[j]
          const rs = vs + j
          const rd = vs + k + j
          depletion(o.pol * v[rs], o.CGS, o.PB, o.M, o.FC, t1)
          depletion(o.pol * v[rd], o.CGD, o.PB, o.M, o.FC, t2)
          Q[rs] = o.pol * t1[0]
          Q[rd] = o.pol * t2[0]
          C[rs * mv + rs] = t1[1]
          C[rd * mv + rd] = t2[1]
        }
      },
      limit(vnew, vold) {
        for (let j = 0; j < 2 * k; j++) {
          const o = d[j % k]
          const r = vs + j
          const vn = pnjlim(o.pol * vnew[r], o.pol * vold[r], o.vtn, o.vcrit)
          vnew[r] = o.pol * fetlim(vn, o.pol * vold[r], o.VTO)
        }
      },
      initial(out) {
        for (let j = 0; j < k; j++) {
          out[vs + j] = -d[j].pol
          out[vs + k + j] = -d[j].pol
        }
      }
    }
  }

  function makeOpAmps(list, vs, is) {
    const k = list.length
    const d = list.map(pl => ({ VSH: pl.mp.VSAT_HI, VSL: pl.mp.VSAT_LO, GOUT: 1 / Math.max(pl.mp.ROUT, 1e-3),
      GC: opampClampG(pl.mp.A0) }))
    const st = { mid: 0, half: 0, halfRaw: 0, u: 0, t: 0 }
    const sh = [0, 0]
    const sl2 = [0, 0]
    function vo(o, vp, vcc, vee) {
      const hi = vcc - o.VSH
      const lo = vee + o.VSL
      st.mid = 0.5 * (hi + lo)
      st.halfRaw = 0.5 * (hi - lo)
      st.half = Math.max(st.halfRaw, OPAMP_HALF_MIN)
      st.u = (vp - st.mid) / st.half
      st.t = Math.tanh(st.u)
      return st
    }
    const pattern = []
    for (const base of [is, is + k]) {
      for (let j = 0; j < k; j++) pattern.push([base + j, vs + j], [base + j, vs + k + j], [base + j, vs + 2 * k + j])
    }
    return {
      pattern,
      hasCharge: false,
      evaluate(v, I, J, mv) {
        for (let j = 0; j < k; j++) {
          const o = d[j]
          const rp = vs + j
          const rcc = vs + k + j
          const ree = vs + 2 * k + j
          const s = vo(o, v[rp], v[rcc], v[ree])
          const sl = 1 - s.t * s.t
          const out = s.mid + s.half * s.t
          const dmid = 1 - sl
          const free = s.halfRaw > OPAMP_HALF_MIN
          const dhalf = free ? s.t - sl * s.u : 0
          const g = o.GOUT
          const ri = is + j
          I[ri] = out * g
          J[ri * mv + rp] = sl * g
          J[ri * mv + rcc] = (0.5 * dmid + 0.5 * dhalf) * g
          J[ri * mv + ree] = (0.5 * dmid - 0.5 * dhalf) * g
          // anti-windup clamp: GC (softplus(vp - mid - K half) - softplus(mid - K half - vp))
          const gc = o.GC
          softplus(v[rp] - s.mid - OPAMP_CLAMP_K * s.half, sh)
          softplus(s.mid - OPAMP_CLAMP_K * s.half - v[rp], sl2)
          const rk = is + k + j
          const kd = free ? 0.5 * OPAMP_CLAMP_K : 0
          I[rk] = gc * (sh[0] - sl2[0])
          J[rk * mv + rp] = gc * (sh[1] + sl2[1])
          J[rk * mv + rcc] = -gc * (sh[1] * (0.5 + kd) + sl2[1] * (0.5 - kd))
          J[rk * mv + ree] = -gc * (sh[1] * (0.5 - kd) + sl2[1] * (0.5 + kd))
        }
      },
      charges() {},
      limit(vnew, vold) {
        for (let j = 0; j < k; j++) {
          const o = d[j]
          const rp = vs + j
          const rcc = vs + k + j
          const ree = vs + 2 * k + j
          const uOld = vo(o, vold[rp], vold[rcc], vold[ree]).u
          const s = vo(o, vnew[rp], vnew[rcc], vnew[ree])
          const uNew = s.u
          if (Math.sign(uNew) !== Math.sign(uOld) && Math.abs(uNew) > 1 && Math.abs(uOld) > 1) {
            vnew[rp] = s.mid + Math.sign(uNew) * 0.5 * s.half
          }
        }
      },
      initial() {}
    }
  }

  // Level-1 MOSFETs (devices.MOSFETGroup). Device-polarity channel current, ngspice mos1load.
  const mc = [0, 0, 0, 0, 0, 0]
  function mosCore(vgs, vds, vbs, o) {
    let vb, vg, vd
    if (vds >= 0) {
      vb = vbs
      vg = vgs
      vd = vds
    } else {
      vb = vbs - vds
      vg = vgs - vds
      vd = -vds
    }
    let von
    let arg = 0
    if (o.GAMMA) {
      let sarg
      if (vb <= 0) sarg = Math.sqrt(o.PHI - vb)
      else {
        sarg = o.sqphi - vb / (o.sqphi + o.sqphi)
        if (sarg < 0) sarg = 0
      }
      von = o.vbi + o.GAMMA * sarg
      arg = sarg > 0 ? o.GAMMA / (sarg + sarg) : 0
    } else von = o.vbi
    const vgst = vg - von
    let cd = 0
    let gm = 0
    let gds = 0
    if (vgst > 0) {
      const betap = o.beta * (1 + o.LAMBDA * vd)
      if (vgst <= vd) {
        cd = 0.5 * betap * vgst * vgst
        gm = betap * vgst
        gds = 0.5 * o.LAMBDA * o.beta * vgst * vgst
      } else {
        cd = betap * vd * (vgst - 0.5 * vd)
        gm = betap * vd
        gds = betap * (vgst - vd) + o.LAMBDA * o.beta * vd * (vgst - 0.5 * vd)
      }
    }
    const gmbs = gm * arg
    if (vds >= 0) {
      mc[0] = cd
      mc[1] = gm
      mc[2] = gds
      mc[3] = gmbs
    } else {
      mc[0] = -cd
      mc[1] = -gm
      mc[2] = gm + gds + gmbs
      mc[3] = -gmbs
    }
    mc[4] = vgst
    mc[5] = vd
    return mc
  }
  // bulk junction diode (N = 1) -> out[0] current, out[1] conductance
  function mosBulk(v, isat, out) {
    if (v <= VT3) {
      out[0] = GMIN * v - isat
      out[1] = GMIN
    } else {
      const x = v / VT
      const e = xexp(x)
      out[0] = isat * (e - 1) + GMIN * v
      out[1] = (isat * dxexp(x, e)) / VT + GMIN
    }
    return out
  }

  function makeMOSFETs(list, vs, is, tied) {
    const k = list.length
    const d = list.map(pl => {
      const p = pl.mp
      const g = pl.geo
      const sqphi = p.PHI > 0 ? Math.sqrt(p.PHI) : 0
      return { pol: pl.pol, beta: g.beta, LAMBDA: p.LAMBDA, GAMMA: p.GAMMA, PHI: p.PHI, sqphi,
        vbi: pl.pol * p.VTO - p.GAMMA * sqphi, VTO: p.VTO, isat: g.isat,
        vcrit: VT * Math.log(VT / (SQRT2 * Math.max(g.isat, 1e-300))), cbd: g.cbd, cbs: g.cbs, PB: p.PB,
        MJ: Math.min(p.MJ, 0.9), FC: Math.min(p.FC, 0.95) }
    })
    const pattern = []
    for (let j = 0; j < k; j++) pattern.push([is + j, vs + j], [is + j, vs + k + j])
    if (!tied) {
      for (let j = 0; j < k; j++) {
        pattern.push([is + j, vs + 2 * k + j], [is + k + j, vs + 2 * k + j], [is + 2 * k + j, vs + 2 * k + j],
          [is + 2 * k + j, vs + k + j])
      }
    }
    const b1 = [0, 0]
    const b2 = [0, 0]
    const t1 = [0, 0]
    const t2 = [0, 0]
    return {
      pattern,
      hasCharge: d.some(o => o.cbd > 0) || (!tied && d.some(o => o.cbs > 0)),
      evaluate(v, I, J, mv) {
        for (let j = 0; j < k; j++) {
          const o = d[j]
          const tp = o.pol
          const rg = vs + j
          const rd = vs + k + j
          const ri = is + j
          const vgs = tp * v[rg]
          const vds = tp * v[rd]
          if (tied) {
            const r = mosCore(vgs, vds, 0, o)
            mosBulk(-vds, o.isat, b1)
            I[ri] = tp * (r[0] - b1[0])
            J[ri * mv + rg] = r[1]
            J[ri * mv + rd] = r[2] + b1[1]
          } else {
            const rb = vs + 2 * k + j
            const vbs = tp * v[rb]
            const r = mosCore(vgs, vds, vbs, o)
            mosBulk(vbs, o.isat, b1)
            mosBulk(vbs - vds, o.isat, b2)
            const rs = is + k + j
            const rdd = is + 2 * k + j
            I[ri] = tp * r[0]
            I[rs] = tp * b1[0]
            I[rdd] = tp * b2[0]
            J[ri * mv + rg] = r[1]
            J[ri * mv + rd] = r[2]
            J[ri * mv + rb] = r[3]
            J[rs * mv + rb] = b1[1]
            J[rdd * mv + rb] = b2[1]
            J[rdd * mv + rd] = -b2[1]
          }
        }
      },
      charges(v, Q, C, mv) {
        for (let j = 0; j < k; j++) {
          const o = d[j]
          const tp = o.pol
          const rd = vs + k + j
          const vds = tp * v[rd]
          if (tied) {
            const ri = is + j
            if (o.cbd > 0) depletion(-vds, o.cbd, o.PB, o.MJ, o.FC, t2)
            else t2[0] = t2[1] = 0
            Q[ri] = -tp * t2[0]
            C[ri * mv + rd] = t2[1]
          } else {
            const rb = vs + 2 * k + j
            const vbs = tp * v[rb]
            if (o.cbs > 0) depletion(vbs, o.cbs, o.PB, o.MJ, o.FC, t1)
            else t1[0] = t1[1] = 0
            if (o.cbd > 0) depletion(vbs - vds, o.cbd, o.PB, o.MJ, o.FC, t2)
            else t2[0] = t2[1] = 0
            const rs = is + k + j
            const rdd = is + 2 * k + j
            Q[rs] = tp * t1[0]
            Q[rdd] = tp * t2[0]
            C[rs * mv + rb] = t1[1]
            C[rdd * mv + rb] = t2[1]
            C[rdd * mv + rd] = -t2[1]
          }
        }
      },
      // ngspice mos1load: fetlim on vgs (or vgd), limvds, pnjlim on the bulk junction
      limit(vnew, vold) {
        for (let j = 0; j < k; j++) {
          const o = d[j]
          const tp = o.pol
          const rg = vs + j
          const rd = vs + k + j
          let vgsN = tp * vnew[rg]
          let vdsN = tp * vnew[rd]
          const vgsO = tp * vold[rg]
          const vdsO = tp * vold[rd]
          let vbsN = 0
          let vbsO = 0
          const rb = vs + 2 * k + j
          if (!tied) {
            vbsN = tp * vnew[rb]
            vbsO = tp * vold[rb]
          }
          const vb = vdsO >= 0 ? vbsO : vbsO - vdsO
          let von = o.vbi
          if (o.GAMMA) {
            const sarg = vb <= 0 ? Math.sqrt(o.PHI - vb) : Math.max(o.sqphi - vb / (o.sqphi + o.sqphi), 0)
            von = o.vbi + o.GAMMA * sarg
          }
          let vgdN = vgsN - vdsN
          if (vdsO >= 0) {
            vgsN = fetlim(vgsN, vgsO, von)
            vdsN = limvds(vgsN - vgdN, vdsO)
          } else {
            vgdN = fetlim(vgdN, vgsO - vdsO, von)
            vdsN = -limvds(-(vgsN - vgdN), -vdsO)
            vgsN = vgdN + vdsN
          }
          vnew[rg] = tp * vgsN
          vnew[rd] = tp * vdsN
          if (!tied) {
            if (vdsN >= 0) vbsN = pnjlim(vbsN, vbsO, VT, o.vcrit)
            else vbsN = pnjlim(vbsN - vdsN, vbsO - vdsO, VT, o.vcrit) + vdsN
            vnew[rb] = tp * vbsN
          }
        }
      },
      initial(out) {
        for (let j = 0; j < k; j++) {
          out[vs + j] = d[j].VTO
          out[vs + k + j] = 0
          if (!tied) out[vs + 2 * k + j] = -d[j].pol
        }
      }
    }
  }

  // ---------------------------------------------------------------- transient (mna.Transient)
  // Matrices of the reduced system for one step size h (mna._Level).
  function buildLevel(net, h, J0, Cq0, outRow) {
    const n = net.n
    const mv = net.mv
    const mi = net.mi
    const nr = net.nr
    const ns = net.ns
    const a = 2 / h
    const Jlin = new Float64Array(mi * mv)
    for (let q = 0; q < mi * mv; q++) Jlin[q] = J0[q] + a * Cq0[q]
    const A = Float64Array.from(net.G)
    for (let r = 0; r < net.nNodes; r++) A[r * n + r] += GSHUNT
    // + a * Cm  (Cm = S^T diag(c) S)
    net.reactive.forEach(([ra, rb, val]) => {
      const cv = a * val
      if (rb === -2) {
        A[ra * n + ra] += cv
        return
      }
      if (ra >= 0) A[ra * n + ra] += cv
      if (rb >= 0) A[rb * n + rb] += cv
      if (ra >= 0 && rb >= 0) {
        A[ra * n + rb] -= cv
        A[rb * n + ra] -= cv
      }
    })
    // + Ni^T Jlin Nv
    for (let ip = 0; ip < mi; ip++) {
      const [ia, ib] = net.iports[ip]
      for (let c = 0; c < mv; c++) {
        const jv = Jlin[ip * mv + c]
        if (jv === 0) continue
        const [va, vb] = net.vports[c]
        if (ia >= 0) {
          if (va >= 0) A[ia * n + va] += jv
          if (vb >= 0) A[ia * n + vb] -= jv
        }
        if (ib >= 0) {
          if (va >= 0) A[ib * n + va] -= jv
          if (vb >= 0) A[ib * n + vb] += jv
        }
      }
    }
    const Ainv = invert(A, n)
    if (!Ainv) throw new Error('the transient matrix is singular (a loop of voltage sources or a floating part)')
    // columns: Mu = Ainv B (n x ns), Mh = Ainv S^T (n x nr), Mi = Ainv Ni^T (n x mi)
    const Mu = new Float64Array(n * ns)
    for (let r = 0; r < n; r++) {
      for (let j = 0; j < ns; j++) {
        let s = 0
        for (let c = 0; c < n; c++) s += Ainv[r * n + c] * net.B[c * ns + j]
        Mu[r * ns + j] = s
      }
    }
    const colOf = (pair, r) => {
      // Ainv[r] . (e_a - e_b)
      const [p, q] = pair
      return (p >= 0 ? Ainv[r * n + p] : 0) - (q >= 0 ? Ainv[r * n + q] : 0)
    }
    const Mh = new Float64Array(n * nr)
    for (let r = 0; r < n; r++) {
      for (let j = 0; j < nr; j++) {
        const [ra, rb] = net.reactive[j]
        Mh[r * nr + j] = rb === -2 ? Ainv[r * n + ra] : colOf([ra, rb], r)
      }
    }
    const Mi = new Float64Array(n * mi)
    for (let r = 0; r < n; r++) for (let j = 0; j < mi; j++) Mi[r * mi + j] = colOf(net.iports[j], r)
    // row projections: Nv (ports), S (reactive), out row
    const proj = (M, w, rowPair) => {
      const out = new Float64Array(w)
      const [p, q] = rowPair
      for (let j = 0; j < w; j++) out[j] = (p >= 0 ? M[p * w + j] : 0) - (q >= 0 ? M[q * w + j] : 0)
      return out
    }
    const stack = (M, w, pairs) => {
      const out = new Float64Array(pairs.length * w)
      pairs.forEach((pr, r) => out.set(proj(M, w, pr), r * w))
      return out
    }
    const rpairs = net.reactive.map(([ra, rb]) => (rb === -2 ? [ra, -1] : [ra, rb]))
    const lv = {
      h, a, Jlin,
      Pu: stack(Mu, ns, net.vports), Ph: stack(Mh, nr, net.vports), K: stack(Mi, mi, net.vports),
      Ru: stack(Mu, ns, rpairs), Rh: stack(Mh, nr, rpairs), Ri: stack(Mi, mi, rpairs),
      Ou: proj(Mu, ns, [outRow, -1]), Oh: proj(Mh, nr, [outRow, -1]), Oi: proj(Mi, mi, [outRow, -1]),
      ac: new Float64Array(nr)
    }
    for (let j = 0; j < nr; j++) lv.ac[j] = a * net.cval[j]
    return lv
  }

  class Transient {
    constructor(net, xop, outRow, opts) {
      opts = opts || {}
      this.net = net
      this.outRow = outRow
      this.vtol = opts.vtol || 1e-6
      this.reltol = opts.reltol || 1e-6
      this.maxiter = opts.maxiter || 30
      this.maxHalvings = opts.maxHalvings === undefined ? 8 : opts.maxHalvings
      this.predict = !!opts.predict
      const mv = net.mv
      const mi = net.mi
      this.v = new Float64Array(mv)
      net.portV(xop, this.v)
      this.q = new Float64Array(mi)
      this.iq = new Float64Array(mi)
      this.J0 = new Float64Array(mi * mv)
      this.Cq0 = new Float64Array(mi * mv)
      this.linearise(this.v)
      if (net.hasCharge && mv) this.q.set(net.q)
      const nr = net.nr
      this.y = new Float64Array(nr)
      net.reactive.forEach(([ra, rb], r) => {
        this.y[r] = rb === -2 ? xop[ra] : (ra >= 0 ? xop[ra] : 0) - (rb >= 0 ? xop[rb] : 0)
      })
      this.z = new Float64Array(nr)
      this.levels = []
      this.failedSteps = 0
      this.substeps = 0
      this.iterations = 0
      this.chargeGroups = net.groups.some(g => g.hasCharge)
      // scratch
      this.hv = new Float64Array(nr)
      this.p = new Float64Array(mv)
      this.ft = new Float64Array(mi)
      this.Jt = new Float64Array(mi * mv)
      this.F = new Float64Array(mv)
      this.Msys = new Float64Array(mv * mv)
      this.dv = new Float64Array(mv)
      this.vtry = new Float64Array(mv)
      this.vold = new Float64Array(mv)
      this.qtmp = new Float64Array(mi)
      this.ynew = new Float64Array(nr)
      this.out = 0
    }

    // J0/Cq0 at port voltages v (also fills net.q with the charges there)
    linearise(v) {
      const net = this.net
      if (!net.mv) return
      net.evalDevices(v)
      this.J0.set(net.J)
      if (net.hasCharge) {
        net.chargeDevices(v)
        this.Cq0.set(net.C)
      } else this.Cq0.fill(0)
    }

    level(h, depth) {
      let lv = this.levels[depth]
      if (!lv) {
        lv = buildLevel(this.net, h, this.J0, this.Cq0, this.outRow)
        this.levels[depth] = lv
      }
      return lv
    }

    // after a pot move: re-linearise around the present state and drop cached levels
    relink() {
      this.linearise(this.v)
      this.levels = []
      this.havePrev = false
    }

    // device currents incl. trapezoidal charge companions -> net.i, net.J ; q -> this.qtmp
    evalAt(v, a) {
      const net = this.net
      net.evalDevices(v)
      if (!this.chargeGroups) return false
      net.chargeDevices(v)
      const mi = net.mi
      const mv = net.mv
      for (let r = 0; r < mi; r++) {
        net.i[r] += a * (net.q[r] - this.q[r]) - this.iq[r]
        this.qtmp[r] = net.q[r]
      }
      for (let q = 0; q < mi * mv; q++) net.J[q] += a * net.C[q]
      return true
    }

    // Solve v - p + K f~(v) = 0 starting from vstart (copied). Result in this.vtry / this.ft.
    newton(lv, p, vstart) {
      const net = this.net
      const mv = net.mv
      const mi = net.mi
      const K = lv.K
      const Jlin = lv.Jlin
      const v = this.vtry
      v.set(vstart)
      const ft = this.ft
      const Jt = this.Jt
      const F = this.F
      const M = this.Msys
      const dv = this.dv
      for (let it = 0; it < this.maxiter; it++) {
        this.iterations++
        this.hasQ = this.evalAt(v, lv.a)
        const J = net.J
        const PI = net.patI
        const PQ = net.patQ
        const PC = net.patC
        const np = PI.length
        for (let r = 0; r < mi; r++) ft[r] = net.i[r]
        for (let t = 0; t < np; t++) {
          const ix = PI[t]
          Jt[ix] = J[ix] - Jlin[ix]
          ft[PQ[t]] -= Jlin[ix] * v[PC[t]]
        }
        M.fill(0)
        for (let r = 0; r < mv; r++) {
          let s = v[r] - p[r]
          for (let c = 0; c < mi; c++) s += K[r * mi + c] * ft[c]
          F[r] = -s
          M[r * mv + r] = 1
        }
        for (let t = 0; t < np; t++) {
          const q = PQ[t]
          const c = PC[t]
          const jt = Jt[PI[t]]
          if (jt === 0) continue
          for (let r = 0; r < mv; r++) M[r * mv + c] += K[r * mi + q] * jt
        }
        dv.set(F)
        if (!luSolve(M, dv, mv, null)) return false
        let conv = true
        let finite = true
        for (let r = 0; r < mv; r++) {
          const ad = Math.abs(dv[r])
          if (!(ad <= this.vtol + this.reltol * Math.abs(v[r]))) conv = false
          if (!Number.isFinite(dv[r])) finite = false
        }
        if (conv) {
          for (let t = 0; t < np; t++) ft[PQ[t]] += Jt[PI[t]] * dv[PC[t]]
          for (let r = 0; r < mv; r++) v[r] += dv[r]
          return true
        }
        if (!finite) return false
        const old = this.vold
        old.set(v)
        for (let r = 0; r < mv; r++) v[r] += dv[r]
        net.limit(v, old)
      }
      return false
    }

    // advance by h with source value uin (input source only varies); returns ok
    accept(lv, uin, ok) {
      const net = this.net
      const nr = net.nr
      const mi = net.mi
      const ns = net.ns
      const hv = this.hv
      const yn = this.ynew
      const u = this.ubuf
      u[net.inputSource] = uin
      for (let r = 0; r < nr; r++) {
        let s = 0
        for (let j = 0; j < ns; j++) s += lv.Ru[r * ns + j] * u[j]
        for (let j = 0; j < nr; j++) s += lv.Rh[r * nr + j] * hv[j]
        if (net.mv) for (let j = 0; j < mi; j++) s -= lv.Ri[r * mi + j] * this.ft[j]
        yn[r] = s
      }
      let o = 0
      for (let j = 0; j < ns; j++) o += lv.Ou[j] * u[j]
      for (let j = 0; j < nr; j++) o += lv.Oh[j] * hv[j]
      if (net.mv) for (let j = 0; j < mi; j++) o -= lv.Oi[j] * this.ft[j]
      this.out = o
      for (let r = 0; r < nr; r++) {
        this.z[r] = lv.ac[r] * yn[r] - hv[r]
        this.y[r] = yn[r]
      }
      if (this.hasQ) {
        for (let r = 0; r < mi; r++) {
          this.iq[r] = lv.a * (this.qtmp[r] - this.q[r]) - this.iq[r]
          this.q[r] = this.qtmp[r]
        }
      }
      if (net.mv) this.v.set(this.vtry)
      if (!ok) this.failedSteps++
    }

    // one step of size h/2^depth ending at input value uEnd (full source value)
    stepAt(depth, h, uStart, uEnd) {
      const net = this.net
      const lv = this.level(h, depth)
      const nr = net.nr
      const mv = net.mv
      const ns = net.ns
      const hv = this.hv
      for (let r = 0; r < nr; r++) hv[r] = lv.ac[r] * this.y[r] + this.z[r]
      if (!this.ubuf) this.ubuf = net.u()
      const u = this.ubuf
      u[net.inputSource] = uEnd
      this.hasQ = false
      let ok = true
      if (mv) {
        const p = this.p
        for (let r = 0; r < mv; r++) {
          let s = 0
          for (let j = 0; j < ns; j++) s += lv.Pu[r * ns + j] * u[j]
          for (let j = 0; j < nr; j++) s += lv.Ph[r * nr + j] * hv[j]
          p[r] = s
        }
        let start = this.v
        if (depth === 0 && this.predict && this.havePrev) {
          const vp = this.vpred
          for (let r = 0; r < mv; r++) vp[r] = 2 * this.v[r] - this.vprev[r]
          start = vp
        }
        ok = this.newton(lv, p, start)
        if (!ok && depth < this.maxHalvings) {
          this.substeps++
          const uMid = 0.5 * (uStart + uEnd)
          const a = this.stepAt(depth + 1, h / 2, uStart, uMid)
          const b = this.stepAt(depth + 1, h / 2, uMid, uEnd)
          return a && b
        }
      } else {
        this.ft.fill(0)
      }
      this.accept(lv, uEnd, ok)
      return ok
    }

    // one sample: input source value goes from uPrev to uNow; returns the output node voltage
    step(h, uPrev, uNow) {
      if (this.predict) {
        if (!this.vprev) {
          this.vprev = new Float64Array(this.net.mv)
          this.vpred = new Float64Array(this.net.mv)
          this.vkeep = new Float64Array(this.net.mv)
        }
        this.vkeep.set(this.v)
        this.stepAt(0, h, uPrev, uNow)
        this.vprev.set(this.vkeep)
        this.havePrev = true
        return this.out
      }
      this.stepAt(0, h, uPrev, uNow)
      return this.out
    }
  }

  // ---------------------------------------------------------------- high-level engine
  class LiveCircuit {
    constructor(cj, sr, opts) {
      opts = opts || {}
      this.opts = opts
      this.cj = cj
      this.sr = sr
      this.h = 1 / sr
      this.net = new Network(cj, opts)
      if (!cj.output || cj.output < 1) throw new Error('the circuit has no output node (*@ out)')
      this.outRow = cj.output - 1
      for (const p of cj.pots || []) this.net.setPot(p.name, p.pos)
      this.reset(0)
    }

    // DC operating point with the input at dc + x0, then a fresh transient
    reset(x0) {
      const net = this.net
      const dcIn = net.sources[net.inputSource].dc
      const op = net.solveDC(net.u(x0 || 0))
      this.converged = op.ok
      this.method = op.method
      this.xop = op.x
      let base = op.x
      if (x0) base = net.solveDC(net.u(0)).x
      this.base = base[this.outRow]
      this.tr = new Transient(net, op.x, this.outRow, {})
      this.uPrev = dcIn + (x0 || 0)
      this.first = true
    }

    setPot(name, pos) {
      if (this.net.setPot(name, pos)) this.tr.relink()
    }

    // process one input sample (volts on top of the source's dc); returns v(out) - dc(out)
    tick(x) {
      const dcIn = this.net.sources[this.net.inputSource].dc
      const uNow = dcIn + x
      const y = this.tr.step(this.h, this.uPrev, uNow)
      this.uPrev = uNow
      return y - this.base
    }

    // the same contract as analyses.process_audio: sample 0 is the operating point
    process(samples) {
      const n = samples.length
      const out = new Float64Array(n)
      if (!n) return out
      if (samples[0] !== 0) this.reset(samples[0])
      out[0] = this.xop[this.outRow] - this.base
      for (let k = 1; k < n; k++) out[k] = this.tick(samples[k])
      return out
    }
  }

  // ---------------------------------------------------------------- real-time voice
  // Wraps a LiveCircuit for audio callbacks: loops a source buffer (or takes the
  // live input), optional 2x decimation, DC blocking, fixed make-up gain and a
  // soft safety limiter. Shared by the AudioWorklet and the ScriptProcessor fallback.
  class LiveVoice {
    constructor(m) {
      this.sr = m.sr
      this.decim = m.decim || 1
      this.lc = new LiveCircuit(m.compiled, m.sr / this.decim)
      this.src = m.source || null
      this.pos = 0
      this.level = m.level || 0.2
      this.makeup = m.makeup || 1
      this.dryGain = m.dryGain || 1
      this.micVolts = m.micVolts || 0.5
      this.useInput = !!m.useInput
      this.mix = m.bypass ? 0 : 1
      this.target = this.mix
      this.volume = m.volume === undefined ? 1 : m.volume
      this.hpA = 1 - (2 * Math.PI * 6) / m.sr
      this.hpY = 0
      this.hpX = 0
      this.acc = 0
      this.phase = 0
      this.yPrev = 0
      this.yCur = 0
      this.peak = 0
      this.resets = 0
      this.alive = true
    }

    message(m) {
      if (!m) return
      if (m.type === 'pot') this.lc.setPot(m.name, m.pos)
      else if (m.type === 'bypass') this.target = m.on ? 0 : 1
      else if (m.type === 'input') this.useInput = !!m.on
      else if (m.type === 'volume') this.volume = Math.max(0, Math.min(2, Number(m.value) || 0))
      else if (m.type === 'source') {
        this.src = m.source
        this.pos = 0
      } else if (m.type === 'makeup') this.makeup = m.value
    }

    nextInput(inp, k) {
      if (this.useInput) return inp ? inp[k] * this.micVolts : 0
      const s = this.src
      if (!s || !s.length) return 0
      const x = s[this.pos]
      this.pos = this.pos + 1 >= s.length ? 0 : this.pos + 1
      return x * this.level
    }

    circuit(x) {
      let y
      try {
        y = this.lc.tick(x)
      } catch (e) {
        y = NaN
      }
      if (!Number.isFinite(y) || Math.abs(y) > 1e3) {
        this.resets++
        this.lc.reset(0)
        return 0
      }
      return y
    }

    // fill outL/outR (Float32Array) from optional input channel inp
    render(inp, outL, outR) {
      const n = outL.length
      const step = 1 / (0.015 * this.sr)
      for (let k = 0; k < n; k++) {
        const x = this.nextInput(inp, k)
        let y
        if (this.decim === 1) {
          y = this.circuit(x)
        } else {
          this.acc += x
          this.phase++
          if (this.phase >= this.decim) {
            this.yPrev = this.yCur
            this.yCur = this.circuit(this.acc / this.decim)
            this.acc = 0
            this.phase = 0
          }
          y = this.yPrev + ((this.yCur - this.yPrev) * this.phase) / this.decim
        }
        // DC blocker on the circuit output (pot moves shift the bias point)
        const hp = this.hpA * (this.hpY + y - this.hpX)
        this.hpX = y
        this.hpY = hp
        if (this.mix !== this.target) {
          this.mix = this.mix < this.target ? Math.min(this.target, this.mix + step) : Math.max(this.target, this.mix - step)
        }
        const wet = hp * this.makeup
        const dry = x * this.dryGain
        let o = (this.mix * wet + (1 - this.mix) * dry) * this.volume
        o = Math.tanh(o * 1.2) / 1.2
        const a = Math.abs(o)
        if (a > this.peak) this.peak = a
        outL[k] = o
        if (outR) outR[k] = o
      }
    }

    meter() {
      const p = this.peak
      this.peak *= 0.5
      return { peak: p, resets: this.resets, failed: this.lc.tr.failedSteps }
    }
  }

  // Karplus-Strong riff used when no audition has been rendered yet (peak 1).
  function builtinRiff(sr, seconds) {
    const n = Math.floor(sr * (seconds || 6))
    const out = new Float32Array(n)
    const notes = [82.41, 82.41, 98.0, 110.0, 82.41, 123.47, 110.0, 98.0]
    const step = Math.floor(n / notes.length)
    let seed = 7
    const rnd = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff
      return seed / 0x3fffffff - 1
    }
    for (let k = 0; k < notes.length; k++) {
      const period = Math.max(2, Math.round(sr / notes[k]))
      const buf = new Float32Array(period)
      for (let i = 0; i < period; i++) buf[i] = rnd()
      const start = k * step
      const end = Math.min(n, start + step + Math.floor(sr * 0.25))
      for (let i = start; i < end; i++) {
        const j = (i - start) % period
        const v = buf[j]
        buf[j] = 0.497 * (buf[j] + buf[(j + 1) % period])
        const env = Math.exp(-(i - start) / (0.9 * sr)) * Math.min(1, (end - i) / (0.02 * sr))
        out[i] += v * env
      }
    }
    let pk = 0
    for (let i = 0; i < n; i++) pk = Math.max(pk, Math.abs(out[i]))
    for (let i = 0; i < n; i++) out[i] /= pk || 1
    return out
  }

  return { LiveCircuit, LiveVoice, builtinRiff, Network, Transient, potFraction, potHalves, modelParams, pnjlim, fetlim,
    limvds, VT }
}
// ===== HOTWIRE LIVE ENGINE END =====

// ===================================================================== UI

const {
  host, atom, useValue, useQuery, queryClient, Button, Badge, Codicon, Skeleton, Tabs, TabsList, TabsTrigger,
  Tip, Select, SelectContent, SelectItem, SelectTrigger, SelectValue, Dialog, DialogContent, DialogTitle,
  DialogDescription
} = sdk

let CTX = null
let ENGINE = null
const engine = () => ENGINE || (ENGINE = hotwireLiveEngine())

// React element helper: h(type, props, ...children) without JSX.
function h(type, props, ...children) {
  const p = props ? Object.assign({}, props) : {}
  const key = p.key
  delete p.key
  const kids = []
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false && c !== true && c !== '') kids.push(c)
  if (kids.length === 1) p.children = kids[0]
  else if (kids.length > 1) p.children = kids
  return kids.length > 1 ? jsxs(type, p, key) : jsx(type, p, key)
}

// -------------------------------------------------------------------- shared state
const $selected = atom({ design: null, version: null })
const $follow = atom(true)
const $pots = atom({}) // "design@v" -> { RV1: 0.5 }
const $checks = atom({}) // "design@v" -> { busy, result, error }
const $live = atom({ status: 'off', key: null, message: '', bypass: false, input: 'take', decim: 1 })
const $ab = atom({ key: null, playing: false, which: 'wet', loading: false, error: '' })
const $tab = atom('bench')
const $showLib = atom(false)
let lastSeenUpdate = 0

function patch($a, key, value) {
  $a.set(Object.assign({}, $a.get(), { [key]: value }))
}

// Scoped animation loops: every loop a component starts is also stopped when the plugin unloads.
const loops = new Set()
function startLoop(draw, fps) {
  const rate = fps || 30
  let off = null
  if (typeof sdk.createBudgetedLoop === 'function') {
    const l = sdk.createBudgetedLoop(draw, { fps: rate })
    if (l && typeof l.wake === 'function') l.wake()
    off = () => {
      try {
        l.dispose()
      } catch (_) {
        /* gone */
      }
      loops.delete(off)
    }
  } else {
    const stop = CTX ? CTX.setInterval(() => draw(performance.now()), Math.round(1000 / rate)) : null
    off = () => {
      if (stop) stop()
      loops.delete(off)
    }
  }
  loops.add(off)
  return off
}

function friendly(e) {
  const raw = String((e && e.message) || e || '')
  const m = raw.match(/"error"\s*:\s*"((?:[^"\\]|\\.)*)"/)
  if (m) {
    try {
      return sentence(JSON.parse('"' + m[1] + '"'))
    } catch (_) {
      return sentence(m[1])
    }
  }
  if (/\b404\b/.test(raw) || /not found/i.test(raw)) return 'The Hotwire backend is not running. Enable the Hotwire plugin and restart Hermes.'
  if (/\b50[23]\b/.test(raw)) return 'That part of the Hotwire bench is not available on this backend yet.'
  if (/bridge unavailable|Failed to fetch|network/i.test(raw)) return 'Hermes is not reachable right now.'
  return sentence(raw.replace(/^Error:\s*/, '').slice(0, 220)) || 'Something went wrong.'
}

async function rest(path, opts) {
  if (!CTX) throw new Error('Hotwire is not loaded.')
  let r
  try {
    r = await CTX.rest(path, opts)
  } catch (e) {
    throw new Error(friendly(e))
  }
  if (r && typeof r === 'object' && typeof r.error === 'string') throw new Error(r.error)
  return r
}

function parseResult(result) {
  if (!result) return {}
  if (typeof result === 'string') {
    try {
      return JSON.parse(result) || {}
    } catch (_) {
      return {}
    }
  }
  return typeof result === 'object' ? result : {}
}

// -------------------------------------------------------------------- words and numbers
// One unit style everywhere: a space before the unit, a real micro sign, % hugging its number.
function tidy(s) {
  if (s === null || s === undefined) return ''
  return String(s)
    .replace(/\s*->\s*/g, ' → ')
    .replace(/(\d)\s*u(?=[FHAVs]\b)/g, '$1 µ')
    .replace(/(\d)\s?([GMkmµnp]?)(V|A|Hz|Ω|F|W|dB)(?![A-Za-z])/g, '$1 $2$3')
    .replace(/(\d)\s+%/g, '$1%')
    .replace(/(\d)\s*\.\.\s*(\d)/g, '$1–$2')
    .replace(/\s+/g, ' ')
    .trim()
}

function sentence(s) {
  const t = tidy(s)
  if (!t) return ''
  return t[0].toUpperCase() + t.slice(1)
}

function plural(n, one, many) {
  return `${n} ${n === 1 ? one : many || one + 's'}`
}

function ago(ts) {
  if (!ts) return ''
  const s = Math.max(0, Date.now() / 1000 - Number(ts))
  if (s < 45) return 'just now'
  if (s < 3600) return `${Math.max(1, Math.round(s / 60))} min ago`
  if (s < 86400) return `${Math.round(s / 3600)} h ago`
  if (s < 172800) return 'yesterday'
  return `${Math.round(s / 86400)} days ago`
}

function shortDate(ts) {
  if (!ts) return ''
  try {
    return new Date(Number(ts) * 1000).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
  } catch (_) {
    return ''
  }
}

function fmtNum(v, unit) {
  if (v === null || v === undefined || v === '') return '—'
  if (typeof v !== 'number') return tidy(String(v) + (unit ? ` ${unit}` : ''))
  const a = Math.abs(v)
  const pre = [[1e9, 'G'], [1e6, 'M'], [1e3, 'k'], [1, ''], [1e-3, 'm'], [1e-6, 'µ'], [1e-9, 'n'], [1e-12, 'p']]
  if (unit && !['dB', '%', '°', 'h', ''].includes(unit) && a !== 0) {
    for (const [f, p] of pre) {
      if (a >= f * 0.9999) return `${+(v / f).toPrecision(3)} ${p}${unit}`
    }
  }
  return `${+v.toPrecision(3)}${unit ? (unit === '%' || unit === '°' ? unit : ' ' + unit) : ''}`
}

function verdictInfo(v) {
  if (v === 'pass') return { variant: 'success', text: 'Pass', icon: 'pass', tone: 'pass' }
  if (v === 'pass_with_warnings') return { variant: 'warn', text: 'Pass with warnings', icon: 'warning', tone: 'warn' }
  if (v === 'fail') return { variant: 'destructive', text: 'Fail', icon: 'error', tone: 'fail' }
  if (v === 'recheck') return { variant: 'muted', text: 'Needs a re-check', icon: 'history', tone: 'warn' }
  return { variant: 'muted', text: 'Not checked', icon: 'circle-large-outline', tone: '' }
}

// The name people see: the agent's design name when it reads like a name, else the circuit's title.
function designTitle(row) {
  if (!row) return ''
  const name = String(row.name || '').trim()
  if (name && name !== row.id && /\s/.test(name)) return name
  const t = String(row.title || '').trim()
  if (t.length > 2 && !/^untitled circuit$/i.test(t)) return t
  const slug = (name || String(row.id || '')).replace(/[-_]+/g, ' ').trim()
  return slug ? slug[0].toUpperCase() + slug.slice(1) : 'Untitled design'
}

// Picker labels: title, plus "· v3" (or a date) when two designs share a title.
function pickerLabels(designs) {
  const out = new Map()
  const groups = new Map()
  for (const d of designs || []) {
    const t = designTitle(d)
    if (!groups.has(t)) groups.set(t, [])
    groups.get(t).push(d)
  }
  for (const [t, rows] of groups) {
    if (rows.length === 1) {
      out.set(rows[0].id, { title: t, suffix: '' })
      continue
    }
    const vs = rows.map(r => `v${r.latest}`)
    const ds = rows.map(r => shortDate(r.updated || r.created))
    const pick = new Set(vs).size === vs.length ? vs : new Set(ds).size === ds.length ? ds : rows.map(r => r.id)
    rows.forEach((r, i) => out.set(r.id, { title: t, suffix: pick[i] }))
  }
  return out
}

function tidyNote(note) {
  const t = tidy(note).replace(/(\d)-(\d)/g, '$1–$2')
  if (!t) return ''
  return t[0].toUpperCase() + t.slice(1)
}

// Engine SVGs are ours, but they carry netlist text: strip anything active anyway.
function cleanSvg(svg) {
  if (typeof svg !== 'string' || !/<svg[\s>]/i.test(svg)) return ''
  return svg
    .replace(/<script[\s\S]*?<\/script\s*>/gi, '')
    .replace(/<foreignObject[\s\S]*?<\/foreignObject\s*>/gi, '')
    .replace(/\son[a-z]+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi, '')
    .replace(/(href\s*=\s*["']?)\s*javascript:[^"'\s>]*/gi, '$1#')
}

const xmlEsc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
const reEsc = s => String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

// The page header already names the circuit: drop the drawing's own visible title line
// (its <title> stays for screen readers) and, on a schematic, the band it sat in.
function stripSvgTitle(svg, title) {
  if (!svg || !title) return svg
  const re = new RegExp(`<text[^>]*font-weight="600"[^>]*>${reEsc(xmlEsc(title))}</text>`)
  if (!re.test(svg)) return svg
  let out = svg.replace(re, '')
  if (/class="hotwire-schematic"/.test(svg.slice(0, 600))) {
    const band = 28
    out = out.replace(/^(<svg[^>]*?)viewBox="([-\d.]+) ([-\d.]+) ([\d.]+) ([\d.]+)"/, (all, head, x, y, w, hh) => `${head}viewBox="${x} ${+y + band} ${w} ${+hh - band}"`)
    out = out.replace(/^(<svg[^>]*?)height="([\d.]+)"/, (all, head, hh) => `${head}height="${+hh - band}"`)
  }
  return out
}

function svgText(url) {
  if (url.startsWith('data:')) {
    const comma = url.indexOf(',')
    const meta = url.slice(5, comma)
    const body = url.slice(comma + 1)
    if (/;base64/i.test(meta)) {
      const bin = atob(body)
      const bytes = new Uint8Array(bin.length)
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
      return Promise.resolve(new TextDecoder().decode(bytes))
    }
    return Promise.resolve(decodeURIComponent(body))
  }
  return fetch(url).then(r => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    return r.text()
  })
}

function b64ToBytes(b64) {
  const bin = atob(b64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return bytes
}

const potsQuery = pots =>
  Object.keys(pots || {})
    .map(k => `${encodeURIComponent(k)}:${Number(pots[k]).toFixed(4)}`)
    .join(',')

function openBench(design, version) {
  if (design) $selected.set({ design, version: version || null })
  $tab.set('bench')
  $showLib.set(false)
  try {
    host.navigate('/hotwire')
  } catch (_) {
    /* older host */
  }
}

async function copyText(text) {
  let ok = false
  try {
    ok = CTX && CTX.os ? await CTX.os.writeClipboard(text) : false
  } catch (_) {
    ok = false
  }
  if (!ok && typeof sdk.writeClipboardText === 'function') {
    try {
      await sdk.writeClipboardText(text)
      ok = true
    } catch (_) {
      ok = false
    }
  }
  return ok
}

function toast(kind, message) {
  try {
    if (host && typeof host.notify === 'function') host.notify({ kind, message })
  } catch (_) {
    /* no toasts here */
  }
}

// Put a prompt in the chat composer (the focused one, else a fresh chat), else copy it.
async function askHermes(text) {
  const c = host && host.composer
  let ok = false
  if (c) {
    try {
      ok = await c.insertText(null, text, { mode: 'block' })
    } catch (_) {
      ok = false
    }
    if (!ok && typeof host.newChat === 'function') {
      try {
        host.newChat()
        await new Promise(r => (CTX ? CTX.setTimeout(r, 450) : setTimeout(r, 450)))
        ok = await c.setDraft('new', text)
      } catch (_) {
        ok = false
      }
    }
    if (ok) {
      try {
        c.focus(null)
      } catch (_) {
        /* fine */
      }
    }
  }
  if (!ok) {
    const copied = await copyText(text)
    toast('info', copied ? 'Copied the prompt. Paste it into a chat with Hermes.' : text)
  }
  return ok
}

// -------------------------------------------------------------------- styles
const CSS = `
.hwb{--hotwire-ink:var(--ui-text-primary);--hotwire-accent:var(--ui-accent);--hotwire-pass:var(--ui-green);--hotwire-fail:var(--ui-red);--hotwire-copper:var(--ui-orange);--hotwire-board:color-mix(in srgb,var(--ui-yellow) 9%,var(--ui-bg-elevated,var(--ui-bg-card,transparent)));--hwb-line:color-mix(in srgb,var(--ui-stroke-secondary) 55%,transparent);--hwb-hair:color-mix(in srgb,var(--ui-stroke-secondary) 32%,transparent);--hwb-tint:color-mix(in srgb,var(--ui-text-primary) 3.5%,transparent);color:var(--ui-text-primary);font-size:13px;line-height:1.45;container-type:inline-size}
.hwb.hwb-dark{--hotwire-board:color-mix(in srgb,var(--ui-yellow) 16%,var(--ui-bg-tertiary,var(--ui-bg-elevated,transparent)));--hwb-line:color-mix(in srgb,var(--ui-stroke-secondary) 80%,transparent);--hwb-tint:color-mix(in srgb,var(--ui-text-primary) 5%,transparent)}
.hwb *,.hwb *::before,.hwb *::after{box-sizing:border-box}
.hwb-root{height:100%;min-height:0;display:flex;flex-direction:column}
.hwb-scroll{flex:1;min-height:0;overflow:auto;scrollbar-gutter:stable}
.hwb-body{padding:14px 16px 40px;display:flex;flex-direction:column;gap:14px;width:100%;max-width:1320px;margin:0 auto}
.hwb-page .hwb-body{padding:18px 24px 48px}
.hwb-row{display:flex;align-items:center;gap:8px;min-width:0}
.hwb-wrap{flex-wrap:wrap}
.hwb-grow{flex:1;min-width:0}
.hwb-head{display:flex;align-items:flex-start;gap:12px}
.hwb-title{font-size:15px;font-weight:600;letter-spacing:-.01em;margin:0;line-height:1.3;overflow-wrap:anywhere}
.hwb-page .hwb-title{font-size:19px}
@container (max-width: 760px){.hwb-page .hwb-head{flex-wrap:wrap}.hwb-page .hwb-head-side{order:-1;width:100%;flex-direction:row-reverse!important;justify-content:space-between;align-items:center!important}}
.hwb-sub{color:var(--ui-text-tertiary);font-size:12px}
.hwb-muted{color:var(--ui-text-secondary)}
.hwb-faint{color:var(--ui-text-tertiary)}
.hwb-mono,.hwb-code,.hwb-spec-src{font-family:var(--font-mono,ui-monospace,monospace);font-variant-ligatures:none;font-feature-settings:"liga" 0,"calt" 0}
.hwb-mono{font-size:12px}
.hwb-spec-src{font-size:12px;white-space:nowrap}
.hwb-quote{margin:6px 0 0;padding:1px 0 1px 10px;border-left:2px solid var(--hwb-line);color:var(--ui-text-secondary);font-size:12.5px;overflow-wrap:anywhere}
.hwb-card{border:1px solid var(--hwb-line);border-radius:10px;padding:12px 14px 14px;min-width:0;background:var(--ui-bg-elevated,transparent)}
.hwb-label{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--ui-text-tertiary);font-weight:600;margin:0 0 10px;display:flex;align-items:center;gap:8px;min-height:22px}
.hwb-label>.hwb-right{margin-left:auto;text-transform:none;letter-spacing:0;font-weight:500;display:flex;align-items:center;gap:6px}
.hwb-svg{color:var(--hotwire-ink);overflow:hidden}
.hwb-svg svg{display:block;width:100%;height:auto;max-height:560px}
.hwb-board .hwb-svg svg{max-height:none;max-width:980px;margin:0 auto}
.hwb-zoombox{max-height:calc(85vh - 90px);overflow:auto;padding:4px 2px}
.hwb-zoombox .hwb-svg svg{width:100%;height:auto;max-height:calc(85vh - 120px);margin:0 auto}
.hwb-svg rect[width="4"][height="10"][rx="0.8"]{stroke:var(--ui-text-tertiary);stroke-opacity:.75;stroke-width:.6}
.hwb-grid{display:flex;flex-direction:column;gap:14px;min-width:0}
.hwb-col{display:flex;flex-direction:column;gap:14px;min-width:0}
.hwb-grid>.hwb-col{display:contents}
.hwb-o1{order:1}.hwb-o2{order:2}.hwb-o3{order:3}.hwb-o4{order:4}.hwb-o5{order:5}.hwb-o6{order:6}
@container (min-width: 900px){.hwb-grid{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);align-items:start}.hwb-grid>.hwb-col{display:flex}}
.hwb-tl{display:flex;align-items:center;gap:0;overflow-x:auto;padding:2px 2px 4px}
.hwb-tl-link{width:14px;height:1px;background:var(--ui-stroke-secondary);flex:none}
.hwb-pill{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--hwb-line);border-radius:999px;padding:3px 10px 3px 8px;background:transparent;color:var(--ui-text-secondary);cursor:pointer;white-space:nowrap;flex:none;font:inherit;font-size:12px}
.hwb-pill:hover{background:var(--ui-control-hover-background,color-mix(in srgb,var(--ui-text-primary) 6%,transparent))}
.hwb-pill[aria-current="true"]{border-color:var(--ui-accent);color:var(--ui-text-primary);box-shadow:inset 0 0 0 1px var(--ui-accent)}
.hwb-pill:focus-visible,.hwb-btnlike:focus-visible{outline:2px solid var(--ui-accent);outline-offset:2px}
.hwb-dot{width:7px;height:7px;border-radius:50%;background:var(--ui-text-quaternary,var(--ui-text-tertiary));flex:none;display:inline-block}
.hwb-dot.pass{background:var(--ui-green)}.hwb-dot.fail{background:var(--ui-red)}.hwb-dot.warn{background:var(--ui-yellow)}
.hwb-opt{display:inline-flex;align-items:center;gap:8px;min-width:0;max-width:100%}
.hwb-opt-t{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hwb-opt-s{color:var(--ui-text-tertiary);white-space:nowrap}
.hwb-opt-d{color:var(--ui-text-tertiary);font-size:11px;margin-left:12px;white-space:nowrap}
.hwb-table{width:100%;border-collapse:collapse;font-size:12.5px}
.hwb-table th{font-weight:500;color:var(--ui-text-tertiary);text-align:left;padding:0 10px 6px 0;font-size:11.5px}
.hwb-table td{padding:8px 10px 8px 0;border-top:1px solid var(--hwb-hair);vertical-align:top}
.hwb-table td:last-child,.hwb-table th:last-child{text-align:right;padding-right:0}
.hwb-table td.hwb-nw{white-space:nowrap;font-variant-numeric:tabular-nums}
.hwb-table.hwb-left td:last-child,.hwb-table.hwb-left th:last-child{text-align:left}
.hwb-speclabel{cursor:help;text-decoration:underline dotted color-mix(in srgb,var(--ui-text-tertiary) 45%,transparent);text-underline-offset:3px;border-radius:3px;outline:none}
.hwb-speclabel:focus-visible{box-shadow:0 0 0 2px var(--ui-accent)}
.hwb-tgt-inline{display:none;color:var(--ui-text-tertiary);font-size:11.5px;margin-top:2px}
@container (max-width: 520px){.hwb-col-tgt{display:none}.hwb-tgt-inline{display:block}}
.hwb-why{color:var(--ui-text-tertiary);font-size:11.5px;margin-top:3px}
.hwb-knobs{display:flex;flex-wrap:wrap;gap:6px 4px;justify-content:flex-start}
.hwb-knob{display:flex;flex-direction:column;align-items:center;width:96px;padding:4px 0;border-radius:10px;outline:none;user-select:none;-webkit-user-select:none}
.hwb-knob:focus-visible{box-shadow:0 0 0 2px var(--ui-accent)}
.hwb-knob svg{display:block;cursor:ns-resize;touch-action:none;overflow:visible}
.hwb-knob-name{font-size:12px;font-weight:600;margin-top:2px;text-align:center;max-width:92px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hwb-knob-val{font-size:11.5px;color:var(--ui-text-secondary);font-variant-numeric:tabular-nums}
.hwb-knob-part{font-size:10.5px;color:var(--ui-text-tertiary);white-space:nowrap}
.hwb-knob-body{fill:color-mix(in srgb,var(--ui-text-primary) 86%,var(--ui-bg-elevated,var(--ui-bg-card,transparent)))}
.hwb-knob-skirt{fill:color-mix(in srgb,var(--ui-text-primary) 72%,var(--ui-bg-elevated,var(--ui-bg-card,transparent)))}
.hwb-knob-shadow{fill:color-mix(in srgb,var(--ui-text-primary) 10%,transparent)}
.hwb-knob-ptr{stroke:var(--ui-bg-elevated,var(--ui-bg-card,transparent));stroke-width:2.6;stroke-linecap:round}
.hwb-knob-grip{stroke:color-mix(in srgb,var(--ui-bg-elevated,var(--ui-bg-card,transparent)) 25%,transparent);stroke-width:1}
.hwb-knob-track{stroke:color-mix(in srgb,var(--ui-stroke-secondary) 70%,transparent);stroke-width:3.5;fill:none;stroke-linecap:round}
.hwb-knob-arc{stroke:var(--ui-accent);stroke-width:3.5;fill:none;stroke-linecap:round}
.hwb-knob-tick{stroke:var(--ui-text-quaternary,var(--ui-text-tertiary));stroke-width:1}
.hwb-knob.dragging .hwb-knob-arc{stroke-width:4.5}
.hwb-transport{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-top:12px;padding-top:12px;border-top:1px solid var(--hwb-hair)}
.hwb-seg{display:inline-flex;align-items:stretch;border:1px solid var(--hwb-line);border-radius:7px;padding:2px;gap:2px;background:var(--hwb-tint);flex:none}
.hwb-seg button{all:unset;box-sizing:border-box;cursor:pointer;font-size:12px;font-weight:500;line-height:20px;padding:1px 10px;border-radius:5px;color:var(--ui-text-secondary);white-space:nowrap;display:inline-flex;align-items:center;gap:6px}
.hwb-seg button:hover{color:var(--ui-text-primary);background:color-mix(in srgb,var(--ui-text-primary) 6%,transparent)}
.hwb-seg button[aria-checked="true"]{color:var(--ui-text-primary);background:color-mix(in srgb,var(--ui-accent) 16%,var(--ui-bg-elevated,transparent));box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--ui-accent) 70%,transparent)}
.hwb-seg button:focus-visible{outline:2px solid var(--ui-accent);outline-offset:1px}
.hwb-seg-k{font-size:10.5px;font-weight:700;letter-spacing:.04em;opacity:.7}
.hwb-scope{position:relative;margin-top:10px;height:60px;border-radius:7px;background:var(--hwb-tint);border:1px solid var(--hwb-hair);overflow:hidden}
.hwb-scope canvas{position:absolute;inset:0;width:100%;height:100%;display:block}
.hwb-scope-tag{position:absolute;right:8px;top:6px;padding:0 6px;border-radius:4px;background:color-mix(in srgb,var(--ui-bg-elevated,var(--ui-bg-card,transparent)) 82%,transparent);font-size:10px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--ui-text-tertiary);pointer-events:none}
.hwb-scope-tag.on{color:var(--ui-accent)}
.hwb-scope-idle{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;padding:0 14px;font-size:11.5px;line-height:1.35;color:var(--ui-text-tertiary);pointer-events:none}
.hwb-lanes{display:flex;flex-direction:column;gap:8px}
.hwb-lane{display:flex;align-items:center;gap:10px;min-width:0}
.hwb-lane-k{all:unset;box-sizing:border-box;cursor:pointer;width:62px;flex:none;font-size:11px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:var(--ui-text-tertiary);padding:3px 0;border-radius:4px}
.hwb-lane-k:focus-visible{outline:2px solid var(--ui-accent);outline-offset:2px}
.hwb-lane.on .hwb-lane-k{color:var(--ui-accent)}
.hwb-lane-w{flex:1;min-width:0;height:58px;border-radius:7px;background:var(--hwb-tint);border:1px solid var(--hwb-hair);cursor:pointer;position:relative;overflow:hidden}
.hwb-lane.on .hwb-lane-w{border-color:color-mix(in srgb,var(--ui-accent) 55%,transparent)}
.hwb-lane-w svg{display:block;width:100%;height:100%}
.hwb-wave-base{fill:var(--ui-text-quaternary,var(--ui-text-tertiary))}
.hwb-lane.on .hwb-wave-base{fill:color-mix(in srgb,var(--ui-text-secondary) 70%,transparent)}
.hwb-wave-played{fill:var(--ui-accent)}
.hwb-lane:not(.on) .hwb-wave-played{fill:color-mix(in srgb,var(--ui-accent) 35%,transparent)}
.hwb-playhead{stroke:var(--ui-accent);stroke-width:1.5}
.hwb-field{display:flex;flex-direction:column;gap:6px;min-width:0}
.hwb-field>label,.hwb-field-k{font-size:11.5px;color:var(--ui-text-tertiary);font-weight:500}
.hwb-range{width:100%;accent-color:var(--ui-accent);margin:0}
.hwb-note{font-size:12px;color:var(--ui-text-secondary);background:color-mix(in srgb,var(--ui-yellow) 11%,transparent);border-radius:7px;padding:9px 11px}
.hwb-note.hwb-note-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.hwb-err{font-size:12.5px;color:var(--ui-text-secondary);display:flex;gap:8px;align-items:flex-start}
.hwb-err .codicon{color:var(--ui-red);margin-top:2px}
.hwb-list{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:7px}
.hwb-list li{display:flex;gap:8px;align-items:flex-start;font-size:12.5px}
.hwb-list .codicon{margin-top:2px;flex:none}
.hwb-headline{display:flex;gap:9px;align-items:flex-start;font-size:13px;font-weight:500;padding-bottom:10px;margin-bottom:10px;border-bottom:1px solid var(--hwb-hair)}
.hwb-headline:last-child{border-bottom:0;padding-bottom:0;margin-bottom:0}
.hwb-headline .codicon{margin-top:2px;flex:none}
.hwb-lv-error{color:var(--ui-red)}.hwb-lv-warning{color:var(--ui-yellow)}.hwb-lv-ok{color:var(--ui-green)}.hwb-lv-info{color:var(--ui-text-tertiary)}
.hwb-code{margin:0;max-height:560px;overflow:auto;font-size:12px;line-height:1.6;counter-reset:ln;padding:2px 0}
.hwb-code-l{display:grid;grid-template-columns:3.2ch minmax(0,1fr);gap:0 12px;white-space:pre-wrap;overflow-wrap:anywhere;color:var(--ui-text-secondary)}
.hwb-code-l::before{counter-increment:ln;content:counter(ln);text-align:right;color:var(--ui-text-quaternary,var(--ui-text-tertiary));user-select:none;-webkit-user-select:none}
.hwb-code-l>span{padding-left:2ch;text-indent:-2ch}
.hwb-tok-c{color:var(--ui-text-tertiary);font-style:italic}
.hwb-tok-a{color:var(--ui-accent);font-weight:600}
.hwb-tok-n{color:var(--ui-text-primary);font-weight:700}
.hwb-tok-d{color:var(--ui-accent)}
.hwb-lib{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}
.hwb-libcard{all:unset;box-sizing:border-box;cursor:pointer;border:1px solid var(--hwb-line);border-radius:10px;display:flex;flex-direction:column;min-width:0;overflow:hidden;background:var(--ui-bg-elevated,transparent);transition:border-color .12s}
.hwb-libcard:hover{border-color:color-mix(in srgb,var(--ui-accent) 55%,transparent)}
.hwb-libcard:focus-visible{outline:2px solid var(--ui-accent);outline-offset:1px}
.hwb-libcard[aria-busy="true"]{opacity:.7;cursor:progress}
.hwb-thumb{height:118px;padding:10px 12px;background:var(--hwb-tint);border-bottom:1px solid var(--hwb-hair);color:var(--ui-text-secondary);display:flex;align-items:center;justify-content:center}
.hwb-thumb .hwb-svg{width:100%;height:100%}
.hwb-thumb .hwb-svg svg{width:100%;height:100%;max-height:none}
.hwb-libtext{padding:10px 12px 12px;display:flex;flex-direction:column;gap:4px;min-width:0}
.hwb-libtext b{font-weight:600;font-size:12.5px;line-height:1.35}
.hwb-libdesc{font-size:11.5px;color:var(--ui-text-tertiary);display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;line-height:1.45}
.hwb-chip{display:inline-flex;align-items:center;gap:4px;font-size:10.5px;color:var(--ui-text-tertiary);border:1px solid var(--hwb-hair);border-radius:999px;padding:0 7px;line-height:17px;align-self:flex-start;text-transform:capitalize}
.hwb-empty{display:flex;flex-direction:column;gap:12px;padding:28px 4px 6px;max-width:720px}
.hwb-page .hwb-empty{padding-top:40px}
.hwb-empty h2{margin:0;font-size:20px;font-weight:600;letter-spacing:-.015em}
.hwb-empty p{margin:0;color:var(--ui-text-secondary);font-size:13.5px;line-height:1.55;max-width:60ch}
.hwb-empty-icon{width:44px;height:44px;border-radius:11px;display:flex;align-items:center;justify-content:center;color:var(--ui-accent);background:color-mix(in srgb,var(--ui-accent) 12%,transparent);border:1px solid color-mix(in srgb,var(--ui-accent) 28%,transparent)}
.hwb-prompts{display:flex;flex-direction:column;gap:8px;margin-top:4px}
.hwb-prompt{all:unset;box-sizing:border-box;cursor:pointer;display:flex;align-items:center;gap:10px;border:1px solid var(--hwb-line);border-radius:9px;padding:9px 12px;font-size:13px;color:var(--ui-text-primary);background:var(--ui-bg-elevated,transparent)}
.hwb-prompt:hover{border-color:color-mix(in srgb,var(--ui-accent) 55%,transparent);background:color-mix(in srgb,var(--ui-accent) 5%,var(--ui-bg-elevated,transparent))}
.hwb-prompt:focus-visible{outline:2px solid var(--ui-accent);outline-offset:1px}
.hwb-prompt .codicon{color:var(--ui-accent);flex:none}
.hwb-prompt span{flex:1;min-width:0}
.hwb-steps3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:6px}
.hwb-steps3>div{border-top:2px solid var(--hwb-line);padding-top:8px;font-size:12px;color:var(--ui-text-secondary)}
.hwb-steps3 b{display:block;color:var(--ui-text-primary);font-weight:600;font-size:12.5px;margin-bottom:2px}
@container (max-width: 520px){.hwb-steps3{grid-template-columns:1fr}}
.hwb-section{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--ui-text-tertiary);font-weight:600;margin:14px 0 0}
.hwb-checks{list-style:none;margin:0;padding:0;display:flex;flex-direction:column}
.hwb-check{display:grid;grid-template-columns:22px 26px minmax(0,1fr);gap:0 10px;align-items:start;padding:10px 4px;border-top:1px solid var(--hwb-hair);font-size:13px;line-height:1.55;max-width:900px}
.hwb-check:first-child{border-top:0}
.hwb-check.done .hwb-check-t{color:var(--ui-text-tertiary)}
.hwb-check-n{font-size:11.5px;color:var(--ui-text-tertiary);font-variant-numeric:tabular-nums;padding-top:2px;text-align:right}
.hwb-tick{all:unset;box-sizing:border-box;cursor:pointer;width:20px;height:20px;border-radius:6px;border:1.5px solid var(--ui-stroke-secondary);display:flex;align-items:center;justify-content:center;margin-top:1px;color:transparent}
.hwb-tick:hover{border-color:var(--ui-accent)}
.hwb-tick:focus-visible{outline:2px solid var(--ui-accent);outline-offset:2px}
.hwb-tick[aria-checked="true"]{background:var(--ui-accent);border-color:var(--ui-accent);color:var(--ui-bg-elevated,var(--ui-text-primary))}
.hwb-progress{height:4px;border-radius:2px;background:color-mix(in srgb,var(--ui-text-primary) 8%,transparent);overflow:hidden;width:120px}
.hwb-progress>i{display:block;height:100%;background:var(--ui-accent)}
.hwb-sk{display:flex;flex-direction:column;gap:10px}
.hwb-chat{max-width:680px;margin:6px 0}
.hwb-spin{display:inline-block;width:10px;height:10px;border-radius:50%;border:1.5px solid var(--ui-text-quaternary,var(--ui-text-tertiary));border-top-color:var(--ui-accent);animation:hwb-rot .8s linear infinite;vertical-align:-1px}
@keyframes hwb-rot{to{transform:rotate(360deg)}}
@media (prefers-reduced-motion: reduce){.hwb *{animation:none!important;transition:none!important}}
`

function Style() {
  return h('style', null, CSS)
}

// The painted theme (a skin can keep a bright surface in "dark"), so drawings can follow it.
function useDark() {
  const t = typeof sdk.useTheme === 'function' ? sdk.useTheme() : null
  return !!(t && (t.renderedMode || t.resolvedMode) === 'dark')
}

// -------------------------------------------------------------------- small pieces
function Icon({ name, className, size }) {
  return h(Codicon, { name, className, size: size || 14 })
}

function VerdictBadge({ verdict, size }) {
  const v = verdictInfo(verdict)
  return h(Badge, { variant: v.variant, size: size || 'default' }, h(Codicon, { name: v.icon, size: 11 }), v.text)
}

function Spinner() {
  return h('span', { className: 'hwb-spin', role: 'status', 'aria-label': 'Working' })
}

function ErrorLine({ message, onRetry }) {
  return h(
    'div',
    { className: 'hwb-err', role: 'alert' },
    h(Icon, { name: 'error' }),
    h('span', { className: 'hwb-grow' }, message),
    onRetry && h(Button, { size: 'xs', variant: 'ghost', onClick: onRetry }, 'Retry')
  )
}

function Svg({ svg, label, stripTitle }) {
  const html = useMemo(() => cleanSvg(stripTitle ? stripSvgTitle(svg, stripTitle) : svg), [svg, stripTitle])
  if (!html) return null
  return h('div', {
    className: 'hwb-svg',
    role: 'img',
    'aria-label': label || 'drawing',
    dangerouslySetInnerHTML: { __html: html }
  })
}

function Card({ label, right, children, className, labelId }) {
  return h(
    'section',
    { className: 'hwb-card' + (className ? ' ' + className : ''), 'aria-labelledby': labelId || undefined },
    label && h('h3', { className: 'hwb-label', id: labelId || undefined }, label, right && h('span', { className: 'hwb-right' }, right)),
    children
  )
}

// A segmented switch whose selected option is obvious in both themes (and never looks disabled).
function Seg({ value, options, onChange, label }) {
  return h(
    'div',
    { className: 'hwb-seg', role: 'radiogroup', 'aria-label': label },
    options.map(o =>
      h(
        'button',
        {
          key: o.id,
          type: 'button',
          role: 'radio',
          'aria-checked': value === o.id ? 'true' : 'false',
          onClick: () => onChange(o.id),
          title: o.title || undefined
        },
        o.icon && h(Icon, { name: o.icon, size: 12 }),
        o.label
      )
    )
  )
}

// Zoom: the drawing in a dialog, at the full width of the window.
function ZoomButton({ title, svg, stripTitle }) {
  const [open, setOpen] = useState(false)
  const dark = useDark()
  if (!svg || typeof Dialog !== 'function' && typeof Dialog !== 'object') return null
  return h(
    'span',
    null,
    h(Button, { size: 'xs', variant: 'ghost', onClick: () => setOpen(true), 'aria-label': `Zoom: ${title}` }, h(Icon, { name: 'zoom-in' }), 'Zoom'),
    open &&
      h(
        Dialog,
        { open, onOpenChange: setOpen },
        h(
          DialogContent,
          { style: { width: 'min(1240px, 94vw)', maxWidth: '94vw' } },
          h(DialogTitle, null, title),
          DialogDescription && h(DialogDescription, { className: 'sr-only', style: { position: 'absolute', width: 1, height: 1, overflow: 'hidden', clip: 'rect(0 0 0 0)' } }, `${title}, enlarged`),
          h('div', { className: 'hwb hwb-zoombox' + (dark ? ' hwb-dark' : '') }, h(Style), h(Svg, { svg, label: title, stripTitle }))
        )
      )
  )
}

// -------------------------------------------------------------------- knob
const KNOB_START = -135
const KNOB_SWEEP = 270

function polar(cx, cy, r, deg) {
  const t = (deg * Math.PI) / 180
  return [cx + r * Math.sin(t), cy - r * Math.cos(t)]
}

function arcPath(cx, cy, r, a0, a1) {
  const [x0, y0] = polar(cx, cy, r, a0)
  const [x1, y1] = polar(cx, cy, r, a1)
  const large = Math.abs(a1 - a0) > 180 ? 1 : 0
  return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`
}

function taperWord(t) {
  return t === 'log' ? 'audio taper' : t === 'revlog' ? 'reverse audio taper' : 'linear'
}

function Knob({ pot, value, onChange, onCommit, size }) {
  const S = size || 72
  const c = S / 2
  const [drag, setDrag] = useState(false)
  const ref = useRef(null)
  const st = useRef({ y: 0, v: 0 })
  const live = useRef({ value, onChange, onCommit })
  live.current = { value, onChange, onCommit }
  const clamp = v => Math.min(1, Math.max(0, v))

  useEffect(() => {
    const el = ref.current
    if (!el) return undefined
    let timer = null
    const onWheel = ev => {
      ev.preventDefault()
      const d = ev.deltaY > 0 ? -1 : 1
      const step = ev.shiftKey ? 0.005 : 0.02
      const nv = clamp(live.current.value + d * step)
      live.current.onChange(nv)
      if (timer) timer()
      timer = CTX ? CTX.setTimeout(() => live.current.onCommit(live.current.value), 160) : null
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => {
      el.removeEventListener('wheel', onWheel)
      if (timer) timer()
    }
  }, [])

  const down = ev => {
    if (ev.button !== undefined && ev.button !== 0) return
    ev.preventDefault()
    try {
      ev.currentTarget.setPointerCapture(ev.pointerId)
    } catch (_) {
      /* ignore */
    }
    st.current = { y: ev.clientY, v: value }
    setDrag(true)
    if (ref.current) ref.current.focus()
  }
  const move = ev => {
    if (!drag) return
    const span = ev.shiftKey ? 700 : 180
    onChange(clamp(st.current.v + (st.current.y - ev.clientY) / span))
  }
  const up = ev => {
    if (!drag) return
    setDrag(false)
    try {
      ev.currentTarget.releasePointerCapture(ev.pointerId)
    } catch (_) {
      /* ignore */
    }
    onCommit(live.current.value)
  }
  const key = ev => {
    const big = ev.shiftKey ? 0.05 : 0.01
    let nv = null
    if (ev.key === 'ArrowUp' || ev.key === 'ArrowRight') nv = value + big
    else if (ev.key === 'ArrowDown' || ev.key === 'ArrowLeft') nv = value - big
    else if (ev.key === 'PageUp') nv = value + 0.1
    else if (ev.key === 'PageDown') nv = value - 0.1
    else if (ev.key === 'Home') nv = 0
    else if (ev.key === 'End') nv = 1
    if (nv === null) return
    ev.preventDefault()
    onChange(clamp(nv))
  }
  const keyUp = ev => {
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'PageUp', 'PageDown', 'Home', 'End'].includes(ev.key)) onCommit(live.current.value)
  }
  const reset = () => {
    onChange(pot.pos)
    onCommit(pot.pos)
  }

  const ang = KNOB_START + KNOB_SWEEP * value
  const rArc = c - 4
  const rBody = c - 13
  const ticks = []
  for (let i = 0; i <= 10; i++) {
    const a = KNOB_START + (KNOB_SWEEP * i) / 10
    const [x0, y0] = polar(c, c, c - 0.5, a)
    const [x1, y1] = polar(c, c, c + 2.5, a)
    ticks.push(h('line', { key: 't' + i, className: 'hwb-knob-tick', x1: x0, y1: y0, x2: x1, y2: y1, opacity: i % 5 === 0 ? 0.9 : 0.45 }))
  }
  const grips = []
  for (let i = 0; i < 24; i++) {
    const a = (360 * i) / 24
    const [x0, y0] = polar(c, c + 1, rBody - 0.5, a)
    const [x1, y1] = polar(c, c + 1, rBody + 2.6, a)
    grips.push(h('line', { key: 'g' + i, className: 'hwb-knob-grip', x1: x0, y1: y0, x2: x1, y2: y1 }))
  }
  const [px0, py0] = polar(c, c, rBody * 0.18, ang)
  const [px1, py1] = polar(c, c, rBody * 0.86, ang)
  const label = pot.label || pot.name
  const valueText = tidy(pot.value_text || '')
  const pct = Math.round(value * 100)
  return h(
    'div',
    {
      ref,
      className: 'hwb-knob' + (drag ? ' dragging' : ''),
      role: 'slider',
      tabIndex: 0,
      'aria-label': `${label} (${pot.name}, ${valueText} ${taperWord(pot.taper)})`,
      'aria-valuemin': 0,
      'aria-valuemax': 100,
      'aria-valuenow': pct,
      'aria-valuetext': `${pct}%`,
      onKeyDown: key,
      onKeyUp: keyUp,
      onDoubleClick: reset,
      title: `${label}: drag, scroll or use the arrow keys. Double-click resets to ${Math.round(pot.pos * 100)}%.`
    },
    h(
      'svg',
      {
        width: S,
        height: S,
        viewBox: `-3 -3 ${S + 6} ${S + 6}`,
        onPointerDown: down,
        onPointerMove: move,
        onPointerUp: up,
        onPointerCancel: up,
        'aria-hidden': 'true'
      },
      ticks,
      h('path', { className: 'hwb-knob-track', d: arcPath(c, c, rArc, KNOB_START, KNOB_START + KNOB_SWEEP) }),
      value > 0.003 && h('path', { className: 'hwb-knob-arc', d: arcPath(c, c, rArc, KNOB_START, ang) }),
      h('circle', { className: 'hwb-knob-shadow', cx: c, cy: c + 2.5, r: rBody + 3 }),
      h('circle', { className: 'hwb-knob-skirt', cx: c, cy: c + 1, r: rBody + 2.6 }),
      grips,
      h('circle', { className: 'hwb-knob-body', cx: c, cy: c, r: rBody }),
      h('line', { className: 'hwb-knob-ptr', x1: px0, y1: py0, x2: px1, y2: py1 })
    ),
    h('div', { className: 'hwb-knob-name' }, label),
    h('div', { className: 'hwb-knob-val' }, `${pct}%`),
    h('div', { className: 'hwb-knob-part' }, `${valueText}${valueText ? ' · ' : ''}${pot.taper === 'log' ? 'log' : pot.taper === 'revlog' ? 'rev log' : 'lin'}`)
  )
}

// -------------------------------------------------------------------- knob state + re-check
const checkTimers = {}
let checkSeq = 0

function potKey(design, version) {
  return `${design}@${version}`
}

function usePotState(design, version, pots) {
  const key = potKey(design, version)
  const all = useValue($pots)
  const checks = useValue($checks)
  const base = useMemo(() => {
    const o = {}
    for (const p of pots || []) o[p.name] = p.pos
    return o
  }, [pots])
  const cur = all[key] || base
  const moved = Object.keys(base).some(k => Math.abs((cur[k] ?? base[k]) - base[k]) > 1e-4)
  const setPot = useCallback(
    (name, pos) => {
      const prev = $pots.get()[key] || base
      patch($pots, key, Object.assign({}, prev, { [name]: pos }))
      liveSetPot(key, name, pos)
    },
    [key, base]
  )
  const commit = useCallback(() => {
    if (checkTimers[key]) checkTimers[key]()
    const run = () => {
      const positions = $pots.get()[key] || base
      const seq = ++checkSeq
      patch($checks, key, Object.assign({}, $checks.get()[key], { busy: true, error: '' }))
      rest('/check', { method: 'POST', body: { design, version, pots: positions } })
        .then(r => {
          if (($checks.get()[key] || {}).seq > seq) return
          patch($checks, key, { busy: false, result: r, error: '', seq, positions })
        })
        .catch(e => patch($checks, key, Object.assign({}, $checks.get()[key], { busy: false, error: friendly(e) })))
    }
    checkTimers[key] = CTX ? CTX.setTimeout(run, 260) : (setTimeout(run, 260), () => {})
  }, [key, base, design, version])
  const reset = useCallback(() => {
    patch($pots, key, Object.assign({}, base))
    for (const n of Object.keys(base)) liveSetPot(key, n, base[n])
    patch($checks, key, { busy: false, result: null, error: '' })
  }, [key, base])
  return { key, positions: cur, moved, setPot, commit, reset, check: checks[key] || {} }
}

// -------------------------------------------------------------------- audio: A/B
let AC = null
let abNodes = null
const bufCache = new Map()

function audioCtx() {
  if (!AC || AC.state === 'closed') {
    const Ctor = globalThis.AudioContext || globalThis.webkitAudioContext
    if (!Ctor) throw new Error('This window has no Web Audio.')
    AC = new Ctor({ latencyHint: 'interactive' })
  }
  if (AC.state === 'suspended') AC.resume().catch(() => {})
  return AC
}

async function decodeBytes(bytes) {
  const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
  // decoding for the waveform needs no running output: an offline context avoids waking audio before a click
  const Off = globalThis.OfflineAudioContext || globalThis.webkitOfflineAudioContext
  if (!AC && Off) {
    try {
      return await new Off(1, 1, 48000).decodeAudioData(copy.slice(0))
    } catch (_) {
      /* fall through to the live context */
    }
  }
  return await audioCtx().decodeAudioData(copy)
}

async function loadWav(path, url) {
  const k = url || path
  if (bufCache.has(k)) return bufCache.get(k)
  let buf
  if (url) {
    const r = await fetch(url)
    if (!r.ok) throw new Error(`could not load the audio (HTTP ${r.status})`)
    buf = await decodeBytes(new Uint8Array(await r.arrayBuffer()))
  } else {
    const r = await rest(`/audio?path=${encodeURIComponent(path)}`)
    buf = await decodeBytes(b64ToBytes(r.b64))
  }
  if (bufCache.size > 12) bufCache.delete(bufCache.keys().next().value)
  bufCache.set(k, buf)
  return buf
}

function abStop() {
  if (abNodes) {
    for (const n of abNodes.srcs) {
      try {
        n.stop()
      } catch (_) {
        /* already stopped */
      }
    }
    abNodes = null
  }
  $ab.set(Object.assign({}, $ab.get(), { playing: false }))
}

function abPlay(key, wet, dry, which, offset) {
  liveStop()
  abStop()
  const ac = audioCtx()
  const t0 = ac.currentTime + 0.05
  const dur = Math.min(wet.duration, dry.duration) || wet.duration
  const at = Math.max(0, Math.min(dur - 0.01, offset || 0))
  const mk = (buf, on) => {
    const s = ac.createBufferSource()
    s.buffer = buf
    s.loop = true
    const g = ac.createGain()
    g.gain.value = on ? 1 : 0
    s.connect(g).connect(ac.destination)
    s.start(t0, at)
    return [s, g]
  }
  const [sw, gw] = mk(wet, which !== 'dry')
  const [sd, gd] = mk(dry, which === 'dry')
  abNodes = { srcs: [sw, sd], gains: { wet: gw, dry: gd }, t0: t0 - at, dur }
  $ab.set({ key, playing: true, which, loading: false, error: '' })
}

function abSwitch(which) {
  const st = $ab.get()
  $ab.set(Object.assign({}, st, { which }))
  if (!abNodes) return
  const ac = audioCtx()
  const t = ac.currentTime
  for (const [name, g] of Object.entries(abNodes.gains)) {
    g.gain.cancelScheduledValues(t)
    g.gain.setValueAtTime(g.gain.value, t)
    g.gain.linearRampToValueAtTime(name === which ? 1 : 0, t + 0.012)
  }
}

function abPosition() {
  if (!abNodes || !AC) return 0
  const d = abNodes.dur || 1
  const t = AC.currentTime - abNodes.t0
  if (t < 0) return 0
  return (t % d) / d
}

function peaksOf(buf, bins) {
  const ch = buf.getChannelData(0)
  const out = new Float32Array(bins)
  const step = Math.max(1, Math.floor(ch.length / bins))
  for (let b = 0; b < bins; b++) {
    let m = 0
    const s = b * step
    const e = Math.min(ch.length, s + step)
    for (let i = s; i < e; i++) {
      const a = Math.abs(ch[i])
      if (a > m) m = a
    }
    out[b] = m
  }
  let pk = 0
  for (let b = 0; b < bins; b++) pk = Math.max(pk, out[b])
  for (let b = 0; b < bins; b++) out[b] /= pk || 1
  return out
}

const WAVE_W = 600
const WAVE_H = 56

function barsPath(peaks) {
  const bw = WAVE_W / peaks.length
  let d = ''
  for (let i = 0; i < peaks.length; i++) {
    const v = Math.max(0.025, peaks[i]) * (WAVE_H / 2 - 4)
    const x = i * bw + bw * 0.18
    d += `M${x.toFixed(2)} ${(WAVE_H / 2 - v).toFixed(2)}h${(bw * 0.64).toFixed(2)}v${(2 * v).toFixed(2)}h${(-bw * 0.64).toFixed(2)}z`
  }
  return d
}

let clipSeq = 0

// Dry over circuit, stacked: the selected lane is lit, the played part is coloured, one playhead runs through both.
function AbLanes({ bufs, which, playing, onPick, onSeek }) {
  const paths = useMemo(() => (bufs ? { dry: barsPath(peaksOf(bufs.dry, 200)), wet: barsPath(peaksOf(bufs.wet, 200)) } : null), [bufs])
  const ids = useMemo(() => ({ dry: `hwb-clip-${++clipSeq}`, wet: `hwb-clip-${++clipSeq}` }), [])
  const refs = useRef({})
  useEffect(() => {
    const set = p => {
      for (const k of ['dry', 'wet']) {
        const r = refs.current[k]
        if (!r) continue
        const x = (p * WAVE_W).toFixed(2)
        if (r.clip) r.clip.setAttribute('width', x)
        if (r.head) {
          r.head.setAttribute('x1', x)
          r.head.setAttribute('x2', x)
          r.head.style.opacity = playing ? '1' : '0'
        }
      }
    }
    if (!playing) {
      set(0)
      return undefined
    }
    return startLoop(() => set(abPosition()), 30)
  }, [playing, paths])
  if (!paths) return h('div', { className: 'hwb-lanes' }, h(Skeleton, { style: { height: 58, width: '100%' } }), h(Skeleton, { style: { height: 58, width: '100%' } }))
  const lane = (k, title) => {
    const on = which === k
    const seek = ev => {
      const r = ev.currentTarget.getBoundingClientRect()
      onSeek(k, Math.min(1, Math.max(0, (ev.clientX - r.left) / Math.max(1, r.width))))
    }
    return h(
      'div',
      { key: k, className: 'hwb-lane' + (on ? ' on' : '') },
      h('button', { type: 'button', className: 'hwb-lane-k', onClick: () => onPick(k), 'aria-pressed': on ? 'true' : 'false' }, title),
      h(
        'div',
        { className: 'hwb-lane-w', onClick: seek, title: `Play the ${title.toLowerCase()} take from here` },
        h(
          'svg',
          { viewBox: `0 0 ${WAVE_W} ${WAVE_H}`, preserveAspectRatio: 'none', 'aria-hidden': 'true' },
          h('defs', null, h('clipPath', { id: ids[k] }, h('rect', { ref: el => (refs.current[k] = Object.assign(refs.current[k] || {}, { clip: el })), x: 0, y: 0, width: 0, height: WAVE_H }))),
          h('path', { className: 'hwb-wave-base', d: paths[k] }),
          h('path', { className: 'hwb-wave-played', d: paths[k], clipPath: `url(#${ids[k]})` }),
          h('line', { ref: el => (refs.current[k] = Object.assign(refs.current[k] || {}, { head: el })), className: 'hwb-playhead', x1: 0, x2: 0, y1: 0, y2: WAVE_H, style: { opacity: 0 } })
        )
      )
    )
  }
  return h('div', { className: 'hwb-lanes', role: 'group', 'aria-label': 'Dry and circuit takes' }, lane('dry', 'Dry'), lane('wet', 'Circuit'))
}

function useTakes(wetPath, dryPath, wetUrl) {
  const [bufs, setBufs] = useState(null)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(false)
  const load = useCallback(async () => {
    if (bufs) return bufs
    setLoading(true)
    setErr('')
    try {
      const [w, d] = await Promise.all([loadWav(wetPath, wetUrl), loadWav(dryPath)])
      const b = { wet: w, dry: d }
      setBufs(b)
      return b
    } catch (e) {
      setErr(friendly(e))
      return null
    } finally {
      setLoading(false)
    }
  }, [bufs, wetPath, dryPath, wetUrl])
  useEffect(() => {
    setBufs(null)
    setErr('')
    let alive = true
    Promise.all([loadWav(wetPath, wetUrl), loadWav(dryPath)])
      .then(([w, d]) => alive && setBufs({ wet: w, dry: d }))
      .catch(e => alive && setErr(friendly(e)))
    return () => {
      alive = false
    }
  }, [wetPath, dryPath, wetUrl])
  return { bufs, err, loading, load }
}

function ABPlayer({ abKey, wetPath, dryPath, wetUrl, compact, right }) {
  const st = useValue($ab)
  const t = useTakes(wetPath, dryPath, wetUrl)
  const mine = st.key === abKey
  const playing = mine && st.playing
  const which = mine ? st.which : 'wet'

  const toggle = async () => {
    if (playing) return abStop()
    const b = await t.load()
    if (b) abPlay(abKey, b.wet, b.dry, which)
  }
  const pick = async w => {
    if (playing) return abSwitch(w)
    $ab.set(Object.assign({}, $ab.get(), { key: abKey, which: w, playing: false }))
  }
  const seek = async (w, frac) => {
    const b = await t.load()
    if (!b) return
    const dur = Math.min(b.wet.duration, b.dry.duration) || b.wet.duration
    abPlay(abKey, b.wet, b.dry, w, frac * dur)
  }
  return h(
    'div',
    { className: 'hwb-col', style: { gap: 12 } },
    h(
      'div',
      { className: 'hwb-row hwb-wrap' },
      h(
        Button,
        { size: 'sm', variant: playing ? 'secondary' : 'default', onClick: toggle, disabled: t.loading, 'aria-pressed': playing },
        t.loading ? h(Spinner) : h(Icon, { name: playing ? 'debug-stop' : 'play' }),
        playing ? 'Stop' : 'Play'
      ),
      h(Seg, {
        label: 'Which take you hear',
        value: which,
        onChange: pick,
        options: [
          { id: 'dry', label: 'A · Dry' },
          { id: 'wet', label: 'B · Circuit' }
        ]
      }),
      right && h('span', { className: 'hwb-sub', style: { marginLeft: 'auto' } }, right)
    ),
    h(AbLanes, { bufs: t.bufs, which, playing, onPick: pick, onSeek: seek }),
    !compact && h('div', { className: 'hwb-sub' }, 'Both takes are loudness matched, so you hear tone, not level. Click a lane to play from there; switch A and B while it plays.'),
    t.err && h(ErrorLine, { message: t.err, onRetry: t.load })
  )
}

// -------------------------------------------------------------------- audio: Live
let liveNode = null
let liveVoiceMain = null // ScriptProcessor fallback voice
let liveStream = null
let liveTimer = null
let liveAnalyser = null
let workletUrl = null
let workletCtx = null

function workletSource() {
  return [
    '// Hotwire live circuit (AudioWorklet module built from the plugin source).',
    `const hotwireLiveEngine = ${hotwireLiveEngine.toString()};`,
    `;(${hotwireWorkletMain.toString()})();`
  ].join('\n')
}

// Runs inside AudioWorkletGlobalScope only (registerProcessor/sampleRate are worklet globals).
function hotwireWorkletMain() {
  const HW = hotwireLiveEngine()
  class HotwireLive extends AudioWorkletProcessor {
    constructor() {
      super()
      this.voice = null
      this.frames = 0
      this.port.onmessage = ev => {
        const m = ev.data || {}
        if (m.type === 'init') {
          try {
            this.voice = new HW.LiveVoice(m)
            this.port.postMessage({ type: 'ready' })
          } catch (e) {
            this.port.postMessage({ type: 'error', message: String((e && e.message) || e) })
          }
        } else if (m.type === 'stop') {
          this.voice = null
        } else if (this.voice) {
          this.voice.message(m)
        }
      }
    }
    process(inputs, outputs) {
      const out = outputs[0]
      if (!out || !out[0]) return true
      if (!this.voice) {
        for (const ch of out) ch.fill(0)
        return true
      }
      const inp = inputs[0] && inputs[0][0] ? inputs[0][0] : null
      this.voice.render(inp, out[0], out[1] || null)
      this.frames += out[0].length
      if (this.frames >= 6000) {
        this.frames = 0
        this.port.postMessage(Object.assign({ type: 'meter' }, this.voice.meter()))
      }
      return true
    }
  }
  registerProcessor('hotwire-live', HotwireLive)
}

async function ensureWorklet(ac) {
  if (workletCtx === ac) return true
  if (!ac.audioWorklet || typeof AudioWorkletNode === 'undefined') return false
  if (!workletUrl) workletUrl = URL.createObjectURL(new Blob([workletSource()], { type: 'text/javascript' }))
  await ac.audioWorklet.addModule(workletUrl)
  workletCtx = ac
  return true
}

function liveSetPot(key, name, pos) {
  const st = $live.get()
  if (st.key !== key || (st.status !== 'on' && st.status !== 'loading')) return
  if (liveNode && liveNode.port) liveNode.port.postMessage({ type: 'pot', name, pos })
  else if (liveVoiceMain) liveVoiceMain.message({ type: 'pot', name, pos })
}

function liveSend(m) {
  if (liveNode && liveNode.port) liveNode.port.postMessage(m)
  else if (liveVoiceMain) liveVoiceMain.message(m)
}

function liveStop() {
  if (liveTimer) {
    liveTimer()
    liveTimer = null
  }
  if (liveAnalyser) {
    try {
      liveAnalyser.disconnect()
    } catch (_) {
      /* gone */
    }
    liveAnalyser = null
  }
  if (liveNode) {
    try {
      if (liveNode.port) liveNode.port.postMessage({ type: 'stop' })
      liveNode.disconnect()
    } catch (_) {
      /* gone */
    }
    liveNode = null
  }
  liveVoiceMain = null
  if (liveStream) {
    for (const t of liveStream.getTracks()) t.stop()
    liveStream = null
  }
  const st = $live.get()
  if (st.status !== 'off' && st.status !== 'error') $live.set(Object.assign({}, st, { status: 'off' }))
}

function rms(arr) {
  let s = 0
  for (let i = 0; i < arr.length; i++) s += arr[i] * arr[i]
  return Math.sqrt(s / Math.max(1, arr.length))
}

async function liveStart({ key, design, version, positions, dryPath, input }) {
  abStop()
  liveStop()
  const bypass = !!$live.get().bypass
  const inp = input || 'take'
  $live.set({ status: 'loading', key, message: 'Compiling the circuit…', bypass, input: inp, decim: 1 })
  try {
    const ac = audioCtx()
    const sr = ac.sampleRate
    const cj = await rest(`/compile/${encodeURIComponent(design)}?v=${version}&pots=${potsQuery(positions)}`)
    const HW = engine()
    // the source: the rendered dry take when there is one (and it is asked for), else a built-in riff
    let source = null
    if (dryPath && inp === 'take') {
      try {
        const buf = await loadWav(dryPath)
        source = Float32Array.from(buf.getChannelData(0))
      } catch (_) {
        source = null
      }
    }
    if (!source) source = HW.builtinRiff(sr, 6)
    let pk = 0
    for (let i = 0; i < source.length; i++) pk = Math.max(pk, Math.abs(source[i]))
    for (let i = 0; i < source.length; i++) source[i] /= pk || 1
    const level = 0.2
    // measure how fast this machine runs the circuit, and the make-up gain
    $live.set(Object.assign({}, $live.get(), { message: 'Measuring the circuit…' }))
    await new Promise(r => setTimeout(r, 0))
    const probeN = Math.min(source.length, Math.floor(sr * 0.25))
    let decim = 1
    let probe = new HW.LiveCircuit(cj, sr)
    const t0 = performance.now()
    const outs = new Float32Array(probeN)
    for (let i = 0; i < probeN; i++) outs[i] = probe.tick(source[i] * level)
    const rt = (performance.now() - t0) / 1000 / (probeN / sr)
    if (rt > 0.55) {
      decim = 2
      if (rt / 2 > 0.8) {
        throw new Error('This circuit needs more CPU than this machine can give it in real time. Use Listen to render it instead.')
      }
    }
    let mean = 0
    for (let i = 0; i < probeN; i++) mean += outs[i]
    mean /= probeN || 1
    for (let i = 0; i < probeN; i++) outs[i] -= mean
    const wetRms = rms(outs.subarray(Math.floor(probeN / 3))) || 1e-6
    const dryRms = rms(source.subarray(0, probeN)) * level || 1e-6
    const makeup = Math.min(400, 0.16 / wetRms)
    const dryGain = Math.min(400, 0.16 / dryRms)
    probe = null
    const init = { type: 'init', compiled: cj, sr, decim, source, level, makeup, dryGain, micVolts: 0.5, useInput: inp === 'mic', bypass, volume: 1 }
    let node = null
    let viaWorklet = false
    try {
      viaWorklet = await ensureWorklet(ac)
    } catch (_) {
      viaWorklet = false
    }
    if (viaWorklet) {
      node = new AudioWorkletNode(ac, 'hotwire-live', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [2] })
      const ready = new Promise((resolve, reject) => {
        node.port.onmessage = ev => {
          const m = ev.data || {}
          if (m.type === 'ready') resolve()
          else if (m.type === 'error') reject(new Error(m.message))
        }
      })
      node.port.postMessage(init)
      await ready
    } else {
      const voice = new HW.LiveVoice(init)
      node = ac.createScriptProcessor(2048, 1, 2)
      node.onaudioprocess = ev => {
        const i0 = ev.inputBuffer.getChannelData(0)
        voice.render(voice.useInput ? i0 : null, ev.outputBuffer.getChannelData(0), ev.outputBuffer.getChannelData(1))
      }
      liveVoiceMain = voice
    }
    liveNode = node
    if (inp === 'mic') await attachMic(ac)
    node.connect(ac.destination)
    try {
      const an = ac.createAnalyser()
      an.fftSize = 2048
      an.smoothingTimeConstant = 0.55
      node.connect(an)
      liveAnalyser = an
    } catch (_) {
      liveAnalyser = null
    }
    // the knobs may have moved while we were loading
    const latest = $pots.get()[key] || positions
    for (const n of Object.keys(latest || {})) liveSend({ type: 'pot', name: n, pos: latest[n] })
    const pref = !!$live.get().bypass
    if (pref !== bypass) liveSend({ type: 'bypass', on: pref })
    $live.set({
      status: 'on',
      key,
      message: decim > 1 ? 'Running at half rate to keep up on this machine.' : viaWorklet ? '' : 'Running on the main thread (no AudioWorklet here).',
      bypass: pref,
      input: inp,
      decim
    })
  } catch (e) {
    liveStop()
    $live.set({ status: 'error', key, message: friendly(e), bypass, input: inp, decim: 1 })
  }
}

async function attachMic(ac) {
  const md = globalThis.navigator && navigator.mediaDevices
  if (!md || !md.getUserMedia) throw new Error('This window cannot open an audio input.')
  liveStream = await md.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } })
  const src = ac.createMediaStreamSource(liveStream)
  src.connect(liveNode)
  liveSend({ type: 'input', on: true })
}

function cssVar(el, name, fallback) {
  try {
    const v = getComputedStyle(el).getPropertyValue(name).trim()
    return v || fallback
  } catch (_) {
    return fallback
  }
}

// A small live scope of what you are hearing, drawn from an AnalyserNode.
function LiveScope({ on, bypass, idleText }) {
  const dark = useDark()
  const wrap = useRef(null)
  const cv = useRef(null)
  useEffect(() => {
    const box = wrap.current
    const canvas = cv.current
    if (!box || !canvas) return undefined
    let w = 0
    let hh = 0
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1)
    const fit = () => {
      const r = box.getBoundingClientRect()
      w = Math.max(10, Math.round(r.width))
      hh = Math.max(10, Math.round(r.height))
      canvas.width = Math.round(w * dpr)
      canvas.height = Math.round(hh * dpr)
    }
    fit()
    const ro = typeof ResizeObserver === 'function' ? new ResizeObserver(fit) : null
    if (ro) ro.observe(box)
    const g = canvas.getContext('2d')
    const data = new Float32Array(2048)
    const spec = new Float32Array(1024)
    const bars = 56
    const level = new Float32Array(bars)
    let gain = 4
    const colors = () => {
      const base = getComputedStyle(box).color
      return {
        line: cssVar(box, '--ui-stroke-secondary', base),
        ink: bypass ? cssVar(box, '--ui-text-secondary', base) : cssVar(box, '--ui-accent', base)
      }
    }
    const col = colors()
    const draw = () => {
      if (!g) return
      g.setTransform(dpr, 0, 0, dpr, 0, 0)
      g.clearRect(0, 0, w, hh)
      const mid = hh / 2
      g.strokeStyle = col.line
      g.globalAlpha = 0.45
      g.lineWidth = 1
      g.beginPath()
      g.moveTo(0, mid)
      g.lineTo(w, mid)
      g.stroke()
      g.globalAlpha = 1
      const an = liveAnalyser
      if (!on || !an) return
      // spectrum behind: log-spaced bars from 40 Hz to 12 kHz
      const sr = (an.context && an.context.sampleRate) || 48000
      const nb = Math.min(spec.length, an.frequencyBinCount)
      an.getFloatFrequencyData(spec.subarray(0, nb))
      const bw = w / bars
      g.fillStyle = col.ink
      for (let b = 0; b < bars; b++) {
        const f0 = 40 * Math.pow(300, b / bars)
        const f1 = 40 * Math.pow(300, (b + 1) / bars)
        const i0 = Math.max(1, Math.floor((f0 / sr) * 2 * nb))
        const i1 = Math.max(i0 + 1, Math.min(nb, Math.ceil((f1 / sr) * 2 * nb)))
        let m = -160
        for (let i = i0; i < i1; i++) m = Math.max(m, spec[i])
        const v = Math.max(0, Math.min(1, (m + 90) / 70))
        level[b] += (v - level[b]) * 0.35
        const bh = level[b] * (hh - 6)
        g.globalAlpha = 0.16
        g.fillRect(b * bw + 1, hh - bh, Math.max(1, bw - 2), bh)
      }
      g.globalAlpha = 1
      // scope in front, triggered on a rising zero crossing so the trace stands still
      const n = Math.min(data.length, an.fftSize)
      an.getFloatTimeDomainData(data.subarray(0, n))
      let s = 0
      for (let i = 1; i < n / 2; i++) {
        if (data[i - 1] < 0 && data[i] >= 0) {
          s = i
          break
        }
      }
      const span = Math.min(n - s, 1024)
      let sum = 0
      for (let i = 0; i < span; i++) sum += data[s + i] * data[s + i]
      const r = Math.sqrt(sum / Math.max(1, span))
      const want = r > 1e-5 ? 0.38 / r : gain
      gain += (Math.min(60, want) - gain) * 0.12
      g.save()
      g.beginPath()
      g.rect(0, 2, w, hh - 4)
      g.clip()
      g.strokeStyle = col.ink
      g.lineWidth = 1.6
      g.lineJoin = 'round'
      g.beginPath()
      for (let i = 0; i < span; i++) {
        const x = (i / (span - 1)) * w
        const y = mid - data[s + i] * gain * (mid - 4)
        if (i) g.lineTo(x, y)
        else g.moveTo(x, y)
      }
      g.stroke()
      g.restore()
    }
    draw()
    const stop = on ? startLoop(draw, 30) : null
    return () => {
      if (stop) stop()
      if (ro) ro.disconnect()
    }
  }, [on, bypass, dark])
  return h(
    'div',
    { ref: wrap, className: 'hwb-scope', 'aria-hidden': 'true' },
    h('canvas', { ref: cv }),
    on && h('span', { className: 'hwb-scope-tag' + (bypass ? '' : ' on') }, bypass ? 'Bypass · dry signal' : 'Circuit · live'),
    !on && idleText && h('span', { className: 'hwb-scope-idle' }, idleText)
  )
}

function LiveBar({ design, version, positions, dryPath, keyId, compact }) {
  const st = useValue($live)
  const mine = st.key === keyId
  const on = mine && st.status === 'on'
  const loading = mine && st.status === 'loading'
  const [micAsk, setMicAsk] = useState(false)
  // the input you hear: mic only while it is actually running; "riff" only when there is a take to tell apart
  let input = st.input || 'take'
  if (input === 'mic' && !on) input = 'take'
  if (input === 'riff' && !dryPath) input = 'take'
  const start = inp => liveStart({ key: keyId, design, version, positions, dryPath, input: inp })
  const toggle = () => (on || loading ? liveStop() : start(input))
  const setBypass = v => {
    const by = v === 'dry'
    if (on) liveSend({ type: 'bypass', on: by })
    $live.set(Object.assign({}, $live.get(), { bypass: by }))
  }
  const pickInput = v => {
    if (v === 'mic') return setMicAsk(true)
    $live.set(Object.assign({}, $live.get(), { input: v, key: on ? st.key : keyId }))
    if (on) start(v)
  }
  const inputs = [
    { id: 'take', label: dryPath ? 'Audition take' : 'Riff' },
    { id: 'riff', label: 'Built-in riff' },
    { id: 'mic', label: 'Instrument in', icon: 'mic' }
  ].filter(o => o.id !== 'riff' || dryPath)
  return h(
    'div',
    null,
    h(
      'div',
      { className: 'hwb-transport' },
      h(
        Button,
        { size: 'sm', variant: on ? 'secondary' : 'default', onClick: toggle, 'aria-pressed': on },
        loading ? h(Spinner) : h(Icon, { name: on ? 'debug-stop' : 'play' }),
        on ? 'Stop' : loading ? 'Starting…' : 'Play live'
      ),
      h(Seg, {
        label: 'Circuit or bypass',
        value: st.bypass ? 'dry' : 'wet',
        onChange: setBypass,
        options: [
          { id: 'wet', label: 'Circuit', title: 'Hear the signal through the circuit' },
          { id: 'dry', label: 'Bypass', title: 'Hear the dry signal (true bypass)' }
        ]
      }),
      h(Seg, { label: 'Live input', value: input, onChange: pickInput, options: inputs })
    ),
    h(LiveScope, {
      on,
      bypass: !!st.bypass,
      idleText: loading ? st.message || 'Starting…' : compact ? 'Play live to see the signal' : 'Press Play live: the circuit runs in real time and the knobs change it as it plays'
    }),
    micAsk &&
      h(
        'div',
        { className: 'hwb-note', style: { marginTop: 10 } },
        h('div', { style: { marginBottom: 8 } }, 'Use headphones: with speakers the output feeds back into the input and howls. Hermes will ask for microphone or audio-interface access.'),
        h(
          'div',
          { className: 'hwb-row' },
          h(Button, { size: 'sm', onClick: () => { setMicAsk(false); start('mic') } }, h(Icon, { name: 'mic' }), 'Use instrument input'),
          h(Button, { size: 'sm', variant: 'ghost', onClick: () => setMicAsk(false) }, 'Cancel')
        )
      ),
    mine && st.status === 'error' && h('div', { style: { marginTop: 10 } }, h(ErrorLine, { message: st.message })),
    on && h('div', { className: 'hwb-sub', style: { marginTop: 8 } }, st.message || 'Turn a knob: the circuit itself is running, sample by sample.')
  )
}

// -------------------------------------------------------------------- bench sections
function Timeline({ versions, current, onPick }) {
  const items = []
  versions.forEach((v, i) => {
    const vi = v.checked ? (v.verdict === 'pass' ? 'pass' : v.verdict === 'fail' ? 'fail' : v.verdict ? 'warn' : '') : ''
    const score = v.checked && v.specs_total ? `${v.specs_pass}/${v.specs_total}` : v.has_audio ? 'audio' : v.checked ? 'checked' : ''
    if (i) items.push(h('span', { key: 'l' + v.v, className: 'hwb-tl-link', 'aria-hidden': 'true' }))
    items.push(
      h(
        Tip,
        { key: 'v' + v.v, label: `${tidyNote(v.note) || 'No note'} · ${ago(v.time)}` },
        h(
          'button',
          {
            className: 'hwb-pill',
            'aria-current': v.v === current ? 'true' : 'false',
            'aria-label': `Version ${v.v}${score ? `, ${score} specs` : ''}${v.verdict ? `, ${verdictInfo(v.verdict).text}` : ''}`,
            onClick: () => onPick(v.v),
            type: 'button'
          },
          h('span', { className: 'hwb-dot ' + vi }),
          `v${v.v}`,
          score && h('span', { className: 'hwb-faint' }, score)
        )
      )
    )
  })
  return h('nav', { className: 'hwb-tl', 'aria-label': 'Design history' }, items)
}

function specLabel(s) {
  const lab = s.label && s.label !== s.spec ? s.label : ''
  return lab ? sentence(lab) : tidy(s.spec || '')
}

function specMeasured(s) {
  if (s.outdated) return '—'
  if (s.measured_text && s.measured_text !== '—') return tidy(s.measured_text)
  if (s.measured === null || s.measured === undefined) return '—'
  return fmtNum(s.measured, s.unit)
}

function specTarget(s) {
  return tidy(s.target_text || s.target || '—')
}

function SpecTable({ specs }) {
  if (!specs || !specs.length) return h('div', { className: 'hwb-sub' }, "No specs yet. Ask Hermes to add measurable targets, like '*@ spec gain_db(1k) > 12'.")
  return h(
    'table',
    { className: 'hwb-table' },
    h('thead', null, h('tr', null, h('th', null, 'Spec'), h('th', { className: 'hwb-col-tgt' }, 'Target'), h('th', null, 'Measured'), h('th', null, h('span', { className: 'sr-only', style: { position: 'absolute', width: 1, height: 1, overflow: 'hidden', clip: 'rect(0 0 0 0)' } }, 'Result')))),
    h(
      'tbody',
      null,
      specs.map((s, i) => {
        const label = specLabel(s)
        const raw = String(s.spec || '')
        return h(
          'tr',
          { key: i },
          h(
            'td',
            null,
            h(Tip, { label: h('span', { className: 'hwb-spec-src' }, raw) }, h('span', { className: 'hwb-speclabel', tabIndex: 0, 'aria-label': `${label} (spec ${raw})` }, label)),
            h('div', { className: 'hwb-tgt-inline' }, `Target: ${specTarget(s)}`),
            !s.pass && !s.outdated && s.explanation && h('div', { className: 'hwb-why' }, sentence(s.explanation)),
            s.outdated && h('div', { className: 'hwb-why' }, 'Not measured: this check ran on an older Hotwire.')
          ),
          h('td', { className: 'hwb-muted hwb-col-tgt hwb-nw' }, specTarget(s)),
          h('td', { className: 'hwb-nw' }, specMeasured(s)),
          h('td', null, s.outdated ? h(Badge, { variant: 'muted' }, 'Not measured') : h(Badge, { variant: s.pass ? 'success' : 'destructive', size: 'default' }, s.pass ? 'Pass' : 'Fail'))
        )
      })
    )
  )
}

function RecheckNote({ design, version, count, onDone, never }) {
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const run = async () => {
    setBusy(true)
    setErr('')
    try {
      await rest('/check', { method: 'POST', body: { design, version, save: true } })
      await queryClient.invalidateQueries({ queryKey: ['hotwire'] })
      if (onDone) onDone()
    } catch (e) {
      setErr(friendly(e))
    } finally {
      setBusy(false)
    }
  }
  return h(
    'div',
    { className: 'hwb-note hwb-note-row', style: { marginBottom: 10 } },
    h(Icon, { name: never ? 'beaker' : 'history' }),
    h('span', { className: 'hwb-grow' }, never ? 'This version has not been checked yet: no verdict, schematic or Bode plot so far.' : `Checked with an older Hotwire: ${plural(count, 'spec')} could not be measured then.`),
    h(Button, { size: 'xs', variant: 'secondary', onClick: run, disabled: busy }, busy ? h(Spinner) : h(Icon, { name: never ? 'play' : 'refresh' }), busy ? 'Checking…' : never ? 'Check it now' : 'Re-check'),
    err && h('div', { style: { flexBasis: '100%' } }, h(ErrorLine, { message: err }))
  )
}

function stressFallback(check) {
  const rows = []
  for (const e of check.erc || []) rows.push({ level: e.level, text: sentence(e.message) })
  for (const s of check.stress || []) {
    let t = String(s.message || '')
    t = t.replace(/lead to '([^']+)' \(([^)]+)\)/g, (all, node, v) => (node === '0' ? `side to ground` : `side at ${v}`))
    rows.push({ level: s.level, text: sentence(t) })
  }
  for (const b of check.bias || []) rows.push({ level: 'warning', text: sentence(b.message) })
  const order = { error: 0, warning: 1 }
  rows.sort((a, b) => (order[a.level] ?? 2) - (order[b.level] ?? 2))
  const errs = rows.filter(r => r.level === 'error').length
  const warns = rows.filter(r => r.level === 'warning').length
  const headline = errs ? `${plural(errs, 'problem')} to fix before you build` : warns ? `Nothing over its ratings · ${plural(warns, 'warning')}` : 'Every part sits inside its ratings at idle'
  return { headline, tone: errs ? 'fail' : warns ? 'warn' : 'pass', rows }
}

function StressCard({ check, className }) {
  let body
  if (!check || (!check.op && !check.stress && !check.erc)) body = h('div', { className: 'hwb-sub' }, 'Not checked yet.')
  else {
    const v = check.stress_view && check.stress_view.headline ? check.stress_view : stressFallback(check)
    const icon = v.tone === 'fail' ? ['error', 'hwb-lv-error'] : v.tone === 'warn' ? ['warning', 'hwb-lv-warning'] : ['pass', 'hwb-lv-ok']
    body = h(
      'div',
      null,
      h('div', { className: 'hwb-headline' }, h(Icon, { name: icon[0], className: icon[1] }), h('span', null, sentence(v.headline))),
      (v.rows || []).length > 0 &&
        h(
          'ul',
          { className: 'hwb-list' },
          v.rows.map((r, i) =>
            h(
              'li',
              { key: i },
              h(Icon, { name: r.level === 'error' ? 'error' : r.level === 'warning' ? 'warning' : 'info', className: 'hwb-lv-' + (r.level === 'error' ? 'error' : r.level === 'warning' ? 'warning' : 'info') }),
              h('span', null, sentence(r.text))
            )
          )
        )
    )
  }
  return h(Card, { label: 'Stress and wiring', className }, body)
}

function KnobsCard({ design, payload, ps, dryPath, compact, className }) {
  const pots = payload.pots || []
  const busy = ps.check.busy
  const right = h(
    'span',
    { className: 'hwb-row' },
    busy && h('span', { className: 'hwb-sub' }, h(Spinner), ' Re-checking'),
    !busy && ps.moved && h('span', { className: 'hwb-sub' }, 'Preview, not saved'),
    ps.moved && h(Button, { size: 'xs', variant: 'ghost', onClick: ps.reset }, 'Reset')
  )
  return h(
    Card,
    { label: pots.length ? 'Knobs' : 'Play', right: pots.length ? right : null, className },
    pots.length
      ? h(
          'div',
          { className: 'hwb-knobs' },
          pots.map(p =>
            h(Knob, {
              key: p.name,
              pot: p,
              value: ps.positions[p.name] ?? p.pos,
              onChange: v => ps.setPot(p.name, v),
              onCommit: () => ps.commit()
            })
          )
        )
      : h('div', { className: 'hwb-sub' }, 'This circuit has no pots. Ask Hermes to add a *@ pot for a gain, tone or volume knob.'),
    ps.check.error && h('div', { style: { marginTop: 8 } }, h(ErrorLine, { message: ps.check.error })),
    h(LiveBar, { design, version: payload.version, positions: ps.positions, dryPath, keyId: ps.key, compact })
  )
}

// -------------------------------------------------------------------- listen tab
const SOURCES = [
  ['guitar_riff', 'Guitar riff'],
  ['bass_line', 'Bass line'],
  ['drum_loop', 'Drum loop'],
  ['sine_sweep', 'Sine sweep']
]
const sourceName = id => (SOURCES.find(s => s[0] === id) || [id, sentence(String(id || 'audition').replace(/_/g, ' '))])[1]

function takeLabel(a, payload) {
  const bits = [sourceName(a.source || a.tag)]
  if (a.guitar && a.guitar.volume !== undefined && a.guitar.volume !== null && Number(a.guitar.volume) !== 1) bits.push(`guitar volume ${Math.round(Number(a.guitar.volume) * 100)}%`)
  if (a.seconds) bits.push(`${a.seconds} s`)
  if (a.version && payload && a.version !== payload.version) bits.push(`v${a.version}`)
  if (a.mtime) bits.push(ago(a.mtime))
  return bits.join(' · ')
}

function SpectrumCard({ path, className }) {
  const q = useQuery({ queryKey: ['hotwire', 'svg', path], queryFn: () => rest(`/svg?path=${encodeURIComponent(path)}`), enabled: !!path, staleTime: Infinity })
  if (!path) return null
  const svg = q.data && q.data.svg
  return h(
    Card,
    { label: 'Spectrum', className, right: svg && h(ZoomButton, { title: 'Spectrum of the take', svg }) },
    q.isLoading ? h(Skeleton, { style: { height: 180, width: '100%' } }) : q.error ? h(ErrorLine, { message: friendly(q.error), onRetry: () => q.refetch() }) : h(Svg, { svg, label: 'Spectrum of the dry and circuit takes' })
  )
}

function RenderCard({ design, payload, ps, onRendered, className }) {
  const [source, setSource] = useState('guitar_riff')
  const [gv, setGv] = useState(1)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const guitar = source.startsWith('guitar')
  const stale = payload.version !== payload._latest
  const uid = useId()
  const srcId = `hwb-src-${uid}`
  const gvId = `hwb-gv-${uid}`
  const render = async () => {
    setBusy(true)
    setErr('')
    try {
      const body = { design, pots: ps.positions || {}, source, seconds: 4 }
      if (guitar) body.guitar_volume = gv
      const r = await rest('/audition', { method: 'POST', body })
      onRendered(r)
    } catch (e) {
      setErr(friendly(e))
    } finally {
      setBusy(false)
    }
  }
  const knobs = (payload.pots || []).map(p => `${p.label || p.name} ${Math.round((ps.positions[p.name] ?? p.pos) * 100)}%`).join(', ')
  return h(
    Card,
    { label: 'Render a take', className },
    h(
      'div',
      { className: 'hwb-col', style: { gap: 12 } },
      h(
        'div',
        { className: 'hwb-field' },
        h('span', { className: 'hwb-field-k', id: srcId }, 'Source'),
        h(
          Select,
          { value: source, onValueChange: setSource },
          h(SelectTrigger, { 'aria-labelledby': srcId, style: { width: '100%', maxWidth: 260 } }, h(SelectValue, null)),
          h(SelectContent, null, SOURCES.map(([id, label]) => h(SelectItem, { key: id, value: id }, label)))
        )
      ),
      guitar &&
        h(
          'div',
          { className: 'hwb-field' },
          h('div', { className: 'hwb-row' }, h('label', { htmlFor: gvId, className: 'hwb-grow' }, 'Guitar volume'), h('span', { className: 'hwb-muted', 'aria-hidden': 'true', style: { fontVariantNumeric: 'tabular-nums' } }, `${Math.round(gv * 100)}%`)),
          h('input', { id: gvId, className: 'hwb-range', type: 'range', min: 0, max: 100, step: 5, value: Math.round(gv * 100), 'aria-valuetext': `${Math.round(gv * 100)}%`, onChange: ev => setGv(Number(ev.target.value) / 100) }),
          h('span', { className: 'hwb-sub' }, 'The guitar’s own volume knob, modelled with a single-coil pickup. Roll it back to hear a fuzz clean up.')
        ),
      knobs && h('div', { className: 'hwb-sub' }, `Knobs: ${knobs}`),
      h(
        'div',
        { className: 'hwb-row hwb-wrap' },
        h(Button, { size: 'sm', onClick: render, disabled: busy || stale }, busy ? h(Spinner) : h(Icon, { name: 'wand' }), busy ? 'Rendering…' : 'Render 4 s with these settings')
      ),
      busy && h('div', { className: 'hwb-sub' }, 'Simulating every sample of the circuit. This takes a few seconds.'),
      stale && h('div', { className: 'hwb-sub' }, `Renders use the latest version (v${payload._latest}).`),
      err && h(ErrorLine, { message: err })
    )
  )
}

function ListenTab({ design, payload, latestAudio, ps, dryPath, refresh, wide }) {
  const auds = payload.auditions && payload.auditions.length ? payload.auditions : latestAudio ? [latestAudio] : []
  const [pickTag, setPickTag] = useState(null)
  const a = auds.find(x => x.tag === pickTag) || auds[0]
  const onRendered = r => {
    setPickTag(null)
    refresh()
    void r
  }
  const player = h(
    Card,
    {
      label: 'A/B listen',
      className: 'hwb-o1',
      right:
        auds.length > 1
          ? h(
              Select,
              { value: a ? a.tag : '', onValueChange: setPickTag },
              h(SelectTrigger, { 'aria-label': 'Take', size: 'sm', style: { maxWidth: 300 } }, h(SelectValue, null)),
              h(SelectContent, null, auds.map(x => h(SelectItem, { key: x.tag, value: x.tag }, takeLabel(x, payload))))
            )
          : a
            ? h('span', { className: 'hwb-sub' }, takeLabel(a, payload))
            : null
    },
    a
      ? h(ABPlayer, {
          abKey: `${design}:${a.wet}`,
          wetPath: a.wet,
          dryPath: a.dry,
          right: typeof a.gain_db_rms === 'number' ? `${a.gain_db_rms > 0 ? '+' : ''}${a.gain_db_rms.toFixed(1)} dB louder before matching` : null
        })
      : h('div', { className: 'hwb-sub', style: { padding: '8px 0' } }, 'No take rendered yet. Pick a source and render one, or press Play live on the knobs to hear the circuit run in real time.')
  )
  const spectrum = a && a.spectrum ? h(SpectrumCard, { path: a.spectrum, className: 'hwb-o4' }) : null
  const knobs = h(KnobsCard, { design, payload, ps, dryPath, compact: !wide, className: 'hwb-o2' })
  const renderCard = h(RenderCard, { design, payload, ps, onRendered, className: 'hwb-o3' })
  return h('div', { className: 'hwb-grid' }, h('div', { className: 'hwb-col' }, player, spectrum), h('div', { className: 'hwb-col' }, knobs, renderCard))
}

// -------------------------------------------------------------------- parts, board, netlist
function PartsTab({ design, version }) {
  const q = useQuery({ queryKey: ['hotwire', 'bom', design, version], queryFn: () => rest(`/bom/${encodeURIComponent(design)}?v=${version}`), staleTime: 60000 })
  if (q.isLoading) return h(Skel, { rows: 4 })
  if (q.error) return h(ErrorLine, { message: friendly(q.error), onRetry: () => q.refetch() })
  const rows = (q.data && q.data.rows) || []
  return h(
    Card,
    { label: 'Parts', right: h('span', { className: 'hwb-sub' }, sentence(q.data.summary || '')) },
    h(
      'table',
      { className: 'hwb-table hwb-left' },
      h('thead', null, h('tr', null, h('th', null, 'Qty'), h('th', null, 'Value'), h('th', null, 'Refs'), h('th', null, 'Part'))),
      h(
        'tbody',
        null,
        rows.map((r, i) =>
          h(
            'tr',
            { key: i },
            h('td', { style: { fontVariantNumeric: 'tabular-nums' } }, String(r.qty ?? 1)),
            h('td', null, h('div', null, tidy(String(r.snapped || r.value || ''))), r.snapped && r.value && String(r.snapped) !== String(r.value) && h('div', { className: 'hwb-why' }, `Designed as ${tidy(r.value)}`)),
            h('td', { className: 'hwb-mono hwb-muted' }, (Array.isArray(r.ref) ? r.ref.join(', ') : String(r.ref || r.refs || '')).replace(/\s*,\s*/g, ', ')),
            h('td', null, h('div', null, tidy(String(r.part || r.package || ''))), (r.note || r.rating) && h('div', { className: 'hwb-why' }, sentence([r.rating, r.note].filter(Boolean).join(' · '))))
          )
        )
      )
    )
  )
}

function stepsKey(design, version) {
  return `steps:${design}@${version}`
}

function BuildSteps({ design, version, steps }) {
  const k = stepsKey(design, version)
  const [done, setDone] = useState(() => {
    try {
      const v = CTX && CTX.storage ? CTX.storage.get(k, []) : []
      return Array.isArray(v) ? v : []
    } catch (_) {
      return []
    }
  })
  useEffect(() => {
    try {
      const v = CTX && CTX.storage ? CTX.storage.get(k, []) : []
      setDone(Array.isArray(v) ? v : [])
    } catch (_) {
      setDone([])
    }
  }, [k])
  const save = next => {
    setDone(next)
    try {
      if (CTX && CTX.storage) CTX.storage.set(k, next)
    } catch (_) {
      /* not persisted */
    }
  }
  const toggle = i => save(done.includes(i) ? done.filter(x => x !== i) : done.concat([i]).sort((a, b) => a - b))
  const n = steps.length
  const nDone = done.filter(i => i < n).length
  return h(
    Card,
    {
      label: 'Build steps',
      right: h(
        'span',
        { className: 'hwb-row' },
        h('span', { className: 'hwb-sub', style: { fontVariantNumeric: 'tabular-nums' } }, `${nDone} of ${n} done`),
        h('span', { className: 'hwb-progress', 'aria-hidden': 'true' }, h('i', { style: { width: `${n ? (100 * nDone) / n : 0}%` } })),
        nDone > 0 && h(Button, { size: 'xs', variant: 'ghost', onClick: () => save([]) }, 'Clear')
      )
    },
    h(
      'ol',
      { className: 'hwb-checks' },
      steps.map((s, i) => {
        const raw = typeof s === 'string' ? s : s.text || s.step || JSON.stringify(s)
        const text = sentence(String(raw).replace(/^\s*\d+[.)]\s*/, ''))
        const on = done.includes(i)
        return h(
          'li',
          { key: i, className: 'hwb-check' + (on ? ' done' : '') },
          h('button', { type: 'button', className: 'hwb-tick', role: 'checkbox', 'aria-checked': on ? 'true' : 'false', 'aria-label': `Step ${i + 1} done`, onClick: () => toggle(i) }, h(Icon, { name: 'check', size: 13 })),
          h('span', { className: 'hwb-check-n' }, String(i + 1)),
          h('span', { className: 'hwb-check-t' }, text)
        )
      })
    )
  )
}

function BoardTab({ design, payload, refresh }) {
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const svg = payload.svg && payload.svg.stripboard
  const build = async () => {
    setBusy(true)
    setErr('')
    try {
      await rest('/build', { method: 'POST', body: { design } })
      refresh()
    } catch (e) {
      setErr(friendly(e))
    } finally {
      setBusy(false)
    }
  }
  const steps = (payload.build && payload.build.steps) || []
  return h(
    'div',
    { className: 'hwb-col' },
    h(
      Card,
      { label: 'Stripboard', className: 'hwb-board', right: svg && h(ZoomButton, { title: `Stripboard · ${payload.title || ''}`, svg }) },
      svg
        ? h(Svg, { svg, label: 'Stripboard layout' })
        : h(
            'div',
            { className: 'hwb-col', style: { gap: 10 } },
            h('div', { className: 'hwb-sub' }, 'No board layout for this version yet.'),
            h('div', null, h(Button, { size: 'sm', onClick: build, disabled: busy || payload.version !== payload._latest }, busy ? h(Spinner) : h(Icon, { name: 'circuit-board' }), 'Lay it out on stripboard'))
          ),
      err && h('div', { style: { marginTop: 8 } }, h(ErrorLine, { message: err }))
    ),
    steps.length > 0 && h(BuildSteps, { design, version: payload.version, steps })
  )
}

function netlistLine(line, i) {
  const t = line
  let kids
  if (/^\s*\*@/.test(t)) {
    const m = t.match(/^(\s*\*@\s*\S*)(.*)$/)
    kids = [h('span', { key: 'a', className: 'hwb-tok-a' }, m[1]), m[2]]
  } else if (/^\s*\*/.test(t) || /^\s*;/.test(t)) {
    kids = [h('span', { key: 'c', className: 'hwb-tok-c' }, t)]
  } else if (/^\s*\./.test(t)) {
    const m = t.match(/^(\s*\.\S*)(.*)$/)
    kids = [h('span', { key: 'd', className: 'hwb-tok-d' }, m[1]), m[2]]
  } else if (/^\s*[A-Za-z]/.test(t)) {
    const m = t.match(/^(\s*\S+)(.*)$/)
    kids = [h('span', { key: 'n', className: 'hwb-tok-n' }, m[1]), m[2]]
  } else kids = [t || ' ']
  return h('div', { key: i, className: 'hwb-code-l' }, h('span', null, kids))
}

function NetlistTab({ design, payload }) {
  const [copied, setCopied] = useState(false)
  const text = payload.netlist || ''
  const name = `${design}-v${payload.version}.cir`
  const copy = async () => {
    const ok = await copyText(text)
    setCopied(ok)
    if (!ok) toast('error', 'Could not reach the clipboard.')
    if (CTX) CTX.setTimeout(() => setCopied(false), 1500)
  }
  const download = async () => {
    const path = payload.dir ? `${payload.dir}/netlist.cir` : ''
    if (path && typeof sdk.captureGatewayFileDownload === 'function') {
      try {
        const dl = sdk.captureGatewayFileDownload()
        const r = await dl(path, name)
        if (r !== false) return
      } catch (_) {
        /* fall back to a local blob */
      }
    }
    try {
      const url = URL.createObjectURL(new Blob([text], { type: 'text/plain' }))
      const a = document.createElement('a')
      a.href = url
      a.download = name
      a.click()
      if (CTX) CTX.setTimeout(() => URL.revokeObjectURL(url), 4000)
    } catch (e) {
      toast('error', friendly(e))
    }
  }
  const lines = useMemo(() => text.replace(/\s+$/, '').split(/\r?\n/), [text])
  return h(
    Card,
    {
      label: 'Netlist',
      right: h(
        'span',
        { className: 'hwb-row' },
        h('span', { className: 'hwb-sub' }, `${plural(lines.length, 'line')} · v${payload.version}`),
        h(Button, { size: 'xs', variant: 'ghost', onClick: copy }, h(Icon, { name: copied ? 'check' : 'copy' }), copied ? 'Copied' : 'Copy'),
        h(Button, { size: 'xs', variant: 'ghost', onClick: download }, h(Icon, { name: 'desktop-download' }), 'Download')
      )
    },
    h('div', { className: 'hwb-code', tabIndex: 0, role: 'region', 'aria-label': 'SPICE netlist' }, lines.map(netlistLine))
  )
}

function Skel({ rows }) {
  const out = []
  for (let i = 0; i < (rows || 3); i++) out.push(h(Skeleton, { key: i, style: { height: i === 0 ? 18 : 64, width: i === 0 ? '45%' : '100%' } }))
  return h('div', { className: 'hwb-sk', 'aria-busy': 'true' }, out)
}

// -------------------------------------------------------------------- library + empty state
function LibThumb({ id }) {
  const q = useQuery({ queryKey: ['hotwire', 'libsvg', id], queryFn: () => rest(`/library/${encodeURIComponent(id)}/schematic`), staleTime: Infinity, retry: 1 })
  if (q.data && q.data.svg) return h('div', { className: 'hwb-thumb', 'aria-hidden': 'true' }, h(Svg, { svg: q.data.svg, label: '' }))
  if (q.error) return h('div', { className: 'hwb-thumb', 'aria-hidden': 'true' }, h(Icon, { name: 'circuit-board', size: 28 }))
  return h('div', { className: 'hwb-thumb', 'aria-hidden': 'true' }, h(Skeleton, { style: { height: 80, width: '80%' } }))
}

function Library({ compact, onOpened, limit }) {
  const q = useQuery({ queryKey: ['hotwire', 'library'], queryFn: () => rest('/library'), staleTime: 300000 })
  const [busy, setBusy] = useState('')
  const [err, setErr] = useState('')
  const open = async id => {
    setBusy(id)
    setErr('')
    try {
      const r = await rest('/open', { method: 'POST', body: { library: id } })
      await queryClient.invalidateQueries({ queryKey: ['hotwire', 'designs'] })
      $selected.set({ design: r.design, version: null })
      $showLib.set(false)
      if (onOpened) onOpened()
    } catch (e) {
      setErr(friendly(e))
    } finally {
      setBusy('')
    }
  }
  if (q.isLoading) return h(Skel, { rows: 3 })
  if (q.error) return h(ErrorLine, { message: friendly(q.error), onRetry: () => q.refetch() })
  const items = ((q.data && q.data.circuits) || []).slice(0, limit || 60)
  return h(
    'div',
    { className: 'hwb-col', style: { gap: 8 } },
    h(
      'div',
      { className: 'hwb-lib', style: compact ? { gridTemplateColumns: 'repeat(auto-fill,minmax(180px,1fr))' } : null },
      items.map(c =>
        h(
          'button',
          { key: c.id, type: 'button', className: 'hwb-libcard', onClick: () => open(c.id), disabled: !!busy, 'aria-busy': busy === c.id ? 'true' : 'false', 'aria-label': `Open ${c.title}`, title: c.description || c.title },
          h(LibThumb, { id: c.id }),
          h(
            'div',
            { className: 'hwb-libtext' },
            h('b', null, busy === c.id ? 'Opening…' : c.title || c.id),
            c.description && h('span', { className: 'hwb-libdesc' }, tidy(c.description)),
            c.category && h('span', { className: 'hwb-chip', style: { marginTop: 4 } }, String(c.category).replace(/[_-]/g, ' '))
          )
        )
      )
    ),
    err && h(ErrorLine, { message: err })
  )
}

const PROMPTS = [
  'Design a 9 V fuzz that cleans up when I roll back the guitar volume, and let me hear it',
  'Make a clean JFET boost with about 15 dB of gain and check it never clips at 1 V in',
  'Build a 1 kHz Sallen-Key low-pass for a 9 V supply and show me the Bode plot'
]

function EmptyBench({ wide }) {
  const [sent, setSent] = useState(-1)
  const ask = async (p, i) => {
    const ok = await askHermes(p)
    setSent(ok ? i : -1)
  }
  return h(
    'div',
    { className: 'hwb-col', style: { gap: 18 } },
    h(
      'div',
      { className: 'hwb-empty' },
      h('div', { className: 'hwb-empty-icon', 'aria-hidden': 'true' }, h(Icon, { name: 'circuit-board', size: 22 })),
      h('h2', null, 'No circuit on the bench yet'),
      h('p', null, 'Describe a pedal, a filter or an amplifier in a chat. Hermes writes the SPICE netlist, proves it against measurable specs, and puts it here: the schematic, a verdict, knobs you can turn while it plays, and a stripboard you can build.'),
      h(
        'div',
        { className: 'hwb-prompts', role: 'list', 'aria-label': 'Things to ask Hermes' },
        PROMPTS.map((p, i) =>
          h('button', { key: i, type: 'button', role: 'listitem', className: 'hwb-prompt', onClick: () => ask(p, i), title: 'Put this prompt in the chat composer' }, h(Icon, { name: sent === i ? 'check' : 'comment-discussion' }), h('span', null, `“${p}”`), h(Icon, { name: 'arrow-right' }))
        )
      ),
      wide &&
        h(
          'div',
          { className: 'hwb-steps3' },
          h('div', null, h('b', null, '1 · Describe'), 'Plain English: what it should sound like, the supply, the targets.'),
          h('div', null, h('b', null, '2 · Prove'), 'A built-in simulator checks every spec and every part’s ratings.'),
          h('div', null, h('b', null, '3 · Hear and build'), 'Turn the knobs live, A/B the take, follow the stripboard steps.')
        )
    ),
    h('h3', { className: 'hwb-section' }, 'Or start from a reference circuit'),
    h(Library, { compact: !wide, limit: wide ? 60 : 6 })
  )
}

function LibraryView({ wide, count }) {
  return h(
    'div',
    { className: 'hwb-col', style: { gap: 16 } },
    h(
      'div',
      { className: 'hwb-head' },
      h(
        'div',
        { className: 'hwb-grow' },
        h('h2', { className: 'hwb-title' }, 'Reference circuits'),
        h('div', { className: 'hwb-sub', style: { marginTop: 2 } }, `${count ? `${count} circuits` : 'Circuits'} that pass their own specs, each cross-checked in ngspice. Open one to put a copy on the bench as a new design.`)
      ),
      h(Button, { size: 'sm', variant: 'ghost', onClick: () => $showLib.set(false) }, h(Icon, { name: 'arrow-left' }), 'Back to the bench')
    ),
    h(Library, { compact: !wide })
  )
}

// -------------------------------------------------------------------- bench root
function useFollow(designs) {
  useEffect(() => {
    if (!designs || !designs.length) return
    const newest = designs[0]
    const sel = $selected.get()
    const stamp = Number(newest.updated || 0)
    if (!sel.design || !designs.some(d => d.id === sel.design)) {
      $selected.set({ design: newest.id, version: null })
      lastSeenUpdate = stamp
      return
    }
    if (stamp > lastSeenUpdate) {
      if (lastSeenUpdate && $follow.get() && (sel.design !== newest.id || sel.version !== null)) $selected.set({ design: newest.id, version: null })
      lastSeenUpdate = stamp
    }
  }, [designs])
}

function verdictTone(v) {
  return v === 'pass' ? 'pass' : v === 'fail' ? 'fail' : v ? 'warn' : ''
}

function DesignPicker({ designs, selected, wide }) {
  const labels = useMemo(() => pickerLabels(designs), [designs])
  if (!designs || designs.length < 2) return null
  const cur = designs.find(d => d.id === selected)
  const curLabel = (cur && labels.get(cur.id)) || { title: designTitle(cur), suffix: '' }
  return h(
    Select,
    { value: selected || '', onValueChange: v => $selected.set({ design: v, version: null }) },
    h(
      SelectTrigger,
      { 'aria-label': 'Design', style: wide ? { width: 280, maxWidth: 320 } : { flex: 1, minWidth: 0 } },
      h(
        SelectValue,
        null,
        cur &&
          h(
            'span',
            { className: 'hwb-opt' },
            h('span', { className: 'hwb-dot ' + verdictTone(cur.verdict), 'aria-hidden': 'true' }),
            h('span', { className: 'hwb-opt-t' }, curLabel.title),
            curLabel.suffix && h('span', { className: 'hwb-opt-s' }, `· ${curLabel.suffix}`)
          )
      )
    ),
    h(
      SelectContent,
      { style: { maxWidth: 420 } },
      designs.map(d => {
        const l = labels.get(d.id) || { title: designTitle(d), suffix: '' }
        return h(
          SelectItem,
          { key: d.id, value: d.id, textValue: l.title },
          h(
            'span',
            { className: 'hwb-opt' },
            h('span', { className: 'hwb-dot ' + verdictTone(d.verdict), 'aria-label': verdictInfo(d.verdict).text, role: 'img' }),
            h('span', { className: 'hwb-opt-t' }, l.title),
            l.suffix && h('span', { className: 'hwb-opt-s' }, `· ${l.suffix}`),
            h('span', { className: 'hwb-opt-d' }, ago(d.updated || d.created))
          )
        )
      })
    )
  )
}

function BenchRoot({ wide }) {
  const dark = useDark()
  return h('div', { className: 'hwb hwb-root' + (wide ? ' hwb-page' : '') + (dark ? ' hwb-dark' : '') }, h(Style), h(Bench, { wide }))
}

function Bench({ wide }) {
  const sel = useValue($selected)
  const follow = useValue($follow)
  const tab = useValue($tab)
  const showLib = useValue($showLib)
  const list = useQuery({ queryKey: ['hotwire', 'designs'], queryFn: () => rest('/designs?limit=60'), refetchInterval: 4000, refetchIntervalInBackground: false })
  const lib = useQuery({ queryKey: ['hotwire', 'library'], queryFn: () => rest('/library'), staleTime: 300000, enabled: showLib })
  const designs = list.data && list.data.designs
  useFollow(designs)
  const row = designs && designs.find(d => d.id === sel.design)
  const detail = useQuery({
    queryKey: ['hotwire', 'design', sel.design, row ? row.updated : 0, row ? row.latest : 0],
    queryFn: () => rest(`/designs/${encodeURIComponent(sel.design)}`),
    enabled: !!sel.design && !!row,
    placeholderData: prev => (prev && prev.id === sel.design ? prev : undefined)
  })
  const d = detail.data
  const wantV = sel.version
  const verQ = useQuery({
    queryKey: ['hotwire', 'v', sel.design, wantV],
    queryFn: () => rest(`/designs/${encodeURIComponent(sel.design)}/v/${wantV}`),
    enabled: !!sel.design && wantV !== null && !!d && d.current && d.current.version !== wantV
  })
  const refresh = () => queryClient.invalidateQueries({ queryKey: ['hotwire'] })
  const has = !!(designs && designs.length)

  const controls = has
    ? h(
        'div',
        { className: 'hwb-row', style: { justifyContent: 'flex-end', gap: 4, width: wide ? 'auto' : '100%' } },
        h(DesignPicker, { designs, selected: sel.design, wide }),
        h(
          Tip,
          { label: follow ? 'Following the agent: new versions open here as they land' : 'Not following: the bench stays on this design' },
          h(Button, { size: 'icon-xs', variant: follow ? 'secondary' : 'ghost', onClick: () => $follow.set(!follow), 'aria-pressed': follow, 'aria-label': 'Follow the agent' }, h(Icon, { name: follow ? 'pinned' : 'pin' }))
        ),
        h(Tip, { label: showLib ? 'Back to the bench' : 'Reference circuits' }, h(Button, { size: 'icon-xs', variant: showLib ? 'secondary' : 'ghost', onClick: () => $showLib.set(!showLib), 'aria-pressed': showLib, 'aria-label': 'Reference circuits' }, h(Icon, { name: 'library' })))
      )
    : null

  let content
  if (list.isLoading) content = h(Skel, { rows: 5 })
  else if (list.error) content = h(ErrorLine, { message: friendly(list.error), onRetry: () => list.refetch() })
  else if (!has) content = h(EmptyBench, { wide })
  else if (showLib) content = h('div', { className: 'hwb-col' }, !wide && controls, h(LibraryView, { wide, count: lib.data && lib.data.circuits ? lib.data.circuits.length : 0 }))
  else if (detail.isLoading || !d) content = detail.error ? h(ErrorLine, { message: friendly(detail.error), onRetry: () => detail.refetch() }) : h(Skel, { rows: 5 })
  else {
    const payload0 = wantV !== null && d.current.version !== wantV ? verQ.data : d.current
    if (!payload0) content = verQ.error ? h(ErrorLine, { message: friendly(verQ.error) }) : h(Skel, { rows: 5 })
    else content = h(DesignView, { d, row, payload: Object.assign({}, payload0, { _latest: d.latest }), wide, tab, refresh, controls })
  }
  const page = wide && showLib && has ? h('div', { className: 'hwb-row', style: { justifyContent: 'flex-end' } }, controls) : null
  return h('div', { className: 'hwb-scroll' }, h('div', { className: 'hwb-body' }, page, content))
}

function DesignView({ d, row, payload, wide, tab, refresh, controls }) {
  const design = d.id
  const ps = usePotState(design, payload.version, payload.pots)
  // a knob re-check counts only while the knobs are away from the saved positions
  const chk = ps.moved && ps.check.result ? ps.check.result : payload.check
  const base = payload.check || {}
  const specs = (chk && chk.specs) || base.specs || []
  const bode = (chk && chk.svg && chk.svg.bode) || (payload.svg && payload.svg.bode)
  const outdated = specs.filter(s => s.outdated)
  const counted = specs.filter(s => !s.outdated)
  const nPass = counted.filter(s => s.pass).length
  let verdict = (chk && chk.verdict) || base.verdict
  if (outdated.length && verdict === 'fail' && counted.every(s => s.pass)) verdict = 'recheck'
  const dryPath = (payload.auditions && payload.auditions[0] && payload.auditions[0].dry) || (d.latest_audition && d.latest_audition.dry) || null
  const pick = v => $selected.set({ design, version: v === d.current.version ? null : v })
  const vrow = d.versions.find(v => v.v === payload.version) || {}
  const title = designTitle(Object.assign({}, row || {}, { id: d.id, name: d.name, title: d.title || payload.title }))
  const circuit = base.title && base.title !== title && !/^untitled circuit$/i.test(base.title) && base.title.length > 2 ? base.title : ''
  const note = tidyNote(payload.note)

  const sub = [
    `v${payload.version}${payload.version !== d.latest ? ` of ${d.latest}` : ''}`,
    circuit,
    vrow.time ? ago(vrow.time) : '',
    base.supply_ma !== null && base.supply_ma !== undefined ? `Draws ${fmtNum(base.supply_ma / 1000, 'A')}` : ''
  ].filter(Boolean)
  const head = h(
    'div',
    { className: 'hwb-head' },
    h(
      'div',
      { className: 'hwb-grow' },
      h('h2', { className: 'hwb-title' }, title),
      h('div', { className: 'hwb-sub', style: { marginTop: 2 } }, sub.join(' · ')),
      note && h('p', { className: 'hwb-quote', title: 'The agent’s note for this version' }, note)
    ),
    h(
      'div',
      { className: 'hwb-col hwb-head-side', style: { gap: 8, alignItems: 'flex-end', flex: 'none' } },
      wide && controls,
      h('div', { className: 'hwb-row' }, counted.length > 0 && h('span', { className: 'hwb-sub', style: { whiteSpace: 'nowrap' } }, `${nPass} of ${plural(counted.length, 'spec')} pass`), h(VerdictBadge, { verdict }))
    )
  )
  const timeline = d.versions.length > 1 && h(Timeline, { versions: d.versions, current: payload.version, onPick: pick })
  const tabs = h(
    Tabs,
    { value: tab, onValueChange: v => $tab.set(v) },
    h(
      TabsList,
      { 'aria-label': 'Bench sections', style: { alignSelf: 'flex-start' } },
      h(TabsTrigger, { value: 'bench' }, 'Bench'),
      h(TabsTrigger, { value: 'listen' }, 'Listen'),
      h(TabsTrigger, { value: 'parts' }, 'Parts'),
      h(TabsTrigger, { value: 'board' }, 'Board'),
      h(TabsTrigger, { value: 'netlist' }, 'Netlist')
    )
  )
  const knobs = h(KnobsCard, { design, payload, ps, dryPath, compact: !wide, className: 'hwb-o1' })
  const verdictCard = h(
    Card,
    { label: 'Verdict', className: 'hwb-o2', right: ps.moved ? h('span', { className: 'hwb-sub' }, 'At these knob settings') : null },
    outdated.length > 0 && !ps.moved && h(RecheckNote, { design, version: payload.version, count: outdated.length }),
    payload.check || ps.check.result ? h(SpecTable, { specs }) : h(RecheckNote, { design, version: payload.version, count: 0, never: true }),
    !specs.length && base.summary && h('div', { className: 'hwb-sub', style: { marginTop: 6 } }, sentence(base.summary))
  )
  const schSvg = payload.svg && payload.svg.schematic
  const schematic = schSvg ? h(Card, { label: 'Schematic', className: 'hwb-o3', right: h(ZoomButton, { title: `Schematic · ${title}`, svg: schSvg }) }, h(Svg, { svg: schSvg, label: `Schematic of ${payload.title}`, stripTitle: payload.title })) : null
  const bodeCard = bode ? h(Card, { label: 'Frequency response', className: 'hwb-o4', right: h(ZoomButton, { title: `Frequency response · ${title}`, svg: bode }) }, h(Svg, { svg: bode, label: 'Bode plot', stripTitle: payload.title })) : null
  const stress = h(StressCard, { check: chk || payload.check, className: 'hwb-o5' })
  let body
  if (tab === 'listen') body = h(ListenTab, { design, payload, latestAudio: d.latest_audition, ps, dryPath, refresh, wide })
  else if (tab === 'parts') body = h(PartsTab, { design, version: payload.version })
  else if (tab === 'board') body = h(BoardTab, { design, payload, refresh })
  else if (tab === 'netlist') body = h(NetlistTab, { design, payload })
  else if (wide && !schematic && !bodeCard) body = h('div', { className: 'hwb-col', style: { maxWidth: 760 } }, knobs, verdictCard, stress)
  else if (wide) body = h('div', { className: 'hwb-grid' }, h('div', { className: 'hwb-col' }, schematic, bodeCard), h('div', { className: 'hwb-col' }, knobs, verdictCard, stress))
  else body = h('div', { className: 'hwb-col' }, knobs, verdictCard, schematic, bodeCard, stress)
  return h('div', { className: 'hwb-col' }, !wide && controls, head, timeline, tabs, body)
}

// -------------------------------------------------------------------- tool media renderers
function useSvgFromFile(file) {
  const [state, setState] = useState({ svg: null, error: '' })
  useEffect(() => {
    let alive = true
    file
      .resolveUrl()
      .then(svgText)
      .then(t => alive && setState({ svg: t, error: '' }))
      .catch(e => alive && setState({ svg: null, error: friendly(e) }))
    return () => {
      alive = false
    }
  }, [file && file.path])
  return state
}

function MediaFrame({ children }) {
  const dark = useDark()
  return h('div', { className: 'hwb hwb-chat' + (dark ? ' hwb-dark' : '') }, h(Style), children)
}

function OpenInBench({ design, version }) {
  if (!design) return null
  return h(Button, { size: 'xs', variant: 'ghost', onClick: () => openBench(design, version) }, h(Icon, { name: 'circuit-board' }), 'Open in Bench')
}

function SvgCard({ file, label, top, design, version, zoomTitle }) {
  const s = useSvgFromFile(file)
  if (s.error) return null
  const topNode = typeof top === 'function' ? top(s.svg) : top
  return h(
    MediaFrame,
    null,
    h(
      Card,
      {
        label,
        right: h('span', { className: 'hwb-row' }, s.svg && h(ZoomButton, { title: zoomTitle || label, svg: s.svg }), h(OpenInBench, { design, version }))
      },
      topNode,
      s.svg ? h(Svg, { svg: s.svg, label }) : h(Skeleton, { style: { height: 160, width: '100%' } })
    )
  )
}

function CheckMedia(props) {
  const { file, fallback } = props
  const r = parseResult(props.result)
  const name = (file && file.name) || ''
  if (!file || !/\.svg$/i.test(name)) return fallback
  const specs = r.specs || []
  const where = `v${r.version || '?'}`
  if (/schematic/i.test(name)) {
    const nPass = specs.filter(s => s.pass).length
    const failed = specs.filter(s => !s.pass)
    const top = h(
      'div',
      { className: 'hwb-col', style: { gap: 6, marginBottom: 10 } },
      h(
        'div',
        { className: 'hwb-row' },
        h(VerdictBadge, { verdict: r.verdict }),
        h('span', { className: 'hwb-sub hwb-grow' }, specs.length > 0 ? `${nPass} of ${plural(specs.length, 'spec')} pass · ${where}` : where)
      ),
      failed.slice(0, 3).map((s, i) => h('div', { key: i, className: 'hwb-why' }, h(Codicon, { name: 'error', size: 11, className: 'hwb-lv-error' }), ' ', sentence(s.explanation || specLabel(s))))
    )
    return h(SvgCard, { file, label: 'Schematic', top, design: r.design, version: r.version, zoomTitle: `Schematic · ${r.title || ''}` })
  }
  const label = /bode|ac/i.test(name) ? 'Frequency response' : /spectrum/i.test(name) ? 'Spectrum' : /tran|scope/i.test(name) ? 'Scope' : 'Plot'
  return h(SvgCard, { file, label, design: r.design, version: r.version, zoomTitle: `${label} · ${r.title || ''}` })
}

function AuditionMedia(props) {
  const { file, fallback } = props
  const r = parseResult(props.result)
  const path = (file && file.path) || ''
  if (r.dry && path === r.dry) return h('span', { hidden: true }) // played inside the A/B card above
  if (/\.svg$/i.test(path)) return h(SvgCard, { file, label: 'Spectrum', design: r.design, version: r.version, zoomTitle: `Spectrum · ${r.title || ''}` })
  if (!r.wet || !r.dry || path !== r.wet || file.kind !== 'audio') return fallback
  return h(AuditionCard, { file, r })
}

function AuditionCard({ file, r }) {
  const [url, setUrl] = useState(null)
  useEffect(() => {
    let alive = true
    file.resolveUrl().then(u => alive && setUrl(u), () => alive && setUrl(''))
    return () => {
      alive = false
    }
  }, [file.path])
  const bits = [sourceName(r.source || 'audition')]
  if (r.guitar && r.guitar.volume !== undefined && r.guitar.volume !== null && Number(r.guitar.volume) !== 1) bits.push(`guitar volume ${Math.round(Number(r.guitar.volume) * 100)}%`)
  if (r.seconds) bits.push(`${r.seconds} s`)
  return h(
    MediaFrame,
    null,
    h(
      Card,
      { label: 'A/B listen', right: h(OpenInBench, { design: r.design, version: r.version }) },
      h('div', { className: 'hwb-row', style: { marginBottom: 10 } }, h('b', { className: 'hwb-grow', style: { fontSize: 13 } }, r.title || 'Circuit'), h('span', { className: 'hwb-sub' }, bits.join(' · '))),
      url === null
        ? h(Skeleton, { style: { height: 128, width: '100%' } })
        : h(ABPlayer, {
            abKey: `chat:${r.wet}`,
            wetPath: r.wet,
            dryPath: r.dry,
            wetUrl: url || null,
            compact: true,
            right: typeof r.gain_db_rms === 'number' ? `${r.gain_db_rms > 0 ? '+' : ''}${Number(r.gain_db_rms).toFixed(1)} dB before matching` : null
          })
    )
  )
}

// The counts the drawing itself prints, so the card and the board agree.
function boardCounts(r, svg) {
  const info = r.board || {}
  const sub = svg ? (svg.match(/Stripboard [^<]*?holes[^<]*/) || [''])[0] : ''
  const num = re => {
    const m = sub.match(re)
    return m ? Number(m[1]) : null
  }
  const cuts = num(/(\d+) track cuts?/) ?? (info.cuts !== undefined ? info.cuts : null)
  const links = num(/(\d+) (?:wire )?links?/) ?? (info.links !== undefined ? info.links : null)
  const pads = num(/(\d+) off-board pads?/) ?? (svg ? (svg.match(/<title>[^<]* pad \(/g) || []).length || null : null)
  const bits = []
  if (Array.isArray(r.steps)) bits.push(plural(r.steps.length, 'build step'))
  if (cuts !== null) bits.push(plural(cuts, 'track cut'))
  if (links !== null) bits.push(plural(links, 'wire link'))
  if (pads !== null) bits.push(plural(pads, 'off-board pad'))
  return bits.join(' · ')
}

function BuildMedia(props) {
  const { file, fallback } = props
  const r = parseResult(props.result)
  if (!file || !/\.svg$/i.test(file.name || '')) return fallback
  const top = svg => {
    const t = boardCounts(r, svg)
    return t ? h('div', { className: 'hwb-sub', style: { marginBottom: 8 } }, t) : null
  }
  return h(SvgCard, { file, label: 'Stripboard', top, design: r.design, version: r.version, zoomTitle: `Stripboard · ${r.title || ''}` })
}

// -------------------------------------------------------------------- ::hotwire-knobs directive
function KnobsDirective({ attrs }) {
  const design = attrs && typeof attrs.design === 'string' && /^[a-z0-9-]{1,48}$/.test(attrs.design) ? attrs.design : null
  const q = useQuery({ queryKey: ['hotwire', 'directive', design], queryFn: () => rest(`/designs/${design}`), enabled: !!design, staleTime: 10000 })
  if (!design) return null
  if (q.isLoading) return h(MediaFrame, null, h(Card, { label: 'Knobs' }, h(Skel, { rows: 2 })))
  if (q.error) return h(MediaFrame, null, h(Card, { label: 'Knobs' }, h(ErrorLine, { message: friendly(q.error) })))
  const d = q.data
  return h(MediaFrame, null, h(DirectiveBody, { d }))
}

function DirectiveBody({ d }) {
  const payload = d.current
  const ps = usePotState(d.id, payload.version, payload.pots)
  const chk = (ps.moved && ps.check.result) || payload.check || {}
  const specs = chk.specs || []
  const dryPath = (d.latest_audition && d.latest_audition.dry) || null
  return h(
    Card,
    { label: `Knobs · v${payload.version}`, right: h(OpenInBench, { design: d.id }) },
    h('div', { className: 'hwb-row', style: { marginBottom: 8 } }, h('b', { className: 'hwb-grow', style: { fontSize: 13 } }, designTitle(d) || payload.title)),
    h(
      'div',
      { className: 'hwb-row hwb-wrap', style: { marginBottom: 10, gap: 6 } },
      h(VerdictBadge, { verdict: chk.verdict }),
      specs.map((s, i) => h(Tip, { key: i, label: h('span', { className: 'hwb-spec-src' }, s.spec) }, h(Badge, { variant: s.outdated ? 'muted' : s.pass ? 'success' : 'destructive' }, `${specLabel(s)}: ${specMeasured(s)}`))),
      ps.check.busy && h(Spinner)
    ),
    (payload.pots || []).length
      ? h(
          'div',
          { className: 'hwb-knobs' },
          payload.pots.map(p => h(Knob, { key: p.name, pot: p, value: ps.positions[p.name] ?? p.pos, onChange: v => ps.setPot(p.name, v), onCommit: () => ps.commit() }))
        )
      : h('div', { className: 'hwb-sub' }, 'This circuit has no knobs.'),
    h(LiveBar, { design: d.id, version: payload.version, positions: ps.positions, dryPath, keyId: ps.key, compact: true })
  )
}

// -------------------------------------------------------------------- plugin
export default {
  id: 'hotwire',
  name: 'Hotwire',
  register(ctx) {
    CTX = ctx
    lastSeenUpdate = 0
    try {
      const t = ctx.storage.get('tab', 'bench')
      if (['bench', 'listen', 'parts', 'board', 'netlist'].includes(t)) $tab.set(t)
    } catch (_) {
      /* first run */
    }
    const unTab = $tab.listen(v => {
      try {
        ctx.storage.set('tab', v)
      } catch (_) {
        /* not persisted */
      }
    })
    ctx.register({
      id: 'bench',
      area: sdk.PANES_AREA || 'panes',
      title: 'Hotwire',
      data: { placement: 'right', width: '440px' },
      render: () => h(BenchRoot, { wide: false })
    })
    const more = [
      { id: 'page', area: sdk.ROUTES_AREA, data: { path: '/hotwire' }, render: () => h(BenchRoot, { wide: true }) },
      { id: 'nav', area: sdk.SIDEBAR_NAV_AREA, data: { path: '/hotwire', label: 'Hotwire', codicon: 'circuit-board' } },
      {
        id: 'open',
        area: sdk.PALETTE_AREA,
        data: { id: 'hotwire.open', label: 'Hotwire: open bench', keywords: ['hotwire', 'circuit', 'pedal', 'spice', 'bench', 'knobs'], run: () => openBench() }
      },
      {
        id: 'stop-audio',
        area: sdk.PALETTE_AREA,
        data: { id: 'hotwire.stop', label: 'Hotwire: stop audio', keywords: ['hotwire', 'stop', 'mute', 'audio'], run: () => { liveStop(); abStop() } }
      }
    ]
    ctx.registerMany(more.filter(c => typeof c.area === 'string'))
    if (typeof sdk.TOOL_MEDIA_RENDERER_AREA === 'string') {
      ctx.register({ id: 'audition-card', area: sdk.TOOL_MEDIA_RENDERER_AREA, data: { tools: ['circuit_audition'], render: p => h(AuditionMedia, p) } })
      ctx.register({ id: 'check-card', area: sdk.TOOL_MEDIA_RENDERER_AREA, data: { tools: ['circuit_check', 'circuit_simulate', 'circuit_tune'], render: p => h(CheckMedia, p) } })
      ctx.register({ id: 'build-card', area: sdk.TOOL_MEDIA_RENDERER_AREA, data: { tools: ['circuit_build'], render: p => h(BuildMedia, p) } })
    }
    if (typeof sdk.TRANSCRIPT_DIRECTIVE_AREA === 'string') {
      ctx.register({ id: 'knobs-directive', area: sdk.TRANSCRIPT_DIRECTIVE_AREA, data: { name: 'hotwire-knobs', render: p => h(KnobsDirective, p) } })
    }
    const onDesign = ev => {
      const p = (ev && ev.payload) || {}
      queryClient.invalidateQueries({ queryKey: ['hotwire'] })
      if (p.design && $follow.get()) $selected.set({ design: p.design, version: null })
    }
    ctx.onEvent('plugin.hotwire.design.updated', onDesign)
    ctx.onEvent('plugin.hotwire.design.audition', onDesign)
    ctx.onDispose(() => {
      if (typeof unTab === 'function') unTab()
      for (const off of Array.from(loops)) off()
      liveStop()
      abStop()
      if (AC) {
        AC.close().catch(() => {})
        AC = null
      }
      if (workletUrl) {
        URL.revokeObjectURL(workletUrl)
        workletUrl = null
      }
      workletCtx = null
      bufCache.clear()
      CTX = null
    })
  }
}
