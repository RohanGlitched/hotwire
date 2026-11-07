"""Built-in part library for Hotwire.

Every part a netlist can name without a ``.model`` line lives in ``PARTS``:

    PARTS["2N3904"] = {
        "type":    "NPN",              # D | NPN | PNP | NJF | PJF | NMOS | PMOS | OPAMP
        "params":  {"IS": ..., ...},   # SPICE level-1 names, upper case, SI units
        "rating":  {"pd_w": ..., ...}, # absolute maximums used by the stress check
        "desc":    "one line for the BOM",
        "package": "TO-92",
        "pins":    "EBC",              # physical pin order (see below)
        "source":  "where the numbers came from",
    }

params
    Diodes (Shockley + series resistance + depletion capacitance):
        IS, N, RS, CJO, VJ, M.
    BJTs (Ebers-Moll transport model, the Gummel-Poon subset Hotwire simulates):
        IS, BF, BR, NF, NR, VAF, RB, RC, RE, CJE, VJE, MJE, CJC, VJC, MJC.
    JFETs (Shichman-Hodges): VTO, BETA, LAMBDA, IS, CGS, CGD, PB.
    MOSFETs (SPICE level 1): VTO, KP, LAMBDA, RD, IS (body diode), CGS/CGD (farads,
        a Hotwire convenience for CGSO*W / CGDO*W), CBD, PB, MJ.
    Op-amps (Hotwire macromodel, per amplifier):
        A0       open-loop DC gain (V/V)
        GBW      gain-bandwidth product (Hz); the single dominant pole is GBW/A0
        RIN      differential input resistance (ohm)
        ROUT     open-loop output resistance (ohm)
        VSAT_HI  output headroom below the positive rail (V)
        VSAT_LO  output headroom above the negative rail (V)
        IQ       quiescent supply current per amplifier (A), drawn VCC -> VEE

    Published manufacturer models carry Gummel-Poon terms that Hotwire does not
    simulate (ISE/NE base leakage, IKF high injection, TF/TR transit times, BV
    breakdown, temperature terms).  Those are dropped here.  Where ISE/IKF shift
    the current gain a lot, BF is replaced by the effective gain the full model
    gives at Ic = 1 mA, so DC bias and gain stay realistic.  Every such change is
    written in ``source``.  Both the Python engine and the ngspice cross-check
    (tests/test_library_ngspice.py) use exactly these numbers.

rating keys (datasheet absolute maximums, 25 C)
    pd_w    total power dissipation (W)
    vmax_v  max reverse voltage (diodes), V_DS/V_DG (JFETs), total supply (op-amps)
    imax_a  max forward/average current (diodes), gate current (JFETs),
            output short-circuit current (op-amps)
    vceo_v  max collector-emitter voltage (BJTs, magnitude)
    ic_a    max collector current (BJTs, magnitude)
    vds_v   max drain-source voltage (MOSFETs, magnitude)
    id_a    max continuous drain current (MOSFETs, magnitude)
    vgs_v   max gate-source voltage (MOSFETs, magnitude)

pins
    Diodes and LEDs: "AK" (band or flat side = cathode; LED long leg = anode).
    TO-92 / metal cans: leads left to right, flat face (or tab) toward you,
    leads down.  TO-220: leads left to right, printed face toward you, metal tab at
    the back, leads down.  DIP-8 op-amps: the standard pin map, written out in full.
"""

PARTS = {
    # ------------------------------------------------------------------ diodes
    "1N4148": {
        "type": "D",
        "params": {"IS": 2.52e-9, "N": 1.752, "RS": 0.568, "CJO": 4e-12, "VJ": 1.0, "M": 0.4},
        "rating": {"pd_w": 0.5, "vmax_v": 100.0, "imax_a": 0.2},
        "desc": "Small-signal silicon switching diode (~0.6 V), the standard pedal clipper",
        "package": "DO-35",
        "pins": "AK",
        "source": "Widely circulated Fairchild/Philips 1N4148 SPICE model (IS=2.52n RS=0.568 N=1.752 "
                  "CJO=4p M=0.4); TT and BV dropped. Ratings: onsemi 1N4148 datasheet.",
    },
    "1N914": {
        "type": "D",
        "params": {"IS": 2.52e-9, "N": 1.752, "RS": 0.568, "CJO": 4e-12, "VJ": 1.0, "M": 0.4},
        "rating": {"pd_w": 0.5, "vmax_v": 100.0, "imax_a": 0.2},
        "desc": "Small-signal silicon switching diode, same die as the 1N4148",
        "package": "DO-35",
        "pins": "AK",
        "source": "Fairchild 1N914 SPICE model (IS=2.52n RS=0.568 N=1.752 CJO=4p M=0.4 TT=20n BV=100); "
                  "TT and BV dropped. Ratings: Fairchild 1N914 datasheet.",
    },
    "1N34A": {
        "type": "D",
        "params": {"IS": 2.0e-7, "N": 1.3, "RS": 5.0, "CJO": 1e-12, "VJ": 0.5, "M": 0.5},
        "rating": {"pd_w": 0.1, "vmax_v": 60.0, "imax_a": 0.05},
        "desc": "Germanium point-contact diode (~0.3 V), soft low-threshold clipper",
        "package": "DO-7 (glass)",
        "pins": "AK",
        "source": "Fitted to 1N34A datasheet curves: Vf ~0.29 V at 1 mA and ~0.37 V at 5 mA "
                  "(datasheet max 1 V at 5 mA), large saturation current for germanium leakage "
                  "(~0.2 uA). Ratings: PIV 60 V, IF 50 mA.",
    },
    "1N5817": {
        "type": "D",
        "params": {"IS": 31.7e-6, "N": 1.373, "RS": 0.051, "CJO": 190e-12, "VJ": 1.0, "M": 0.3},
        "rating": {"pd_w": 1.25, "vmax_v": 20.0, "imax_a": 1.0},
        "desc": "1 A 20 V Schottky rectifier, low-drop reverse-polarity protection",
        "package": "DO-41",
        "pins": "AK",
        "source": "onsemi 1N5817 model as shipped in the LTspice standard diode library "
                  "(Is=31.7u Rs=.051 N=1.373 Cjo=190p M=.3). Ratings: onsemi 1N5817 datasheet.",
    },
    "1N4001": {
        "type": "D",
        "params": {"IS": 14.11e-9, "N": 1.984, "RS": 33.89e-3, "CJO": 25.89e-12, "VJ": 0.3245, "M": 0.44},
        "rating": {"pd_w": 3.0, "vmax_v": 50.0, "imax_a": 1.0},
        "desc": "1 A 50 V silicon rectifier, series reverse-polarity protection",
        "package": "DO-41",
        "pins": "AK",
        "source": "Motorola/onsemi 1N4001 SPICE model (Is=14.11n N=1.984 Rs=33.89m Cjo=25.89p "
                  "M=.44 Vj=.3245); IKF, TT and BV dropped. Ratings: onsemi 1N4001 datasheet.",
    },
    "LED_RED": {
        "type": "D",
        "params": {"IS": 93.2e-12, "N": 3.73, "RS": 0.042, "CJO": 2.97e-12, "VJ": 0.75, "M": 0.333},
        "rating": {"pd_w": 0.1, "vmax_v": 5.0, "imax_a": 0.03},
        "desc": "5 mm red indicator LED (~1.85 V at 20 mA), also a soft clipper",
        "package": "5 mm LED",
        "pins": "AK",
        "source": "Widely circulated red LED SPICE model (IS=93.2p RS=42m N=3.73 CJO=2.97p VJ=.75 "
                  "M=.333); TT and BV dropped. Ratings: typical 5 mm red LED datasheet "
                  "(30 mA continuous, 5 V reverse).",
    },
    "LED_GREEN": {
        "type": "D",
        "params": {"IS": 1.7e-12, "N": 3.5, "RS": 0.5, "CJO": 3e-12, "VJ": 0.75, "M": 0.333},
        "rating": {"pd_w": 0.1, "vmax_v": 5.0, "imax_a": 0.03},
        "desc": "5 mm green (GaP) indicator LED (~2.1 V at 20 mA)",
        "package": "5 mm LED",
        "pins": "AK",
        "source": "Fitted to a typical GaP green 5 mm LED datasheet: Vf 2.1 V at 20 mA, ~1.8 V at "
                  "1 mA. Ratings: 30 mA continuous, 5 V reverse.",
    },
    "BAT41": {
        "type": "D",
        "params": {"IS": 8.3e-10, "N": 1.05, "RS": 2.0, "CJO": 2e-12, "VJ": 0.5, "M": 0.5},
        "rating": {"pd_w": 0.2, "vmax_v": 100.0, "imax_a": 0.1},
        "desc": "Small-signal 100 V Schottky diode (~0.38 V at 1 mA), low-threshold clipper",
        "package": "DO-35",
        "pins": "AK",
        "source": "Fitted to the ST BAT41 datasheet: VF ~0.38 V at 1 mA (max 0.45 V), under 1 V at "
                  "200 mA, C ~2 pF. Ratings: ST BAT41 datasheet (VRRM 100 V, IF 100 mA).",
    },
    # ------------------------------------------------------------- NPN BJTs
    "2N3904": {
        "type": "NPN",
        "params": {"IS": 6.734e-15, "BF": 134.0, "BR": 0.7371, "NF": 1.0, "NR": 1.0, "VAF": 74.03,
                   "RB": 10.0, "RC": 1.0, "RE": 0.0,
                   "CJE": 4.493e-12, "VJE": 0.75, "MJE": 0.2593,
                   "CJC": 3.638e-12, "VJC": 0.75, "MJC": 0.3085},
        "rating": {"pd_w": 0.625, "vceo_v": 40.0, "ic_a": 0.2},
        "desc": "General-purpose small-signal NPN, 40 V 200 mA",
        "package": "TO-92",
        "pins": "EBC",
        "source": "Fairchild 2N3904 model (LTspice standard.bjt): IS=6.734f BF=416.4 BR=.7371 "
                  "VAF=74.03 RB=10 RC=1 CJE=4.493p MJE=.2593 CJC=3.638p MJC=.3085, ISE=6.734f "
                  "NE=1.259 IKF=66.78m. ISE/IKF folded into BF=134 (the full model's hFE at "
                  "Ic=1 mA); TF/TR dropped. Ratings: onsemi 2N3904 datasheet.",
    },
    "2N2222": {
        "type": "NPN",
        "params": {"IS": 14.34e-15, "BF": 148.0, "BR": 6.092, "NF": 1.0, "NR": 1.0, "VAF": 74.03,
                   "RB": 10.0, "RC": 1.0, "RE": 0.0,
                   "CJE": 22.01e-12, "VJE": 0.75, "MJE": 0.377,
                   "CJC": 7.306e-12, "VJC": 0.75, "MJC": 0.3416},
        "rating": {"pd_w": 0.5, "vceo_v": 40.0, "ic_a": 0.8},
        "desc": "General-purpose NPN, 40 V 800 mA (2N2222A ratings)",
        "package": "TO-18",
        "pins": "EBC",
        "source": "Fairchild 2N2222 model (LTspice standard.bjt): IS=14.34f BF=255.9 BR=6.092 "
                  "VAF=74.03 RB=10 RC=1 CJE=22.01p MJE=.377 CJC=7.306p MJC=.3416, ISE=14.34f "
                  "NE=1.307 IKF=.2847. ISE/IKF folded into BF=148 (hFE at Ic=1 mA); TF/TR "
                  "dropped. Ratings: 2N2222A datasheet (TO-18).",
    },
    "BC549C": {
        "type": "NPN",
        "params": {"IS": 2.39e-14, "BF": 520.0, "BR": 10.0, "NF": 1.008, "NR": 1.0, "VAF": 80.0,
                   "RB": 10.0, "RC": 1.0, "RE": 0.0,
                   "CJE": 11.5e-12, "VJE": 0.75, "MJE": 0.33,
                   "CJC": 5.25e-12, "VJC": 0.75, "MJC": 0.33},
        "rating": {"pd_w": 0.5, "vceo_v": 30.0, "ic_a": 0.1},
        "desc": "Low-noise high-gain NPN (hFE group C, 420-800)",
        "package": "TO-92",
        "pins": "CBE",
        "source": "Based on the Philips/NXP BC549C Gummel-Poon model (IS=23.9f NF=1.008), with BF "
                  "set to the datasheet typical hFE of 520 at 2 mA and junction capacitances from "
                  "the datasheet (Cobo ~4.5 pF). Ratings: NXP BC549 datasheet.",
    },
    "2N5088": {
        "type": "NPN",
        "params": {"IS": 5.911e-15, "BF": 628.0, "BR": 1.271, "NF": 1.0, "NR": 1.0, "VAF": 62.37,
                   "RB": 10.0, "RC": 1.61, "RE": 0.0,
                   "CJE": 4.973e-12, "VJE": 0.75, "MJE": 0.4146,
                   "CJC": 4.017e-12, "VJC": 0.75, "MJC": 0.3174},
        "rating": {"pd_w": 0.625, "vceo_v": 30.0, "ic_a": 0.05},
        "desc": "High-gain low-noise NPN (hFE 350-1050), classic fuzz and wah transistor",
        "package": "TO-92",
        "pins": "EBC",
        "source": "Fairchild 2N5088 model: IS=5.911f BF=1122 BR=1.271 VAF=62.37 RB=10 RC=1.61 "
                  "CJE=4.973p MJE=.4146 CJC=4.017p MJC=.3174, ISE=5.911f NE=1.394 IKF=14.92m. "
                  "ISE/IKF folded into BF=628 (hFE at Ic=1 mA); TF/TR dropped. Ratings: onsemi "
                  "2N5088 datasheet.",
    },
    "MPSA18": {
        "type": "NPN",
        "params": {"IS": 33.58e-15, "BF": 900.0, "BR": 6.0, "NF": 1.0, "NR": 1.0, "VAF": 100.0,
                   "RB": 10.0, "RC": 1.0, "RE": 0.0,
                   "CJE": 8e-12, "VJE": 0.75, "MJE": 0.33,
                   "CJC": 3.5e-12, "VJC": 0.75, "MJC": 0.33},
        "rating": {"pd_w": 0.625, "vceo_v": 45.0, "ic_a": 0.2},
        "desc": "Very high gain low-noise NPN (hFE 500-1500)",
        "package": "TO-92",
        "pins": "EBC",
        "source": "Based on the Fairchild MPSA18 model (IS=33.58f, VAF=100) with BF set to 900, "
                  "mid-range of the datasheet hFE (500-1500 at 1 mA); capacitances from the "
                  "datasheet (Cobo 3 pF). Ratings: onsemi MPSA18 datasheet.",
    },
    # ------------------------------------------------------------- PNP BJTs
    "2N3906": {
        "type": "PNP",
        "params": {"IS": 1.41e-15, "BF": 180.7, "BR": 4.977, "NF": 1.0, "NR": 1.0, "VAF": 18.7,
                   "RB": 10.0, "RC": 2.5, "RE": 0.0,
                   "CJE": 8.063e-12, "VJE": 0.75, "MJE": 0.3677,
                   "CJC": 9.728e-12, "VJC": 0.75, "MJC": 0.5776},
        "rating": {"pd_w": 0.625, "vceo_v": 40.0, "ic_a": 0.2},
        "desc": "General-purpose small-signal PNP, 40 V 200 mA",
        "package": "TO-92",
        "pins": "EBC",
        "source": "Fairchild 2N3906 model (LTspice standard.bjt): IS=1.41f BF=180.7 BR=4.977 "
                  "VAF=18.7 RB=10 RC=2.5 CJE=8.063p MJE=.3677 CJC=9.728p MJC=.5776 (ISE=0, "
                  "IKF=80m has little effect at 1 mA); TF/TR dropped. Ratings: onsemi 2N3906.",
    },
    "BC559C": {
        "type": "PNP",
        "params": {"IS": 1.02e-14, "BF": 500.0, "BR": 10.0, "NF": 1.0, "NR": 1.0, "VAF": 45.0,
                   "RB": 10.0, "RC": 1.0, "RE": 0.0,
                   "CJE": 13e-12, "VJE": 0.75, "MJE": 0.33,
                   "CJC": 7e-12, "VJC": 0.75, "MJC": 0.33},
        "rating": {"pd_w": 0.5, "vceo_v": 30.0, "ic_a": 0.1},
        "desc": "Low-noise high-gain PNP (hFE group C, 420-800)",
        "package": "TO-92",
        "pins": "CBE",
        "source": "Based on the Philips/NXP BC559C model family, with BF set to the datasheet "
                  "typical hFE (~500 at 2 mA) and capacitances from the datasheet (Cobo ~6 pF). "
                  "Ratings: NXP BC559 datasheet.",
    },
    "AC128": {
        "type": "PNP",
        "params": {"IS": 3.0e-7, "BF": 90.0, "BR": 5.0, "NF": 1.0, "NR": 1.0, "VAF": 30.0,
                   "RB": 50.0, "RC": 1.0, "RE": 0.0,
                   "CJE": 100e-12, "VJE": 0.4, "MJE": 0.33,
                   "CJC": 40e-12, "VJC": 0.4, "MJC": 0.33},
        "rating": {"pd_w": 0.26, "vceo_v": 16.0, "ic_a": 1.0},
        "desc": "Germanium PNP (Vbe ~0.2 V, leaky), the vintage fuzz transistor",
        "package": "TO-1 (metal can)",
        "pins": "EBC",
        "source": "Fitted to the AC128 datasheet and measured vintage samples: Vbe ~0.21 V at 1 mA, "
                  "hFE 90 (datasheet 55-175), large IS so the junctions leak like germanium "
                  "(ICEO tens of uA). Ratings: AC128 datasheet (VCEO -16 V, IC -1 A).",
    },
    # ----------------------------------------------------------------- JFETs
    "J201": {
        "type": "NJF",
        "params": {"VTO": -0.7, "BETA": 1.0e-3, "LAMBDA": 0.005, "IS": 1e-14,
                   "CGS": 4e-12, "CGD": 3e-12, "PB": 1.0},
        "rating": {"pd_w": 0.35, "vmax_v": 40.0, "imax_a": 0.05},
        "desc": "Low-cutoff N-channel JFET (Vgs(off) ~-0.7 V, Idss ~0.5 mA), boosts and buffers",
        "package": "TO-92",
        "pins": "DSG",
        "source": "Fitted to the onsemi/Fairchild J201 datasheet typicals: VGS(off) -0.3 to -1.5 V "
                  "and IDSS 0.2-1 mA; VTO=-0.7 V and BETA=1 mA/V^2 give IDSS=0.49 mA. "
                  "Ratings: J201 datasheet.",
    },
    "2N5457": {
        "type": "NJF",
        "params": {"VTO": -1.372, "BETA": 1.125e-3, "LAMBDA": 2.3e-3, "IS": 181.3e-15,
                   "CGS": 4.627e-12, "CGD": 4e-12, "PB": 0.5},
        "rating": {"pd_w": 0.31, "vmax_v": 25.0, "imax_a": 0.01},
        "desc": "General-purpose N-channel JFET (Idss 1-5 mA), buffers and boosts",
        "package": "TO-92",
        "pins": "DSG",
        "source": "Fairchild 2N5457 model (LTspice standard.jft): Beta=1.125m Lambda=2.3m "
                  "Vto=-1.372 Is=181.3f Cgd=4p Cgs=4.627p Pb=.5 (IDSS 2.1 mA); RD/RS, Isr and "
                  "temperature terms dropped. Ratings: onsemi 2N5457 datasheet.",
    },
    "MPF102": {
        "type": "NJF",
        "params": {"VTO": -3.0, "BETA": 0.6e-3, "LAMBDA": 0.02, "IS": 1e-14,
                   "CGS": 4e-12, "CGD": 3e-12, "PB": 1.0},
        "rating": {"pd_w": 0.35, "vmax_v": 25.0, "imax_a": 0.01},
        "desc": "N-channel JFET with wide pinch-off (Idss 2-20 mA), clean boosts and RF",
        "package": "TO-92",
        "pins": "DSG",
        "source": "Fitted to onsemi MPF102 datasheet typicals: VGS(off) about -3 V, IDSS 5.4 mA "
                  "(datasheet 2-20 mA), Ciss 7 pF / Crss 3 pF. Ratings: MPF102 datasheet.",
    },
    # --------------------------------------------------------------- MOSFETs
    # SPICE level 1 (Shichman-Hodges), W = L = 100 um, so KP is the device beta and
    # Id = KP/2 (VGS - VTO)^2 (1 + LAMBDA VDS) in saturation.  Fitted to datasheet
    # typicals: VTO from VGS(th) at the datasheet's test current, KP from the forward
    # transconductance (gfs^2 / 2 Id), RD so that RDS(on) at VGS = 10 V matches,
    # CGS/CGD from Ciss - Crss and Crss, CBD from Coss - Crss at 25 V (graded junction,
    # PB 0.8, MJ 0.5), IS for the body diode drop.  Level 1 has no subthreshold
    # current or mobility degradation, so currents well below 1 mA or well above the
    # fitted point are approximate.
    "2N7000": {
        "type": "NMOS",
        "params": {"VTO": 2.0, "KP": 0.25, "LAMBDA": 0.01, "RD": 0.7, "IS": 1e-15,
                   "CGS": 16e-12, "CGD": 4e-12, "CBD": 40e-12, "PB": 0.8, "MJ": 0.5},
        "rating": {"pd_w": 0.4, "vds_v": 60.0, "id_a": 0.2, "vgs_v": 20.0},
        "desc": "Small-signal N-channel enhancement MOSFET, 60 V 200 mA, VGS(th) ~2.1 V",
        "package": "TO-92",
        "pins": "SGD",
        "source": "Datasheet-fitted level 1 (onsemi 2N7000): VGS(th) typ 2.1 V at 1 mA -> VTO 2.0; "
                  "gfs typ 320 mS at 200 mA -> KP 0.25; RDS(on) typ 1.2 ohm at 10 V -> RD 0.7; "
                  "Ciss 20 p / Crss 4 p / Coss 11 p at 25 V. Ratings: VDSS 60 V, ID 200 mA, "
                  "PD 400 mW, VGS +/-20 V.",
    },
    "BS170": {
        "type": "NMOS",
        "params": {"VTO": 2.0, "KP": 0.25, "LAMBDA": 0.01, "RD": 0.7, "IS": 1e-15,
                   "CGS": 17e-12, "CGD": 7e-12, "CBD": 55e-12, "PB": 0.8, "MJ": 0.5},
        "rating": {"pd_w": 0.83, "vds_v": 60.0, "id_a": 0.5, "vgs_v": 20.0},
        "desc": "Small-signal N-channel enhancement MOSFET, 60 V 500 mA (pinout reversed vs 2N7000)",
        "package": "TO-92",
        "pins": "DGS",
        "source": "Datasheet-fitted level 1 (onsemi/Fairchild BS170): VGS(th) typ 2.1 V at 1 mA -> "
                  "VTO 2.0; gfs typ 320 mS at 200 mA -> KP 0.25; RDS(on) typ 1.2 ohm -> RD 0.7; "
                  "Ciss 24 p / Crss 7 p / Coss 17 p at 25 V. Ratings: VDS 60 V, ID 500 mA, "
                  "PD 830 mW, VGS +/-20 V.",
    },
    "BS250": {
        "type": "PMOS",
        "params": {"VTO": -2.2, "KP": 0.1, "LAMBDA": 0.01, "RD": 4.0, "IS": 1e-15,
                   "CGS": 35e-12, "CGD": 10e-12, "CBD": 60e-12, "PB": 0.8, "MJ": 0.5},
        "rating": {"pd_w": 0.7, "vds_v": 45.0, "id_a": 0.23, "vgs_v": 20.0},
        "desc": "Small-signal P-channel enhancement MOSFET, -45 V -230 mA, complement of the BS170",
        "package": "TO-92",
        "pins": "DGS",
        "source": "Datasheet-fitted level 1 (Diodes/Zetex BS250P): VGS(th) -1 to -3.5 V -> VTO -2.2; "
                  "gfs ~200 mS at -200 mA -> KP 0.1; RDS(on) ~6 ohm at -10 V -> RD 4; Ciss ~45 p, "
                  "Crss ~10 p. Ratings: VDS -45 V, ID -230 mA, PD 700 mW, VGS +/-20 V. Pin order "
                  "differs between makers: check yours.",
    },
    "IRF510": {
        "type": "NMOS",
        "params": {"VTO": 3.0, "KP": 0.6, "LAMBDA": 0.005, "RD": 0.15, "IS": 1e-14,
                   "CGS": 165e-12, "CGD": 15e-12, "CBD": 370e-12, "PB": 0.8, "MJ": 0.5},
        "rating": {"pd_w": 2.0, "vds_v": 100.0, "id_a": 5.6, "vgs_v": 20.0},
        "desc": "N-channel power MOSFET, 100 V 5.6 A (2 W free air; 43 W on a heatsink)",
        "package": "TO-220",
        "pins": "GDS",
        "source": "Simplified level-1 fit to the Vishay IRF510 datasheet, not the vendor's "
                  "subcircuit: VGS(th) 2-4 V -> VTO 3.0; transfer curve (~1.5 A at 5 V, ~4 A at 6 V) "
                  "and gfs >= 1.3 S at 3.4 A -> KP 0.6; RDS(on) <= 0.54 ohm -> RD 0.15; Ciss 180 p, "
                  "Coss 81 p, Crss 15 p. Ratings: VDS 100 V, ID 5.6 A at TC 25 C; pd_w is the "
                  "free-air limit (RthJA 62 C/W -> 2 W); add '*@ rating M1 20W' with a heatsink.",
    },
    "IRF9510": {
        "type": "PMOS",
        "params": {"VTO": -3.0, "KP": 0.35, "LAMBDA": 0.005, "RD": 0.5, "IS": 1e-14,
                   "CGS": 180e-12, "CGD": 18e-12, "CBD": 430e-12, "PB": 0.8, "MJ": 0.5},
        "rating": {"pd_w": 2.0, "vds_v": 100.0, "id_a": 4.0, "vgs_v": 20.0},
        "desc": "P-channel power MOSFET, -100 V -4 A (2 W free air; 43 W on a heatsink)",
        "package": "TO-220",
        "pins": "GDS",
        "source": "Simplified level-1 fit to the Vishay IRF9510 datasheet, not the vendor's "
                  "subcircuit: VGS(th) -2 to -4 V -> VTO -3.0; gfs >= 1.1 S at -2.4 A -> KP 0.35; "
                  "RDS(on) <= 1.2 ohm -> RD 0.5; Ciss 200 p, Coss 94 p, Crss 18 p. Ratings: VDS "
                  "-100 V, ID -4 A at TC 25 C; pd_w is the free-air limit (RthJA 62 C/W -> 2 W).",
    },
    # --------------------------------------------------------------- op-amps
    "TL072": {
        "type": "OPAMP",
        "params": {"A0": 2.0e5, "GBW": 3.0e6, "RIN": 1e12, "ROUT": 75.0,
                   "VSAT_HI": 1.5, "VSAT_LO": 1.5, "IQ": 1.4e-3},
        "rating": {"pd_w": 0.68, "vmax_v": 36.0, "imax_a": 0.04},
        "desc": "Dual JFET-input op-amp, low noise, the pedal workhorse",
        "package": "DIP-8",
        "pins": "1 OUT A, 2 IN- A, 3 IN+ A, 4 V-, 5 IN+ B, 6 IN- B, 7 OUT B, 8 V+",
        "source": "TI TL072 datasheet (SLOS080) typicals: AVD 200 V/mV, GBW 3 MHz, ri 1e12 ohm, "
                  "swing 1.5 V from each rail into 10k, ICC 1.4 mA per amplifier; ROUT 75 ohm from "
                  "the TI Boyle macromodel. Ratings: supply 36 V total, DIP 680 mW.",
    },
    "TL071": {
        "type": "OPAMP",
        "params": {"A0": 2.0e5, "GBW": 3.0e6, "RIN": 1e12, "ROUT": 75.0,
                   "VSAT_HI": 1.5, "VSAT_LO": 1.5, "IQ": 1.4e-3},
        "rating": {"pd_w": 0.68, "vmax_v": 36.0, "imax_a": 0.04},
        "desc": "Single JFET-input op-amp, low noise",
        "package": "DIP-8",
        "pins": "1 OFFSET N1, 2 IN-, 3 IN+, 4 V-, 5 OFFSET N2, 6 OUT, 7 V+, 8 NC",
        "source": "TI TL071 datasheet (SLOS080) typicals, same die as the TL072: AVD 200 V/mV, "
                  "GBW 3 MHz, ri 1e12 ohm, swing 1.5 V from each rail, ICC 1.4 mA; ROUT 75 ohm "
                  "from the TI Boyle macromodel.",
    },
    "NE5532": {
        "type": "OPAMP",
        "params": {"A0": 1.0e5, "GBW": 10.0e6, "RIN": 300e3, "ROUT": 30.0,
                   "VSAT_HI": 2.0, "VSAT_LO": 2.0, "IQ": 4.0e-3},
        "rating": {"pd_w": 1.0, "vmax_v": 44.0, "imax_a": 0.038},
        "desc": "Dual bipolar low-noise audio op-amp, drives 600 ohm loads",
        "package": "DIP-8",
        "pins": "1 OUT A, 2 IN- A, 3 IN+ A, 4 V-, 5 IN+ B, 6 IN- B, 7 OUT B, 8 V+",
        "source": "TI NE5532 datasheet (SLOS075) typicals: AVD 100 V/mV, GBW 10 MHz, ri 300 kohm, "
                  "swing about 2 V from each rail, ICC 8 mA per package (4 mA per amplifier). "
                  "Ratings: supply +/-22 V, short-circuit current 38 mA.",
    },
    "LM358": {
        "type": "OPAMP",
        "params": {"A0": 1.0e5, "GBW": 0.7e6, "RIN": 10e6, "ROUT": 50.0,
                   "VSAT_HI": 1.5, "VSAT_LO": 0.02, "IQ": 0.35e-3},
        "rating": {"pd_w": 0.83, "vmax_v": 32.0, "imax_a": 0.04},
        "desc": "Dual low-power single-supply op-amp, output swings to ground",
        "package": "DIP-8",
        "pins": "1 OUT A, 2 IN- A, 3 IN+ A, 4 GND/V-, 5 IN+ B, 6 IN- B, 7 OUT B, 8 V+",
        "source": "TI LM358 datasheet (SNOSBT3) typicals: AVD 100 V/mV, unity-gain bandwidth "
                  "0.7 MHz, high swing VCC-1.5 V, low swing within 20 mV of ground, ICC 0.7 mA per "
                  "package (0.35 mA per amplifier); RIN and ROUT are typical macromodel values.",
    },
    "RC4558": {
        "type": "OPAMP",
        "params": {"A0": 3.0e5, "GBW": 3.0e6, "RIN": 5e6, "ROUT": 75.0,
                   "VSAT_HI": 1.0, "VSAT_LO": 1.0, "IQ": 1.25e-3},
        "rating": {"pd_w": 0.68, "vmax_v": 36.0, "imax_a": 0.025},
        "desc": "Dual general-purpose bipolar op-amp, the classic overdrive chip",
        "package": "DIP-8",
        "pins": "1 OUT A, 2 IN- A, 3 IN+ A, 4 V-, 5 IN+ B, 6 IN- B, 7 OUT B, 8 V+",
        "source": "TI RC4558 datasheet (SLOS073) typicals: AVD 300 V/mV, unity-gain bandwidth "
                  "3 MHz, ri 5 Mohm, swing +/-14 V on +/-15 V into 10k, ICC 2.5 mA per package "
                  "(1.25 mA per amplifier). Ratings: supply +/-18 V.",
    },
    "OPAMP_IDEAL": {
        "type": "OPAMP",
        "params": {"A0": 1.0e6, "GBW": 1.0e9, "RIN": 1e12, "ROUT": 1e-3,
                   "VSAT_HI": 0.0, "VSAT_LO": 0.0, "IQ": 0.0},
        "rating": {"pd_w": 10.0, "vmax_v": 100.0, "imax_a": 1.0},
        "desc": "Near-ideal op-amp for textbook circuits (rail-to-rail, 1 GHz GBW, no supply current)",
        "package": "virtual",
        "pins": "in+, in-, vcc, vee, out",
        "source": "Hotwire idealisation, not a real part: A0 1e6, GBW 1 GHz, rail-to-rail output, "
                  "1 mohm output resistance, zero quiescent current.",
    },
}

# Convenience groupings (used by the BOM and the library docs).
DIODES = [k for k, v in PARTS.items() if v["type"] == "D"]
BJTS = [k for k, v in PARTS.items() if v["type"] in ("NPN", "PNP")]
JFETS = [k for k, v in PARTS.items() if v["type"] in ("NJF", "PJF")]
MOSFETS = [k for k, v in PARTS.items() if v["type"] in ("NMOS", "PMOS")]
OPAMPS = [k for k, v in PARTS.items() if v["type"] == "OPAMP"]


# Parts language models reach for first. Same model families as above, with the datasheet hFE or
# op-amp typicals of the named part, so a netlist written from memory runs as written.
_ALIASES = {
    "BC547B": ("BC549C", {"BF": 290.0}, "General-purpose NPN, hFE group B (200-450)",
               "BC549C model family with BF at the BC547B datasheet typical (290 at 2 mA). Ratings: BC547 datasheet (VCEO 45 V)."),
    "BC547C": ("BC549C", {"BF": 520.0}, "General-purpose NPN, hFE group C (420-800)", "BC549C model; BC547 ratings (VCEO 45 V)."),
    "BC548B": ("BC549C", {"BF": 290.0}, "General-purpose NPN, hFE group B (200-450)", "BC549C model family, BF 290; BC548 ratings (VCEO 30 V)."),
    "BC550C": ("BC549C", {"BF": 520.0}, "Low-noise NPN, hFE group C (420-800)", "BC549C model; BC550 ratings (VCEO 45 V)."),
    "2N5089": ("2N5088", {"BF": 800.0}, "Very high-gain low-noise NPN (hFE 450-1800)", "2N5088 model family with BF 800 at 1 mA (2N5089 datasheet). Ratings: 2N5089 (VCEO 25 V)."),
    "TL074": ("TL072", {}, "Quad JFET-input op-amp (per amplifier same as TL072)", "TL072 macromodel; TL074 is the quad version (DIP-14)."),
    "TL071": ("TL072", {}, "Single JFET-input op-amp", "TL072 macromodel (single)."),
    "LM741": ("TL072", {"A0": 2e5, "GBW": 1e6, "RIN": 2e6, "ROUT": 75.0, "VSAT_HI": 2.0, "VSAT_LO": 2.0, "IQ": 1.7e-3},
              "Classic general-purpose op-amp (bipolar input, 1 MHz)", "LM741 datasheet typicals: 200 V/mV, 1 MHz, 2 Mohm input, 1.7 mA supply current; needs about 2 V from each rail."),
}
for _name, (_base, _over, _desc, _src) in _ALIASES.items():
    if _name in PARTS:
        continue
    _e = {k: (dict(v) if isinstance(v, dict) else v) for k, v in PARTS[_base].items()}
    _e["params"].update(_over)
    _e["desc"] = _desc
    _e["source"] = _src
    if _name == "TL074":
        _e["package"] = "DIP-14"
    if _name in ("BC547B", "BC547C", "BC550C"):
        _e["rating"] = dict(_e["rating"], vceo_v=45.0)
    if _name == "2N5089":
        _e["rating"] = dict(_e["rating"], vceo_v=25.0)
    PARTS[_name] = _e
