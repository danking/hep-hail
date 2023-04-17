import math
import hail as hl

ht: hl.Table
if not hl.hadoop_exists('array-of-structs.ht'):
    ht = hl.read_table('Run2012B_SingleMu-1000.ht')
    muon_fields =[
        'Muon_pt',
        'Muon_eta',
        'Muon_phi',
        'Muon_mass',
        'Muon_charge',
        'Muon_pfRelIso03_all',
        'Muon_pfRelIso04_all',
        'Muon_tightId',
        'Muon_softId',
        'Muon_dxy',
        'Muon_dxyErr',
        'Muon_dz',
        'Muon_dzErr',
        'Muon_jetIdx',
        'Muon_genPartIdx',
    ]

    ht = ht.transmute(
        muons = hl.zip(
            *[ht[f] for f in muon_fields],
            fill_missing=True
        ).map(
            lambda x: hl.struct(**{name.replace('Muon_', ''): x[i] for i, name in enumerate(muon_fields)})
        )
    )

    electron_fields = [
        'Electron_pt',
        'Electron_eta',
        'Electron_phi',
        'Electron_mass',
        'Electron_charge',
        'Electron_pfRelIso03_all',
        'Electron_dxy',
        'Electron_dxyErr',
        'Electron_dz',
        'Electron_dzErr',
        'Electron_cutBasedId',
        'Electron_pfId',
        'Electron_jetIdx',
        'Electron_genPartIdx',
    ]

    ht = ht.transmute(
        electrons = hl.zip(
            *[ht[f] for f in electron_fields],
            fill_missing=True
        ).map(
            lambda x: hl.struct(**{name.replace('Electron_', ''): x[i] for i, name in enumerate(electron_fields)})
        )
    )

    tau_fields = [
        'Tau_pt',
        'Tau_eta',
        'Tau_phi',
        'Tau_mass',
        'Tau_charge',
        'Tau_decayMode',
        'Tau_relIso_all',
        'Tau_jetIdx',
        'Tau_genPartIdx',
        'Tau_idDecayMode',
        'Tau_idIsoRaw',
        'Tau_idIsoVLoose',
        'Tau_idIsoLoose',
        'Tau_idIsoMedium',
        'Tau_idIsoTight',
        'Tau_idAntiEleLoose',
        'Tau_idAntiEleMedium',
        'Tau_idAntiEleTight',
        'Tau_idAntiMuLoose',
        'Tau_idAntiMuMedium',
        'Tau_idAntiMuTight',
    ]

    ht = ht.transmute(
        taus = hl.zip(
            *[ht[f] for f in tau_fields],
            fill_missing=True
        ).map(
            lambda x: hl.struct(**{name.replace('Tau_', ''): x[i] for i, name in enumerate(tau_fields)})
        )
    )

    photon_fields = [
        'Photon_pt',
        'Photon_eta',
        'Photon_phi',
        'Photon_mass',
        'Photon_charge',
        'Photon_pfRelIso03_all',
        'Photon_jetIdx',
        'Photon_genPartIdx',
    ]

    ht = ht.transmute(
        photons = hl.zip(
            *[ht[f] for f in photon_fields],
            fill_missing=True
        ).map(
            lambda x: hl.struct(**{name.replace('Photon_', ''): x[i] for i, name in enumerate(photon_fields)})
        )
    )

    jet_fields = [
        'Jet_pt',
        'Jet_eta',
        'Jet_phi',
        'Jet_mass',
        'Jet_puId',
        'Jet_btag',
    ]

    ht = ht.transmute(
        jets = hl.zip(
            *[ht[f] for f in jet_fields],
            fill_missing=True
        ).map(
            lambda x: hl.struct(**{name.replace('Jet_', ''): x[i] for i, name in enumerate(jet_fields)})
        )
    )
    ht = ht.add_index('index_in_original_file')
    ht = ht.key_by('run', 'luminosityBlock', 'event')
    ht.write('array-of-structs.ht')

ht: hl.Table = hl.read_table('array-of-structs.ht')
ht.describe()


# Listing 3: Simple unnesting of array elements
# df.Define("goodJet_pt", "Jet_pt[abs(Jet_eta) < 1]")
ht.annotate(
    goodJet_pt = ht.jets.filter(
        lambda j: j.eta < 1
    ).map(
        lambda j: j.pt
    )
)

# Listing 4: Querying unnested array elements
ht.filter(
    hl.len(
        ht.jets.filter(lambda j: j.pt > 40)
    ) > 1
)


# # The 8 queries

# 1. Plot the ETmiss of all events.

met_hist = ht.aggregate(
    hl.agg.hist(ht.MET_pt, 0, 2000, 100)
)
hl.plot.histogram(met_hist)

# 2. Plot the pT of all jets.

xx = ht
xx = xx.explode('jets')
xx = xx.transmute(**xx.jets)
jet_hist = xx.aggregate(
    hl.agg.hist(xx.pt, 15, 60, 100)
)
hl.plot.histogram(jet_hist)

# 3. Plot the pT of jets with |η| < 1.

xx = ht
xx = xx.explode('jets')
xx = xx.transmute(**xx.jets)
xx = xx.filter(xx.eta < 1)
jet_hist = xx.aggregate(
    hl.agg.hist(xx.pt, 15, 60, 100)
)
hl.plot.histogram(jet_hist)

# 4. Plot the ETmiss of events that have at least two jets with pT > 40 GeV.

xx = ht
xx = xx.filter(
    hl.len(xx.jets.filter(lambda j: j.pt > 40)) >= 2
)
met_pt_hist = xx.aggregate(
    hl.agg.hist(xx.MET_pt, 0, 2000, 100)
)
hl.plot.histogram(met_pt_hist)

# 5. Plot the ETmiss of events that have an opposite-charge muon pair with an invariant mass between
#    60 and 120 GeV.

def all_pairs(xs):
    n = hl.len(xs)
    return hl.range(n).flatmap(
        lambda i: hl.range(i + 1, n).map(
            lambda j: (xs[i], xs[j])
        )
    )

xx = ht
xx = xx.filter(
    all_pairs(xx.muons).any(
        lambda p: hl.all(
            p[0].charge != p[1].charge,
            p[0].mass + p[1].mass > 60,
            p[0].mass + p[1].mass < 120,
        )
    )
)
jet_hist = xx.aggregate(
    hl.agg.hist(xx.MET_pt, 0, 200, 100)
)
hl.plot.histogram(met_hist)

# 6. For events with at least three jets, plot the pT of the trijet four-momentum that has the
#    invariant mass closest to 172.5 GeV in each event and plot the maximum b-tagging discriminant
#    value among the jets in this trijet.

def all_triplets(xs):
    n = hl.len(xs)
    return hl.range(n).flatmap(
        lambda i: hl.range(i + 1, n).flatmap(
            lambda j: hl.range(j + 1, n).map(
                lambda k: (xs[i], xs[j], xs[k])
            )
        )
    )

xx = ht
xx = xx.filter(hl.len(xx.jets) >= 3)
triplets = all_triplets(xx.jets)
min_mass_triplet = triplets[
    hl.argmin(hl.starmap(lambda x, y, z: hl.abs(172.5 - (x.mass + y.mass + z.mass)), triplets))
]
xx = xx.annotate(
    min_mass_triplet = min_mass_triplet,
    max_btag_of_min_mass = hl.max(
        min_mass_triplet[0].btag, min_mass_triplet[1].btag, min_mass_triplet[2].btag
    ),
    trijet_pt = min_mass_triplet[0].pt + min_mass_triplet[1].pt + min_mass_triplet[2].pt
)
pt_hist, btag_hist = xx.aggregate((
    hl.agg.hist(xx.trijet_pt, 0, 200, 100),
    hl.agg.hist(xx.max_btag_of_min_mass, 0, 1, 100)
))
hl.plot.histogram(pt_hist)
hl.plot.histogram(btag_hist)

# 7. Plot the scalar sum in each event of the pT of jets with pT > 30 GeV that are not within 0.4 in
#    ΔR of any light lepton with pT > 10 GeV.


def delta_r(x, y):
    return hl.sqrt((x.eta - y.eta) ** 2 + (x.phi - y.phi) ** 2)

def too_far_or_too_weak(jet, lepton):
    return hl.any(lepton.pt <= 10, delta_r(lepton, jet) > 0.4)

xx = ht
xx = xx.annotate(jets = xx.jets.filter(lambda x: x.pt > 30))
xx = xx.annotate(jets = xx.jets.filter(
    lambda jet: hl.all(
        xx.muons.all(lambda muon: too_far_or_too_weak(jet, muon)),
        xx.electrons.all(lambda electron: too_far_or_too_weak(jet, electron))
    )
))
hist = xx.aggregate(hl.agg.hist(
    hl.sum(xx.jets.pt),
    15,
    200,
    100
))
hl.plot.histogram(hist)

# 8. For events with at least three light leptons and a same-flavor opposite-charge light lepton
#    pair, find such a pair that has the invariant mass closest to 91.2 GeV in each event and plot
#    the transverse mass of the system consisting of the missing tranverse momentum and the
#    highest-pT light lepton not in this pair.

xx = ht
n_light_leptons = hl.len(xx.muons) + hl.len(xx.electrons)
xx = xx.filter(n_light_leptons >= 3)

muons = hl.starmap(lambda i, muon: muon.select('mass', 'pt', 'charge', id=i), hl.enumerate(xx.muons))
electrons = hl.starmap(lambda i, muon: muon.select('mass', 'pt', 'charge', id=hl.len(muons) + i), hl.enumerate(xx.electrons))
light_leptons = muons.extend(electrons)
candidate_pairs = all_pairs(light_leptons).filter(lambda x: x[0].charge != x[1].charge)
xx = xx.annotate(
    light_leptons = light_leptons,
    best_pair = candidate_pairs[
        hl.argmin(candidate_pairs.map(lambda p: hl.abs(91.2 - (p[0].mass + p[1].mass))))
    ]
)
xx = xx.annotate(
    selected_leptons = hl.set([xx.best_pair[0].id, xx.best_pair[1].id])
)
other_light_leptons = xx.light_leptons.filter(lambda x: ~xx.selected_leptons.contains(x.id))
best_other_light_lepton = other_light_leptons[hl.argmax(other_light_leptons.pt)]
xx = xx.annotate(
    tmass = hl.sqrt(
        2.0 * best_other_light_lepton.pt * xx.MET_pt * (
            1.0 - hl.cos(xx.MET_phi - best_other_light_lepton.phi))
    )
)
hist = xx.aggregate(
    hl.agg.hist(xx.tmass, 0, 200, 100)
)
hl.plot.histogram(hist)
