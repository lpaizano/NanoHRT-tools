import math
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection

from .HeavyFlavBaseProducer import HeavyFlavBaseProducer
from ..helpers.utils import deltaR, closest, polarP4, sumP4, get_subjets, corrected_svmass, configLogger, deltaPhi, closest_dphi
from ..helpers.triggerHelper import passTrigger

class DiTauSampleProducer(HeavyFlavBaseProducer):

    def __init__(self, **kwargs):
        super(DiTauSampleProducer, self).__init__(channel='ditau', **kwargs)

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        super(DiTauSampleProducer, self).beginFile(inputFile, outputFile, inputTree, wrappedOutputTree)

        # trigger variables
        self.out.branch("passMuTrig", "O")
        self.out.branch("passTrig_HLT_Mu50", "O")
        self.out.branch("passTrig_HLT_IsoMu24", "O")

        # Event Variables
        self.out.branch("muons_multiplicity", "I")
        self.out.branch("ak8_jets_multiplicity", "I")
        self.out.branch("loose_leptons_multiplicity", "I")
        self.out.branch("ak4_jets_multiplicity", "I")
        self.out.branch("ak4_all_jets_multiplicity", "I")
        self.out.branch("bjet_multiplicity", "I")

        self.out.branch("muons_pt", "F","1","muons_multiplicity")
        self.out.branch("muons_eta", "F","1","muons_multiplicity")
        self.out.branch("muons_phi", "F","1","muons_multiplicity")
        self.out.branch("muons_m", "F","1","muons_multiplicity")
        self.out.branch("muons_iso", "F","1","muons_multiplicity")

        self.out.branch("ak8_jets_pt", "F","1","ak8_jets_multiplicity")
        self.out.branch("ak8_jets_eta", "F","1","ak8_jets_multiplicity")
        self.out.branch("ak8_jets_phi", "F","1","ak8_jets_multiplicity")
        self.out.branch("ak8_jets_m", "F","1","ak8_jets_multiplicity")

        self.out.branch("ak4_all_jets_pt", "F","1","ak4_all_jets_multiplicity")
        self.out.branch("ak4_all_jets_eta", "F","1","ak4_all_jets_multiplicity")
        self.out.branch("ak4_all_jets_phi", "F","1","ak4_all_jets_multiplicity")
        self.out.branch("ak4_all_jets_m", "F","1","ak4_all_jets_multiplicity")

        self.out.branch("ak4_jets_pt", "F","1","ak4_jets_multiplicity")
        self.out.branch("ak4_jets_eta", "F","1","ak4_jets_multiplicity")
        self.out.branch("ak4_jets_phi", "F","1","ak4_jets_multiplicity")
        self.out.branch("ak4_jets_m", "F","1","ak4_jets_multiplicity")

        self.out.branch("bjets_pt", "F","1","bjet_multiplicity")
        self.out.branch("bjets_eta", "F","1","bjet_multiplicity")
        self.out.branch("bjets_phi", "F","1","bjet_multiplicity")
        self.out.branch("bjets_m", "F","1","bjet_multiplicity")

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        self.selectLeptons(event)
        self.correctJetsAndMET(event)

        # select lepton-cleaned jets
        event.fatjets = [fj for fj in event._allFatJets if fj.pt > 200 and abs(fj.eta) < 2.4 and (fj.jetId & 2)]
        event.ak4jets_all = [j for j in event._allJets if j.pt > 25 and abs(j.eta) < 2.4 and (j.jetId & 4)]

        # muon selection                                                                            
        event._allMuons = Collection(event, "Muon")
        event.muons = [mu for mu in event._allMuons if mu.pt > 30 and abs(mu.eta) < 2.4 and abs(mu.dxy) < 0.05 and abs(mu.dz) < 0.2 and mu.tightId and mu.miniPFRelIso_all < 0.10]

        if len(event.ak4jets_all) < 1:
            return False

        if len(event.fatjets) < 1:
            return False

        if len(event.muons) != 1:
            return False

        if len(event.looseLeptons) != 1:
            return False

        if closest(event.muons[0],event.fatjets)[1] > 0.8:
            return False

        probe_jets = []
        probe_jets.append(closest(event.muons[0],event.fatjets)[0])
        probe_jets.append(event.fatjets[0])

        event.ak4jets = [j for j in event.ak4jets_all if deltaR(j.eta,j.phi,probe_jets[0].eta,probe_jets[0].phi) >= self._jetConeSize and deltaR(j.eta,j.phi,event.muons[0].eta,event.muons[0].phi) >= self._jetConeSize]

        event.bjets = [j for j in event.ak4jets if j.btagDeepFlavB > self.DeepJet_WP_M]

        muons_pt = []
        muons_eta = []
        muons_phi = []
        muons_m = []
        muons_iso = []
        for mu in event.muons:
            muons_pt.append(mu.pt)
            muons_eta.append(mu.eta)
            muons_phi.append(mu.phi)
            muons_m.append(mu.mass)
            muons_iso.append(mu.miniPFRelIso_all)

        ak8_jet_pt = []
        ak8_jet_eta = []
        ak8_jet_phi = []
        ak8_jet_m = []
        for fj in event.fatjets:
            ak8_jet_pt.append(fj.pt)
            ak8_jet_eta.append(fj.eta)
            ak8_jet_phi.append(fj.phi)
            ak8_jet_m.append(fj.mass)

        ak4_all_jet_pt = []
        ak4_all_jet_eta = []
        ak4_all_jet_phi = []
        ak4_all_jet_m = []
        for fj in event.ak4jets_all:
            ak4_all_jet_pt.append(fj.pt)
            ak4_all_jet_eta.append(fj.eta)
            ak4_all_jet_phi.append(fj.phi)
            ak4_all_jet_m.append(fj.mass)

        ak4_jet_pt = []
        ak4_jet_eta = []
        ak4_jet_phi = []
        ak4_jet_m = []
        for fj in event.ak4jets:
            ak4_jet_pt.append(fj.pt)
            ak4_jet_eta.append(fj.eta)
            ak4_jet_phi.append(fj.phi)
            ak4_jet_m.append(fj.mass)

        bjet_pt = []
        bjet_eta = []
        bjet_phi = []
        bjet_m = []
        for fj in event.bjets:
            bjet_pt.append(fj.pt)
            bjet_eta.append(fj.eta)
            bjet_phi.append(fj.phi)
            bjet_m.append(fj.mass)

        self.loadGenHistory(event, probe_jets)
        self.evalTagger(event, probe_jets)
        self.evalMassRegression(event, probe_jets)

        # fill output branches
        self.fillBaseEventInfo(event)
        self.fillFatJetInfo(event, probe_jets)

        #fill
        self.out.fillBranch("muons_multiplicity", len(event.muons))
        self.out.fillBranch("ak8_jets_multiplicity", len(event.fatjets))
        self.out.fillBranch("ak4_jets_multiplicity", len(event.ak4jets))
        self.out.fillBranch("ak4_all_jets_multiplicity", len(event.ak4jets_all))
        self.out.fillBranch("bjet_multiplicity", len(event.bjets))
        self.out.fillBranch("loose_leptons_multiplicity", len(event.looseLeptons))

        self.out.fillBranch("muons_pt",muons_pt)
        self.out.fillBranch("muons_eta",muons_eta)
        self.out.fillBranch("muons_phi",muons_phi)
        self.out.fillBranch("muons_m",muons_m)
        self.out.fillBranch("muons_iso",muons_iso)

        self.out.fillBranch("ak8_jets_pt",ak8_jet_pt)
        self.out.fillBranch("ak8_jets_eta",ak8_jet_eta)
        self.out.fillBranch("ak8_jets_phi",ak8_jet_phi)
        self.out.fillBranch("ak8_jets_m",ak8_jet_m)

        self.out.fillBranch("ak4_all_jets_pt",ak4_all_jet_pt)
        self.out.fillBranch("ak4_all_jets_eta",ak4_all_jet_eta)
        self.out.fillBranch("ak4_all_jets_phi",ak4_all_jet_phi)
        self.out.fillBranch("ak4_all_jets_m",ak4_all_jet_m)

        self.out.fillBranch("ak4_jets_pt",ak4_jet_pt)
        self.out.fillBranch("ak4_jets_eta",ak4_jet_eta)
        self.out.fillBranch("ak4_jets_phi",ak4_jet_phi)
        self.out.fillBranch("ak4_jets_m",ak4_jet_m)

        self.out.fillBranch("bjets_pt",bjet_pt)
        self.out.fillBranch("bjets_eta",bjet_eta)
        self.out.fillBranch("bjets_phi",bjet_phi)
        self.out.fillBranch("bjets_m",bjet_m)

        self.out.fillBranch("passTrig_HLT_Mu50", event.HLT_Mu50)
        self.out.fillBranch("passTrig_HLT_IsoMu24", event.HLT_IsoMu24)
        self.out.fillBranch("passMuTrig", passTrigger(event, ['HLT_Mu50', 'HLT_IsoMu24']))

        return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
def DiTauTree_2016(): return DiTauSampleProducer(year=2016)
def DiTauTree_2017(): return DiTauSampleProducer(year=2017)
def DiTauTree_2018(): return DiTauSampleProducer(year=2018)
