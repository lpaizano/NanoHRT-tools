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
        self.out.branch("passHtTrig", "O")
        self.out.branch("passTauTrig", "O")

        # Event Variables
        self.out.branch("ak8_jets_multiplicity", "I")
        self.out.branch("loose_leptons_multiplicity", "I")
        self.out.branch("ak4_jets_multiplicity", "I")
        self.out.branch("ak4_all_jets_multiplicity", "I")
        self.out.branch("loose_bjet_multiplicity", "I")
        self.out.branch("bjet_multiplicity", "I")
        self.out.branch("events", "I")

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
        
        self.out.branch("bjets_pt", "F","1","medium_bjet_multiplicity")
        self.out.branch("bjets_eta", "F","1","medium_bjet_multiplicity")
        self.out.branch("bjets_phi", "F","1","medium_bjet_multiplicity")
        self.out.branch("bjets_m", "F","1","medium_bjet_multiplicity")

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        self.selectLeptons(event)
        self.correctJetsAndMET(event)

        # select lepton-cleaned jets
        event.fatjets = [fj for fj in event._allFatJets if fj.pt > 200 and abs(fj.eta) < 2.4 and (fj.jetId & 2)]
        event.ak4jets_all = [j for j in event._allJets if j.pt > 25 and abs(j.eta) < 2.4 and (j.jetId & 4)]

        if len(event.looseLeptons) != 0:
            return False
        
        if len(event.fatjets) < 1:
            return False
        
        max_score1 = -1
        highest_score1_jet = -1
        max_score2 = -1
        highest_score2_jet = -1
        for fj in event.fatjets:
            try:
                score1 = fj.ParticleNet_raw_probHtt
            except ZeroDivisionError:
                score1 = 0
                
            if score1 >= max_score1:
                max_score1 = score1
                highest_score1_jet += 1

            if fj.particleNet_QCD >= max_score2:
                max_score2 = fj.particleNet_QCD
                highest_score2_jet += 1
                
        count = 0
        ztt_jet = -1
        for fj in event.fatjets:
            if((count != highest_score2_jet) and (count < 100)):
                ztt_jet = count
                count = 100
            count += 1
        
        probe_jets = []
        probe_jets.append(event.fatjets[0])
        probe_jets.append(event.fatjets[1])

        event.ak4jets = [j for j in event.ak4jets_all if deltaR(j.eta,j.phi,probe_jets[0].eta,probe_jets[0].phi) >= self._jetConeSize]
        event.bjets_loose = [j for j in event.ak4jets if j.btagDeepFlavB > self.DeepJet_WP_L]
        event.bjets_medium = [j for j in event.ak4jets if j.btagDeepFlavB > self.DeepJet_WP_M]
        
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
        for fj in event.bjets_medium:
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
        self.out.fillBranch("ak8_jets_multiplicity", len(event.fatjets))
        self.out.fillBranch("ak4_jets_multiplicity", len(event.ak4jets))
        self.out.fillBranch("ak4_all_jets_multiplicity", len(event.ak4jets_all))
        self.out.fillBranch("bjet_multiplicity", len(event.bjets_medium))
        self.out.fillBranch("loose_bjet_multiplicity", len(event.bjets_loose))
        self.out.fillBranch("loose_leptons_multiplicity", len(event.looseLeptons))
        self.out.fillBranch("events", 1)

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
        
        self.out.fillBranch("passHtTrig", passTrigger(event, ['HLT_AK8PFHT800_TrimMass50', 'HLT_AK8PFJet400_TrimMass30', 'HLT_AK8PFJet500', 'HLT_PFJet500', 'HLT_PFHT1050', 'HLT_PFHT500_PFMET100_PFMHT100_IDTight', 'HLT_PFHT700_PFMET85_PFMHT85_IDTight', 'HLT_PFHT800_PFMET75_PFMHT75_IDTight']))
        self.out.fillBranch("passTauTrig", passTrigger(event, ['DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg','HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1']))

        return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
def DiTauTree_2016(): return DiTauSampleProducer(year=2016)
def DiTauTree_2017(): return DiTauSampleProducer(year=2017)
def DiTauTree_2018(): return DiTauSampleProducer(year=2018)
