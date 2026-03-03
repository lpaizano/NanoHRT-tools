import os
import itertools
import numpy as np
import ROOT
import json
import math

ROOT.PyConfig.IgnoreCommandLineOptions = True

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from ..helpers.utils import deltaR, closest, polarP4, sumP4, get_subjets, corrected_svmass, configLogger, furthest, transverseMass, deltaPhi, sameflavor, emg
from ..helpers.xgbHelper import XGBEnsemble
from ..helpers.nnHelper import convert_prob, ensemble
from ..helpers.jetmetCorrector import JetMETCorrector, rndSeed

import logging
logger = logging.getLogger('nano')
configLogger('nano', loglevel=logging.INFO)

lumi_dict = {2015: 19.52, 2016: 16.81, 2017: 41.48, 2018: 59.83,20220: 7.98, 20221: 26.67, 2024: 108.96}


class _NullObject:
    '''An null object which does not store anything, and does not raise exception.'''

    def __bool__(self):
        return False

    def __nonzero__(self):
        return False

    def __getattr__(self, name):
        pass

    def __setattr__(self, name, value):
        pass


class METObject(Object):

    def p4(self):
        return polarP4(self, eta=None, mass=None)


class HeavyFlavBaseProducer(Module, object):

    def __init__(self, channel, **kwargs):
        self._channel = channel  # 'qcd', 'ditau', 'photon', 'inclusive', 'muon'
        self.year = int(kwargs['year'])
        self.jetType = kwargs.get('jetType', 'ak8').lower()
        self._jmeSysts = {'jec': False, 'jes': None, 'jes_source': '', 'jes_uncertainty_file_prefix': '',
                          'jer': None, 'jmr': None, 'met_unclustered': None, 'smearMET': True, 'applyHEMUnc': False}
        self._opts = {'sfbdt_threshold': -99,
                      'run_tagger': False, 'tagger_versions': ['V02b', 'V02c', 'V02d'],
                      'run_mass_regression': False, 'mass_regression_versions': ['V01a', 'V01b', 'V01c'],
                      'WRITE_CACHE_FILE': False}
        for k in kwargs:
            if k in self._jmeSysts:
                self._jmeSysts[k] = kwargs[k]
            else:
                self._opts[k] = kwargs[k]
        self._needsJMECorr = any([self._jmeSysts['jec'], self._jmeSysts['jes'],
                                  self._jmeSysts['jer'], self._jmeSysts['jmr'],
                                  self._jmeSysts['met_unclustered'], self._jmeSysts['applyHEMUnc']])

        logger.info('Running %s channel for %s jets with JME systematics %s, other options %s',
                    self._channel, self.jetType, str(self._jmeSysts), str(self._opts))

        if self.jetType == 'ak8':
            self._jetConeSize = 0.8
            self._fj_name = 'FatJet'
            self._sj_name = 'SubJet'
            self._fj_gen_name = 'GenJetAK8'
            self._sj_gen_name = 'SubGenJetAK8'
            self._sfbdt_files = [
                os.path.expandvars(
                    '$CMSSW_BASE/src/PhysicsTools/NanoHRTTools/data/sfBDT/ak15/xgb_train_qcd.model.%d' % idx)
                for idx in range(10)]  # FIXME: update to AK8 training
            self._sfbdt_vars = ['fj_2_tau21', 'fj_2_sj1_rawmass', 'fj_2_sj2_rawmass',
                                'fj_2_ntracks_sv12', 'fj_2_sj1_sv1_pt', 'fj_2_sj2_sv1_pt']
        elif self.jetType == 'ak15':
            self._jetConeSize = 1.5
            self._fj_name = 'AK15Puppi'
            self._sj_name = 'AK15PuppiSubJet'
            self._fj_gen_name = 'GenJetAK15'
            self._sj_gen_name = 'GenSubJetAK15'
            self._sfbdt_files = [
                os.path.expandvars(
                    '$CMSSW_BASE/src/PhysicsTools/NanoHRTTools/data/sfBDT/ak15/xgb_train_qcd.model.%d' % idx)
                for idx in range(10)]
            self._sfbdt_vars = ['fj_2_tau21', 'fj_2_sj1_rawmass', 'fj_2_sj2_rawmass',
                                'fj_2_ntracks_sv12', 'fj_2_sj1_sv1_pt', 'fj_2_sj2_sv1_pt']
        else:
            raise RuntimeError('Jet type %s is not recognized!' % self.jetType)

        if self._needsJMECorr:
            if (self.year == 2015 or self.year == 2016 or self.year == 2017 or self.year == 2018): 
                self.jetmetCorr = JetMETCorrector(year=self.year, jetType="AK4PFchs", **self._jmeSysts)
            else:
                self.jetmetCorr = JetMETCorrector(year=self.year, jetType="AK4PFPuppi", **self._jmeSysts)
            self.fatjetCorr = JetMETCorrector(year=self.year, jetType="AK8PFPuppi", **self._jmeSysts)
            self.subjetCorr = JetMETCorrector(year=self.year, jetType="AK4PFPuppi", **self._jmeSysts)

        if self._opts['run_tagger'] or self._opts['run_mass_regression']:
            from ..helpers.makeInputs import ParticleNetTagInfoMaker
            from ..helpers.runPrediction import ParticleNetJetTagsProducer
            self.tagInfoMaker = ParticleNetTagInfoMaker(
                fatjet_branch=self._fj_name, pfcand_branch='PFCands', sv_branch='SV', jetR=self._jetConeSize)
            prefix = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoHRTTools/data')
            if self._opts['run_tagger']:
                self.pnTaggers = [ParticleNetJetTagsProducer(
                    '%s/ParticleNet-MD/%s/{version}/particle-net.onnx' % (prefix, self.jetType),
                    '%s/ParticleNet-MD/%s/{version}/preprocess.json' % (prefix, self.jetType),
                    version=ver, cache_suffix='tagger') for ver in self._opts['tagger_versions']]
            if self._opts['run_mass_regression']:
                self.pnMassRegressions = [ParticleNetJetTagsProducer(
                    '%s/MassRegression/%s/{version}/particle_net_regression.onnx' % (prefix, self.jetType),
                    '%s/MassRegression/%s/{version}/preprocess.json' % (prefix, self.jetType),
                    version=ver, cache_suffix='mass') for ver in self._opts['mass_regression_versions']]

        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
        if self.year != 2024:
            self.DeepJet_WP_L = {2015: 0.0508, 2016: 0.0480, 2017: 0.0532, 2018: 0.0490,20220: 0.0583,20221: 0.0614}[self.year]
            self.DeepJet_WP_M = {2015: 0.2598, 2016: 0.2489, 2017: 0.3040, 2018: 0.2783,20220: 0.3086,20221: 0.3196}[self.year]
            self.DeepJet_WP_T = {2015: 0.6502, 2016: 0.6377, 2017: 0.7476, 2018: 0.7100,20220: 0.7183,20221: 0.7300}[self.year]

        if self.year == 2024:
            self.UParT_WP_L = {2024: 0.0246}[self.year]
            self.UParT_WP_M = {2024: 0.1272}[self.year]
            self.UParT_WP_T = {2024: 0.4648}[self.year]

    def beginJob(self):
        if self._needsJMECorr:
            self.jetmetCorr.beginJob()
            self.fatjetCorr.beginJob()
            self.subjetCorr.beginJob()
        if self._opts['sfbdt_threshold'] > -99:
            self.xgb = XGBEnsemble(self._sfbdt_files, self._sfbdt_vars)

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        self.hasParticleNetProb = bool(inputTree.GetBranch(self._fj_name + '_ParticleNetMD_probXbb'))

        # remove all possible h5 cache files
        for f in os.listdir('.'):
            if f.endswith('.h5'):
                os.remove(f)

        if self._opts['run_tagger']:
            for p in self.pnTaggers:
                p.load_cache(inputFile)

        if self._opts['run_mass_regression']:
            for p in self.pnMassRegressions:
                p.load_cache(inputFile)

        if self._opts['run_tagger'] or self._opts['run_mass_regression']:
            self.tagInfoMaker.init_file(inputFile, fetch_step=1000)

        self.out = wrappedOutputTree

        # NOTE: branch names must start with a lower case letter
        # check keep_and_drop_output.txt
        self.out.branch("year", "I")
        self.out.branch("lumiwgt", "F")
        self.out.branch("jetR", "F")
        self.out.branch("passmetfilters", "O")
        self.out.branch("l1PreFiringWeight", "F")
        self.out.branch("l1PreFiringWeightUp", "F")
        self.out.branch("l1PreFiringWeightDown", "F")
        self.out.branch("nlep", "I")
        self.out.branch("ht", "F")
        self.out.branch("met", "F")
        self.out.branch("puppi_met", "F")
        self.out.branch("met_significance", "F")
        self.out.branch("puppi_met_significance", "F")
        self.out.branch("metphi", "F")
        self.out.branch("puppi_metphi", "F")

        self.out.branch("puweight_nom", "F")
        self.out.branch("puweight_up", "F")
        self.out.branch("puweight_down", "F")
        self.out.branch("pileup_nTrueInt", "I")

        # Large-R jets
        for idx in ([1, 2] if (self._channel == 'qcd' or self._channel == 'ditau') else [1]):
            prefix = 'fj_%d_' % idx

            # fatjet kinematics
            self.out.branch(prefix + "is_qualified", "O")
            self.out.branch(prefix + "id", "F")
            self.out.branch(prefix + "pt", "F")
            self.out.branch(prefix + "eta", "F")
            self.out.branch(prefix + "phi", "F")
            self.out.branch(prefix + "rawfactor", "F")
            self.out.branch(prefix + "mass", "F")
            self.out.branch(prefix + "rawmass", "F")
            self.out.branch(prefix + "sdmass", "F")
            self.out.branch(prefix + "sdmass_v15", "F")
            self.out.branch(prefix + "trmass", "F")
            self.out.branch(prefix + "regressed_mass", "F")
            self.out.branch(prefix + "tau_regressed_mass", "F")
            self.out.branch(prefix + "ParticleNet_regressed_mass", "F")
            self.out.branch(prefix + "globalParT3_massCorrGeneric_regressed_mass", "F")
            self.out.branch(prefix + "globalParT3_massCorrX2p_regressed_mass", "F")
            self.out.branch(prefix + "tau21", "F")
            self.out.branch(prefix + "tau32", "F")
            self.out.branch(prefix + "met_dphi", "F")

            # subjets
            self.out.branch(prefix + "deltaR_sj12", "F")
            self.out.branch(prefix + "sj1_pt", "F")
            self.out.branch(prefix + "sj1_eta", "F")
            self.out.branch(prefix + "sj1_phi", "F")
            self.out.branch(prefix + "sj1_rawmass", "F")
            self.out.branch(prefix + "sj1_btagdeepcsv", "F")
            self.out.branch(prefix + "sj2_pt", "F")
            self.out.branch(prefix + "sj2_eta", "F")
            self.out.branch(prefix + "sj2_phi", "F")
            self.out.branch(prefix + "sj2_rawmass", "F")
            self.out.branch(prefix + "sj2_btagdeepcsv", "F")

            #PNet Taggers
            self.out.branch(prefix + "ParticleNet_newlabel_raw_probHbb", "F")
            self.out.branch(prefix + "ParticleNet_newlabel_raw_probHcc", "F")
            self.out.branch(prefix + "ParticleNet_newlabel_raw_probHee", "F")
            self.out.branch(prefix + "ParticleNet_newlabel_raw_probHem", "F")
            self.out.branch(prefix + "ParticleNet_newlabel_raw_probHgg", "F")
            self.out.branch(prefix + "ParticleNet_newlabel_raw_probHmm", "F")
            self.out.branch(prefix + "ParticleNet_newlabel_raw_probHqq", "F")
            self.out.branch(prefix + "ParticleNet_newlabel_raw_probHte", "F")
            self.out.branch(prefix + "ParticleNet_newlabel_raw_probHtm", "F")
            self.out.branch(prefix + "ParticleNet_newlabel_raw_probHtt", "F")
            self.out.branch(prefix + "ParticleNet_newlabel_raw_probQCD0hf", "F")
            self.out.branch(prefix + "ParticleNet_newlabel_raw_probQCD1hf", "F")
            self.out.branch(prefix + "ParticleNet_newlabel_raw_probQCD2hf", "F")
            self.out.branch(prefix + "ParticleNet_newlabel_raw_probSingleTau", "F")

            self.out.branch(prefix + "ParticleNet_newlabelwjets_raw_probHbb", "F")
            self.out.branch(prefix + "ParticleNet_newlabelwjets_raw_probHcc", "F")
            self.out.branch(prefix + "ParticleNet_newlabelwjets_raw_probHee", "F")
            self.out.branch(prefix + "ParticleNet_newlabelwjets_raw_probHem", "F")
            self.out.branch(prefix + "ParticleNet_newlabelwjets_raw_probHgg", "F")
            self.out.branch(prefix + "ParticleNet_newlabelwjets_raw_probHmm", "F")
            self.out.branch(prefix + "ParticleNet_newlabelwjets_raw_probHqq", "F")
            self.out.branch(prefix + "ParticleNet_newlabelwjets_raw_probHte", "F")
            self.out.branch(prefix + "ParticleNet_newlabelwjets_raw_probHtm", "F")
            self.out.branch(prefix + "ParticleNet_newlabelwjets_raw_probHtt", "F")
            self.out.branch(prefix + "ParticleNet_newlabelwjets_raw_probQCD0hf", "F")
            self.out.branch(prefix + "ParticleNet_newlabelwjets_raw_probQCD1hf", "F")
            self.out.branch(prefix + "ParticleNet_newlabelwjets_raw_probQCD2hf", "F")
            self.out.branch(prefix + "ParticleNet_newlabelwjets_raw_probSingleTau", "F")

            self.out.branch(prefix + "ParticleNet_raw_masscorr", "F")
            self.out.branch(prefix + "ParticleNet_raw_probHbb", "F")
            self.out.branch(prefix + "ParticleNet_raw_probHcc", "F")
            self.out.branch(prefix + "ParticleNet_raw_probHgg", "F")
            self.out.branch(prefix + "ParticleNet_raw_probHqq", "F")
            self.out.branch(prefix + "ParticleNet_raw_probHte", "F")
            self.out.branch(prefix + "ParticleNet_raw_probHtm", "F")
            self.out.branch(prefix + "ParticleNet_raw_probHtt", "F")
            self.out.branch(prefix + "ParticleNet_raw_probQCD0hf", "F")
            self.out.branch(prefix + "ParticleNet_raw_probQCD1hf", "F")
            self.out.branch(prefix + "ParticleNet_raw_probQCD2hf", "F")

            self.out.branch(prefix + "particleNetLegacy_QCD", "F")
            self.out.branch(prefix + "particleNetLegacy_Xqq", "F")
            self.out.branch(prefix + "particleNetLegacy_Xbb", "F")
            self.out.branch(prefix + "particleNetLegacy_Xcc", "F")
            self.out.branch(prefix + "particleNetLegacy_mass", "F")

            self.out.branch(prefix + "particleNetWithMass_H4qvsQCD", "F")
            self.out.branch(prefix + "particleNetWithMass_HbbvsQCD", "F")
            self.out.branch(prefix + "particleNetWithMass_HccvsQCD", "F")
            self.out.branch(prefix + "particleNetWithMass_QCD", "F")
            self.out.branch(prefix + "particleNetWithMass_TvsQCD", "F")
            self.out.branch(prefix + "particleNetWithMass_WvsQCD", "F")
            self.out.branch(prefix + "particleNetWithMass_ZvsQCD", "F")

            self.out.branch(prefix + "particleNet_QCD", "F")
            self.out.branch(prefix + "particleNet_QCD0HF", "F")
            self.out.branch(prefix + "particleNet_QCD1HF", "F")
            self.out.branch(prefix + "particleNet_QCD2HF", "F")
            self.out.branch(prefix + "particleNet_WVsQCD", "F")
            self.out.branch(prefix + "particleNet_XbbVsQCD", "F")
            self.out.branch(prefix + "particleNet_XccVsQCD", "F")
            self.out.branch(prefix + "particleNet_XggVsQCD", "F")
            self.out.branch(prefix + "particleNet_XqqVsQCD", "F")
            self.out.branch(prefix + "particleNet_XteVsQCD", "F")
            self.out.branch(prefix + "particleNet_XtmVsQCD", "F")
            self.out.branch(prefix + "particleNet_XttVsQCD", "F")
            self.out.branch(prefix + "particleNet_masscorr", "F")

            #GpartT Taggers
            self.out.branch(prefix + "globalParT3_QCD", "F")
            self.out.branch(prefix + "globalParT3_TopbWev", "F")
            self.out.branch(prefix + "globalParT3_TopbWmv", "F")
            self.out.branch(prefix + "globalParT3_TopbWq", "F")
            self.out.branch(prefix + "globalParT3_TopbWqq", "F")
            self.out.branch(prefix + "globalParT3_TopbWtauhv", "F")
            self.out.branch(prefix + "globalParT3_WvsQCD", "F")
            self.out.branch(prefix + "globalParT3_XWW3q", "F")
            self.out.branch(prefix + "globalParT3_XWW4q", "F")
            self.out.branch(prefix + "globalParT3_XWWqqev", "F")
            self.out.branch(prefix + "globalParT3_XWWqqmv", "F")
            self.out.branch(prefix + "globalParT3_Xbb", "F")
            self.out.branch(prefix + "globalParT3_Xcc", "F")
            self.out.branch(prefix + "globalParT3_Xcs", "F")
            self.out.branch(prefix + "globalParT3_Xqq", "F")
            self.out.branch(prefix + "globalParT3_Xtauhtaue", "F")
            self.out.branch(prefix + "globalParT3_Xtauhtauh", "F")
            self.out.branch(prefix + "globalParT3_Xtauhtaum", "F")
            self.out.branch(prefix + "globalParT3_massCorrGeneric", "F")
            self.out.branch(prefix + "globalParT3_massCorrX2p", "F")
            self.out.branch(prefix + "globalParT3_withMassTopvsQCD", "F")
            self.out.branch(prefix + "globalParT3_withMassWvsQCD", "F")
            self.out.branch(prefix + "globalParT3_withMassZvsQCD", "F")

            #Nine Transformations
            self.out.branch(prefix + "nine_ParticleNet_newlabel_raw_probHte", "F")
            self.out.branch(prefix + "nine_ParticleNet_newlabel_raw_probHtm", "F")
            self.out.branch(prefix + "nine_ParticleNet_newlabel_raw_probHtt", "F")
            self.out.branch(prefix + "nine_ParticleNet_newlabel_raw_probSingleTau", "F")

            self.out.branch(prefix + "nine_ParticleNet_newlabelwjets_raw_probHte", "F")
            self.out.branch(prefix + "nine_ParticleNet_newlabelwjets_raw_probHtm", "F")
            self.out.branch(prefix + "nine_ParticleNet_newlabelwjets_raw_probHtt", "F")
            self.out.branch(prefix + "nine_ParticleNet_newlabelwjets_raw_probSingleTau", "F")

            self.out.branch(prefix + "nine_ParticleNet_raw_probHte", "F")
            self.out.branch(prefix + "nine_ParticleNet_raw_probHtm", "F")
            self.out.branch(prefix + "nine_ParticleNet_raw_probHtt", "F")

            self.out.branch(prefix + "nine_globalParT3_Xtauhtaue", "F")
            self.out.branch(prefix + "nine_globalParT3_Xtauhtauh", "F")
            self.out.branch(prefix + "nine_globalParT3_Xtauhtaum", "F")

            self.out.branch(prefix + "nine_particleNet_XteVsQCD", "F")
            self.out.branch(prefix + "nine_particleNet_XtmVsQCD", "F")
            self.out.branch(prefix + "nine_particleNet_XttVsQCD", "F")

            #More Discriminators
            self.out.branch(prefix + "ParticleNet_newlabel_Httvssf", "F")
            self.out.branch(prefix + "ParticleNet_newlabelwjetsHttvssf", "F")
            self.out.branch(prefix + "nine_ParticleNet_newlabel_Httvssf", "F")
            self.out.branch(prefix + "nine_ParticleNet_newlabelwjetsHttvssf", "F")

            self.out.branch(prefix + "ParticleNet_newlabel_Httvsemg", "F")
            self.out.branch(prefix + "ParticleNet_newlabelwjetsHttvsemg", "F")
            self.out.branch(prefix + "nine_ParticleNet_newlabel_Httvsemg", "F")
            self.out.branch(prefix + "nine_ParticleNet_newlabelwjetsHttvsemg", "F")

            if self._opts['run_tagger']:
                self.out.branch(prefix + "origParticleNetMD_XccVsQCD", "F")
                self.out.branch(prefix + "origParticleNetMD_XbbVsQCD", "F")

            # matching variables
            if self.isMC:
                self.out.branch(prefix + "dr_genTaus", "F")
                self.out.branch(prefix + "dr_genTaus_Lep", "F")

                self.out.branch("gen" + "_" + "x_decay", "F")
                self.out.branch("gen" + "_" + "tau_lep_decay", "F")

                self.out.branch("gen" + "_" + "tau1_pt", "F")
                self.out.branch("gen" + "_" + "tau1_eta", "F")
                self.out.branch("gen" + "_" + "tau1_phi", "F")
                self.out.branch("gen" + "_" + "tau1_e", "F")

                self.out.branch("gen" + "_" + "tau2_pt", "F")
                self.out.branch("gen" + "_" + "tau2_eta", "F")
                self.out.branch("gen" + "_" + "tau2_phi", "F")
                self.out.branch("gen" + "_" + "tau2_e", "F")

                self.out.branch("gen" + "_" + "vis_tau1_pt", "F")
                self.out.branch("gen" + "_" + "vis_tau1_eta", "F")
                self.out.branch("gen" + "_" + "vis_tau1_phi", "F")
                self.out.branch("gen" + "_" + "vis_tau1_e", "F")

                self.out.branch("gen" + "_" + "vis_tau2_pt", "F")
                self.out.branch("gen" + "_" + "vis_tau2_eta", "F")
                self.out.branch("gen" + "_" + "vis_tau2_phi", "F")
                self.out.branch("gen" + "_" + "vis_tau2_e", "F")

                self.out.branch("gen" + "_" + "neutrino1_pt", "F")
                self.out.branch("gen" + "_" + "neutrino1_pz", "F")
                self.out.branch("gen" + "_" + "neutrino1_eta", "F")
                self.out.branch("gen" + "_" + "neutrino1_phi", "F")
                self.out.branch("gen" + "_" + "neutrino1_e", "F")

                self.out.branch("gen" + "_" + "neutrino2_pt", "F")
                self.out.branch("gen" + "_" + "neutrino2_pz", "F")
                self.out.branch("gen" + "_" + "neutrino2_eta", "F")
                self.out.branch("gen" + "_" + "neutrino2_phi", "F")
                self.out.branch("gen" + "_" + "neutrino2_e", "F")

                self.out.branch("gen" + "_" + "neutrinos_pt", "F")
                self.out.branch("gen" + "_" + "neutrinos_pz", "F")
                self.out.branch("gen" + "_" + "neutrinos_eta", "F")
                self.out.branch("gen" + "_" + "neutrinos_phi", "F")
                self.out.branch("gen" + "_" + "neutrinos_e", "F")

                self.out.branch("gen" + "_" + "ditau_pt", "F")
                self.out.branch("gen" + "_" + "ditau_eta", "F")
                self.out.branch("gen" + "_" + "ditau_phi", "F")
                self.out.branch("gen" + "_" + "ditau_e", "F")
                self.out.branch("gen" + "_" + "ditau_m", "F")

                self.out.branch("gen" + "_" + "ditau_vis_pt", "F")
                self.out.branch("gen" + "_" + "ditau_vis_eta", "F")
                self.out.branch("gen" + "_" + "ditau_vis_phi", "F")
                self.out.branch("gen" + "_" + "ditau_vis_e", "F")
                self.out.branch("gen" + "_" + "ditau_vis_m", "F")

                self.out.branch("gen" + "_" + "met_pt", "F")
                self.out.branch("gen" + "_" + "met_phi", "F")

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        if self._opts['run_tagger'] and self._opts['WRITE_CACHE_FILE']:
            for p in self.pnTaggers:
                p.update_cache()

        if self._opts['run_mass_regression'] and self._opts['WRITE_CACHE_FILE']:
            for p in self.pnMassRegressions:
                p.update_cache()

        # remove all h5 cache files
        if self._opts['run_tagger'] or self._opts['run_mass_regression']:
            for f in os.listdir('.'):
                if f.endswith('.h5'):
                    os.remove(f)

    def selectLeptons(self, event):
        # do lepton selection
        event.looseLeptons = []  # used for jet lepton cleaning & lepton counting
        event.looseElectrons = []
        event.looseMuons = []

        electrons = Collection(event, "Electron")
        for el in electrons:
            if (self.year == 2015 or self.year == 2016 or self.year == 2017):
                el.iso = el.mvaFall17V2noIso_WP90
            else:
                el.iso = el.mvaNoIso_WP90
            el.etaSC = el.eta + el.deltaEtaSC
            if el.pt > 10 and abs(el.eta) < 2.5 and abs(el.dxy) < 0.05 and abs(el.dz) < 0.2 \
                    and el.iso and el.miniPFRelIso_all < 0.4:
                event.looseLeptons.append(el)
                event.looseElectrons.append(el)
        muons = Collection(event, "Muon")
        for mu in muons:
            if mu.pt > 10 and abs(mu.eta) < 2.4 and abs(mu.dxy) < 0.05 and abs(mu.dz) < 0.2 \
                    and mu.looseId and mu.miniPFRelIso_all < 0.4:
                event.looseLeptons.append(mu)
                event.looseMuons.append(mu)

        event.looseLeptons.sort(key=lambda x: x.pt, reverse=True)
        event.looseElectrons.sort(key=lambda x: x.pt, reverse=True)
        event.looseMuons.sort(key=lambda x: x.pt, reverse=True)

    def correctJetsAndMET(self, event):
        # correct Jets and MET
        event.idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
        event._allJets = Collection(event, "Jet")
        if (self.year == 2015 or self.year == 2016 or self.year == 2017):
            event.met = METObject(event, "MET")
            event.rawmet = METObject(event, "RawMET")
        else:
            event.met = METObject(event, "PuppiMET")
            event.rawmet = METObject(event, "RawPuppiMET")
        event._allFatJets = Collection(event, self._fj_name)
        event.subjets = Collection(event, self._sj_name)  # do not sort subjets after updating!!

        if self._needsJMECorr:
            try:
                rho = event.fixedGridRhoFastjetAll
            except:
                rho = event.Rho_fixedGridRhoFastjetAll
            # correct AK4 jets and MET
            self.jetmetCorr.setSeed(rndSeed(event, event._allJets))
            self.jetmetCorr.correctJetAndMET(jets=event._allJets, lowPtJets=Collection(event, "CorrT1METJet"),
                                             met=event.met, rawMET=event.rawmet,
                                             defaultMET=event.met,
                                             rho=rho, genjets=Collection(event, 'GenJet') if self.isMC else None,
                                             isMC=self.isMC, runNumber=event.run)
            event._allJets = sorted(event._allJets, key=lambda x: x.pt, reverse=True)  # sort by pt after updating

            # correct fatjets
            self.fatjetCorr.setSeed(rndSeed(event, event._allFatJets))
            self.fatjetCorr.correctJetAndMET(jets=event._allFatJets, met=None, rho=rho,
                                             genjets=Collection(event, self._fj_gen_name) if self.isMC else None,
                                             isMC=self.isMC, runNumber=event.run)
            # correct subjets
            self.subjetCorr.setSeed(rndSeed(event, event.subjets))
            self.subjetCorr.correctJetAndMET(jets=event.subjets, met=None, rho=rho,
                                             genjets=Collection(event, self._sj_gen_name) if self.isMC else None,
                                             isMC=self.isMC, runNumber=event.run)

        # jet mass resolution smearing
        if self.isMC and self._jmeSysts['jmr']:
            raise NotImplementedError

        for j in event._allJets:
            j.jetId = 0
            if ((j.neHEF < 0.90) and (j.neEmEF < 0.90) and (j.nConstituents > 1) and (j.chHEF > 0) and (j.chMultiplicity > 0)):
                j.jetId = 2

        # link fatjet to subjets and recompute softdrop mass
        for idx, fj in enumerate(event._allFatJets):
            fj.jetId = 0
            if ((fj.neHEF < 0.90) and (fj.neEmEF < 0.90) and (fj.nConstituents > 1) and (fj.chHEF > 0) and (fj.chMultiplicity > 0)):
                fj.jetId = 2
            fj.idx = idx
            fj.is_qualified = True
            fj.subjets = get_subjets(fj, event.subjets, ('subJetIdx1', 'subJetIdx2'))
            fj.softdrop = sumP4(*fj.subjets).M()
        event._allFatJets = sorted(event._allFatJets, key=lambda x: x.pt, reverse=True)  # sort by pt

        # select lepton-cleaned jets
        event.fatjets = [fj for fj in event._allFatJets if fj.pt > 200 and abs(fj.eta) < 2.4 and (
            fj.jetId & 2) and closest(fj, event.looseLeptons)[1] >= self._jetConeSize]

        event.ak4jets = [j for j in event._allJets if j.pt > 25 and abs(j.eta) < 2.4 and (
            j.jetId & 2) and closest(j, event.looseLeptons)[1] >= 0.4]

        event.ht = sum([j.pt for j in event.ak4jets])

    def selectSV(self, event):
        event._allSV = Collection(event, "SV")
        event.secondary_vertices = []
        for sv in event._allSV:
            if True:
                event.secondary_vertices.append(sv)
        event.secondary_vertices = sorted(event.secondary_vertices, key=lambda x: x.pt, reverse=True)  # sort by pt

    def loadGenHistory(self, event, fatjets):
        # gen matching
        if not self.isMC:
            return

        with open("/afs/cern.ch/user/l/lpaizano/NanoHRT/CMSSW_11_1_0_pre5_PY3/src/PhysicsTools/NanoHRTTools/data/JSON/puWeights_2018.json") as f:
            j = json.load(f)

            content = j["corrections"][0]["data"]["content"]
            nTrueInt = int(round(event.Pileup_nTrueInt))

            for item in content:
                if item["key"] == "nominal":
                    weights_nom = item["value"]["content"]
                elif item["key"] == "up":
                    weights_up = item["value"]["content"]
                elif item["key"] == "down":
                    weights_down = item["value"]["content"]

            weight_nom = weights_nom[nTrueInt]
            weight_up = weights_up[nTrueInt]
            weight_down = weights_down[nTrueInt]

        self.out.fillBranch("puweight_nom",weight_nom)
        self.out.fillBranch("puweight_up",weight_up)
        self.out.fillBranch("puweight_down",weight_down)
        self.out.fillBranch("pileup_nTrueInt",nTrueInt)

        try:
            genparts = event.genparts
        except RuntimeError as e:
            genparts = Collection(event, "GenPart")
            for idx, gp in enumerate(genparts):
                if 'dauIdx' not in gp.__dict__:
                    gp.dauIdx = []
                if gp.genPartIdxMother >= 0:
                    mom = genparts[gp.genPartIdxMother]
                    if 'dauIdx' not in mom.__dict__:
                        mom.dauIdx = [idx]
                    else:
                        mom.dauIdx.append(idx)
            event.genparts = genparts

        def isHadronic(gp):
            if len(gp.dauIdx) == 0:
                raise ValueError('Particle has no daughters!')
            for idx in gp.dauIdx:
                if abs(genparts[idx].pdgId) < 6:
                    return True
            return False

        def isLeptonic(gp):
            if len(gp.dauIdx) == 0:
                raise ValueError('Particle has no daughters!')
            for idx in gp.dauIdx:
                if (abs(genparts[idx].pdgId) >= 11 and abs(genparts[idx].pdgId) <= 14):
                    return True
            return False

        def getFinal(gp):
            for idx in gp.dauIdx:
                dau = genparts[idx]
                if dau.pdgId == gp.pdgId:
                    return getFinal(dau)
            return gp

        GenTaus = []
        GenTaus_Lep = []
        X_Decay = []
        Tau_Lep_Decay = []
        Gen_Vis_Tau1 = []
        Gen_Vis_Tau2 = []
        Gen_Neut1 = []
        Gen_Neut2 = []

        for gp in genparts:
            if gp.statusFlags & (1 << 13) == 0:
                continue
            Tau_Had = 0
            Tau_Lep = 0
            Lep = 0
            if abs(gp.pdgId) == 23 or abs(gp.pdgId) == 24 or abs(gp.pdgId) == 25:
                temp_GenTaus =[]
                temp_GenTaus_Lep =[]
                temp_Gen_Vis_Tau1 = []
                temp_Gen_Vis_Tau2 = []
                temp_Gen_Neut1 = []
                temp_Gen_Neut2 = []
                for idx in gp.dauIdx:
                    dau = genparts[idx]
                    if abs(dau.pdgId) == 15:
                        genTau = getFinal(dau)
                        gp.genTau = genTau
                        temp_GenTaus.append(genTau)
                        if isLeptonic(genTau):
                            Tau_Lep += 1
                            for idy in gp.genTau.dauIdx:
                                daudau = genparts[idy]
                                gp.genTau.daus = daudau
                                if abs(daudau.pdgId) == 11:
                                    temp_GenTaus_Lep.append(daudau)
                                    Lep = 2
                                elif abs(daudau.pdgId) == 13:
                                    temp_GenTaus_Lep.append(daudau)
                                    Lep = 4
                        else:
                            temp_GenTaus_Lep.append(genTau)
                            Tau_Had +=1
                        for idy in gp.genTau.dauIdx:
                            daudau = genparts[idy]
                            gp.genTau.daus = daudau
                            if dau.pdgId == 15:
                                if abs(daudau.pdgId) != 16 and abs(daudau.pdgId) != 14 and abs(daudau.pdgId) != 12:
                                    temp_Gen_Vis_Tau1.append(daudau)
                                else:
                                    temp_Gen_Neut1.append(daudau)
                            else:
                                if abs(daudau.pdgId) != 16 and abs(daudau.pdgId) != 14 and abs(daudau.pdgId) != 12:
                                    temp_Gen_Vis_Tau2.append(daudau)
                                else:
                                    temp_Gen_Neut2.append(daudau)

                if Tau_Had + Tau_Lep == 2:
                    GenTaus = temp_GenTaus
                    GenTaus_Lep = temp_GenTaus_Lep
                    Gen_Vis_Tau1 = temp_Gen_Vis_Tau1
                    Gen_Vis_Tau2 = temp_Gen_Vis_Tau2
                    Gen_Neut1 = temp_Gen_Neut1
                    Gen_Neut2 = temp_Gen_Neut2
                    X_Decay.append(Tau_Had)
                    Tau_Lep_Decay.append(Lep)

        for fj in fatjets:
            fj.dr_genTaus = furthest(fj, GenTaus)
            fj.dr_genTaus_Lep = furthest(fj, GenTaus_Lep)

        tau1 = ROOT.TLorentzVector()
        tau2 = ROOT.TLorentzVector()
        vis_tau1 = ROOT.TLorentzVector()
        vis_tau2 = ROOT.TLorentzVector()
        neutrino1 = ROOT.TLorentzVector()
        neutrino2 = ROOT.TLorentzVector()
        x_decay = -1
        tau_lep = -1

        if(len(Gen_Vis_Tau1) != 0 and len(Gen_Vis_Tau2) != 0):
            x_decay = X_Decay[0]
            tau_lep = Tau_Lep_Decay[0]
            tau1 = GenTaus[0].p4()
            tau2 = GenTaus[1].p4()

            for i in range(len(Gen_Vis_Tau1)):
                vis_tau1 += Gen_Vis_Tau1[i].p4()
            for i in range(len(Gen_Vis_Tau2)):
                vis_tau2 += Gen_Vis_Tau2[i].p4()

            for i in range(len(Gen_Neut1)):
                neutrino1 += Gen_Neut1[i].p4()
            for i in range(len(Gen_Neut2)):
                neutrino2 += Gen_Neut2[i].p4()

        neutrinos = neutrino1 + neutrino2
        ditau = tau1 + tau2
        ditau_vis = vis_tau1 + vis_tau2

        self.out.fillBranch("gen" + "_" + "x_decay",x_decay)
        self.out.fillBranch("gen" + "_" + "tau_lep_decay",tau_lep)

        self.out.fillBranch("gen" + "_" + "tau1_pt",tau1.Pt())
        self.out.fillBranch("gen" + "_" + "tau1_eta",tau1.Eta())
        self.out.fillBranch("gen" + "_" + "tau1_phi",tau1.Phi())
        self.out.fillBranch("gen" + "_" + "tau1_e",tau1.E())

        self.out.fillBranch("gen" + "_" + "tau2_pt",tau2.Pt())
        self.out.fillBranch("gen" + "_" + "tau2_eta",tau2.Eta())
        self.out.fillBranch("gen" + "_" + "tau2_phi",tau2.Phi())
        self.out.fillBranch("gen" + "_" + "tau2_e",tau2.E())

        self.out.fillBranch("gen" + "_" + "vis_tau1_pt",vis_tau1.Pt())
        self.out.fillBranch("gen" + "_" + "vis_tau1_eta",vis_tau1.Eta())
        self.out.fillBranch("gen" + "_" + "vis_tau1_phi",vis_tau1.Phi())
        self.out.fillBranch("gen" + "_" + "vis_tau1_e",vis_tau1.E())

        self.out.fillBranch("gen" + "_" + "vis_tau2_pt",vis_tau2.Pt())
        self.out.fillBranch("gen" + "_" + "vis_tau2_eta",vis_tau2.Eta())
        self.out.fillBranch("gen" + "_" + "vis_tau2_phi",vis_tau2.Phi())
        self.out.fillBranch("gen" + "_" + "vis_tau2_e",vis_tau2.E())

        self.out.fillBranch("gen" + "_" + "neutrino1_pt",neutrino1.Pt())
        self.out.fillBranch("gen" + "_" + "neutrino1_pz",neutrino1.Pz())
        self.out.fillBranch("gen" + "_" + "neutrino1_eta",neutrino1.Eta())
        self.out.fillBranch("gen" + "_" + "neutrino1_phi",neutrino1.Phi())
        self.out.fillBranch("gen" + "_" + "neutrino1_e",neutrino1.E())

        self.out.fillBranch("gen" + "_" + "neutrino2_pt",neutrino2.Pt())
        self.out.fillBranch("gen" + "_" + "neutrino2_pz",neutrino2.Pz())
        self.out.fillBranch("gen" + "_" + "neutrino2_eta",neutrino2.Eta())
        self.out.fillBranch("gen" + "_" + "neutrino2_phi",neutrino2.Phi())
        self.out.fillBranch("gen" + "_" + "neutrino2_e",neutrino2.E())

        self.out.fillBranch("gen" + "_" + "neutrinos_pt",neutrinos.Pt())
        self.out.fillBranch("gen" + "_" + "neutrinos_pz",neutrinos.Pz())
        self.out.fillBranch("gen" + "_" + "neutrinos_eta",neutrinos.Eta())
        self.out.fillBranch("gen" + "_" + "neutrinos_phi",neutrinos.Phi())
        self.out.fillBranch("gen" + "_" + "neutrinos_e",neutrinos.E())

        self.out.fillBranch("gen" + "_" + "ditau_pt",ditau.Pt())
        self.out.fillBranch("gen" + "_" + "ditau_eta",ditau.Eta())
        self.out.fillBranch("gen" + "_" + "ditau_phi",ditau.Phi())
        self.out.fillBranch("gen" + "_" + "ditau_e",ditau.E())
        self.out.fillBranch("gen" + "_" + "ditau_m",ditau.M())

        self.out.fillBranch("gen" + "_" + "ditau_vis_pt",ditau_vis.Pt())
        self.out.fillBranch("gen" + "_" + "ditau_vis_eta",ditau_vis.Eta())
        self.out.fillBranch("gen" + "_" + "ditau_vis_phi",ditau_vis.Phi())
        self.out.fillBranch("gen" + "_" + "ditau_vis_e",ditau_vis.E())
        self.out.fillBranch("gen" + "_" + "ditau_vis_m",ditau_vis.M())

        self.out.fillBranch("gen" + "_" + "met_pt",event.GenMET_pt)
        self.out.fillBranch("gen" + "_" + "met_phi",event.GenMET_phi)

    def evalTagger(self, event, jets):
        for j in jets:
            if self._opts['run_tagger']:
                outputs = [p.predict_with_cache(self.tagInfoMaker, event.idx, j.idx, j) for p in self.pnTaggers]
                outputs = ensemble(outputs, np.mean)
                j.pn_Xbb = outputs['probXbb']
                j.pn_Xcc = outputs['probXcc']
                j.pn_Xqq = outputs['probXqq']
                j.pn_QCD = convert_prob(outputs, None, prefix='prob')
            else:
                if self.hasParticleNetProb:
                    j.pn_Xbb = j.ParticleNetMD_probXbb
                    j.pn_Xcc = j.ParticleNetMD_probXcc
                    j.pn_Xqq = j.ParticleNetMD_probXqq
                    j.pn_QCD = convert_prob(j, None, prefix='ParticleNetMD_prob')
                else:
                    if (self.year == 2015 or self.year == 2016 or self.year == 2017):
                        j.pn_Xbb = j.particleNetMD_Xbb
                        j.pn_Xcc = j.particleNetMD_Xcc
                        j.pn_Xqq = j.particleNetMD_Xqq
                        j.pn_QCD = j.particleNetMD_QCD
                    else:
                        j.pn_Xbb = j.particleNet_XbbVsQCD
                        j.pn_Xcc = j.particleNet_XccVsQCD
                        j.pn_Xqq = j.particleNet_XqqVsQCD
                        j.pn_QCD = j.particleNet_QCD

            j.pn_XbbVsQCD = convert_prob(j, ['Xbb'], ['QCD'], prefix='pn_')
            j.pn_XccVsQCD = convert_prob(j, ['Xcc'], ['QCD'], prefix='pn_')
            j.pn_XccOrXqqVsQCD = convert_prob(j, ['Xcc', 'Xqq'], ['QCD'], prefix='pn_')

    def evalMassRegression(self, event, jets):
        for j in jets:
            if self._opts['run_mass_regression']:
                outputs = [p.predict_with_cache(self.tagInfoMaker, event.idx, j.idx, j) for p in self.pnMassRegressions]
                j.regressed_mass = ensemble(outputs, np.median)['mass']
            else:
                try:
                    j.regressed_mass = j.particleNet_mass
                except RuntimeError:
                    j.regressed_mass = 0

    def fillBaseEventInfo(self, event):
        self.out.fillBranch("jetR", self._jetConeSize)
        self.out.fillBranch("year", self.year)
        self.out.fillBranch("lumiwgt", lumi_dict[self.year])

        met_filters = bool(
            event.Flag_goodVertices and
            event.Flag_globalSuperTightHalo2016Filter and
            event.Flag_HBHENoiseFilter and
            event.Flag_HBHENoiseIsoFilter and
            event.Flag_EcalDeadCellTriggerPrimitiveFilter and
            event.Flag_BadPFMuonFilter and
            event.Flag_BadPFMuonDzFilter and
            event.Flag_eeBadScFilter
        )
        if self.year in (2017, 2018, 20220, 20221, 2024):
            met_filters = met_filters and event.Flag_ecalBadCalibFilter
        self.out.fillBranch("passmetfilters", met_filters)

        # L1 prefire weights
        if self.year <= 2017:
            self.out.fillBranch("l1PreFiringWeight", event.L1PreFiringWeight_Nom)
            self.out.fillBranch("l1PreFiringWeightUp", event.L1PreFiringWeight_Up)
            self.out.fillBranch("l1PreFiringWeightDown", event.L1PreFiringWeight_Dn)
        else:
            self.out.fillBranch("l1PreFiringWeight", 1.0)
            self.out.fillBranch("l1PreFiringWeightUp", 1.0)
            self.out.fillBranch("l1PreFiringWeightDown", 1.0)

        self.out.fillBranch("nlep", len(event.looseLeptons))
        self.out.fillBranch("ht", event.ht)
        self.out.fillBranch("met", event.met.pt)
        self.out.fillBranch("puppi_met", event.PuppiMET_pt)
        self.out.fillBranch("metphi", event.met.phi)
        self.out.fillBranch("puppi_metphi", event.PuppiMET_phi)

        try:
            met_sig = event.met.pt/math.sqrt(event.ht)
        except ZeroDivisionError:
            met_sig = -1

        try:
            Puppi_met_sig = event.PuppiMET_pt/math.sqrt(event.ht)
        except ZeroDivisionError:
            Puppi_met_sig = -1

        self.out.fillBranch("met_significance", met_sig)
        self.out.fillBranch("puppi_met_significance", Puppi_met_sig)
        
    def _get_filler(self, obj):

        def filler(branch, value, default=0):
            self.out.fillBranch(branch, value if obj else default)

        return filler

    def fillFatJetInfo(self, event, fatjets):
        for idx in ([1, 2] if (self._channel == 'qcd' or self._channel == 'ditau') else [1]):
            prefix = 'fj_%d_' % idx
            fj = fatjets[idx - 1]

            if not fj.is_qualified:
                # fill zeros if fatjet fails probe selection
                for b in self.out._branches.keys():
                    if b.startswith(prefix):
                        self.out.fillBranch(b, 0)
                continue
            rawmass = (1-fj.rawFactor)*fj.mass
            # fatjet kinematics
            self.out.fillBranch(prefix + "is_qualified", fj.is_qualified)
            self.out.fillBranch(prefix + "pt", fj.pt)
            self.out.fillBranch(prefix + "eta", fj.eta)
            self.out.fillBranch(prefix + "phi", fj.phi)
            self.out.fillBranch(prefix + "rawfactor", fj.rawFactor)
            self.out.fillBranch(prefix + "mass", fj.mass)
            self.out.fillBranch(prefix + "rawmass", rawmass)
            self.out.fillBranch(prefix + "sdmass", fj.softdrop)
            self.out.fillBranch(prefix + "sdmass_v15", fj.msoftdrop)
            self.out.fillBranch(prefix + "trmass", transverseMass(fj,event.met))
            self.out.fillBranch(prefix + "regressed_mass", fj.regressed_mass)
            self.out.fillBranch(prefix + "tau_regressed_mass", fj.ParticleNet_raw_masscorr*fj.mass)
            self.out.fillBranch(prefix + "ParticleNet_regressed_mass", fj.ParticleNet_raw_masscorr*rawmass)
            self.out.fillBranch(prefix + "globalParT3_massCorrGeneric_regressed_mass", fj.globalParT3_massCorrGeneric*rawmass)
            self.out.fillBranch(prefix + "globalParT3_massCorrX2p_regressed_mass", fj.globalParT3_massCorrX2p*rawmass)
            self.out.fillBranch(prefix + "tau21", fj.tau2 / fj.tau1 if fj.tau1 > 0 else 99)
            self.out.fillBranch(prefix + "tau32", fj.tau3 / fj.tau2 if fj.tau2 > 0 else 99)
            self.out.fillBranch(prefix + "met_dphi", deltaPhi(fj.phi,event.met.phi))

            # subjets
            self.out.fillBranch(prefix + "deltaR_sj12", deltaR(*fj.subjets) if len(fj.subjets) == 2 else 99)
            for idx_sj, sj in enumerate(fj.subjets):
                prefix_sj = prefix + 'sj%d_' % (idx_sj + 1)
                self.out.fillBranch(prefix_sj + "pt", sj.pt)
                self.out.fillBranch(prefix_sj + "eta", sj.eta)
                self.out.fillBranch(prefix_sj + "phi", sj.phi)
                self.out.fillBranch(prefix_sj + "rawmass", sj.mass)
                try:
                    self.out.fillBranch(prefix_sj + "btagdeepcsv", sj.btagDeepB)
                except RuntimeError:
                    self.out.fillBranch(prefix_sj + "btagdeepcsv", -1)

            self.out.fillBranch(prefix + "ParticleNet_newlabel_raw_probHbb", fj.ParticleNet_newlabel_raw_probHbb)
            self.out.fillBranch(prefix + "ParticleNet_newlabel_raw_probHcc", fj.ParticleNet_newlabel_raw_probHcc)
            self.out.fillBranch(prefix + "ParticleNet_newlabel_raw_probHee", fj.ParticleNet_newlabel_raw_probHee)
            self.out.fillBranch(prefix + "ParticleNet_newlabel_raw_probHem", fj.ParticleNet_newlabel_raw_probHem)
            self.out.fillBranch(prefix + "ParticleNet_newlabel_raw_probHgg", fj.ParticleNet_newlabel_raw_probHgg)
            self.out.fillBranch(prefix + "ParticleNet_newlabel_raw_probHmm", fj.ParticleNet_newlabel_raw_probHmm)
            self.out.fillBranch(prefix + "ParticleNet_newlabel_raw_probHqq", fj.ParticleNet_newlabel_raw_probHqq)
            self.out.fillBranch(prefix + "ParticleNet_newlabel_raw_probHte", fj.ParticleNet_newlabel_raw_probHte)
            self.out.fillBranch(prefix + "ParticleNet_newlabel_raw_probHtm", fj.ParticleNet_newlabel_raw_probHtm)
            self.out.fillBranch(prefix + "ParticleNet_newlabel_raw_probHtt", fj.ParticleNet_newlabel_raw_probHtt)
            self.out.fillBranch(prefix + "ParticleNet_newlabel_raw_probQCD0hf", fj.ParticleNet_newlabel_raw_probQCD0hf)
            self.out.fillBranch(prefix + "ParticleNet_newlabel_raw_probQCD1hf", fj.ParticleNet_newlabel_raw_probQCD1hf)
            self.out.fillBranch(prefix + "ParticleNet_newlabel_raw_probQCD2hf", fj.ParticleNet_newlabel_raw_probQCD2hf)
            self.out.fillBranch(prefix + "ParticleNet_newlabel_raw_probSingleTau", fj.ParticleNet_newlabel_raw_probSingleTau)

            self.out.fillBranch(prefix + "ParticleNet_newlabelwjets_raw_probHbb", fj.ParticleNet_newlabelwjets_raw_probHbb)
            self.out.fillBranch(prefix + "ParticleNet_newlabelwjets_raw_probHcc", fj.ParticleNet_newlabelwjets_raw_probHcc)
            self.out.fillBranch(prefix + "ParticleNet_newlabelwjets_raw_probHee", fj.ParticleNet_newlabelwjets_raw_probHee)
            self.out.fillBranch(prefix + "ParticleNet_newlabelwjets_raw_probHem", fj.ParticleNet_newlabelwjets_raw_probHem)
            self.out.fillBranch(prefix + "ParticleNet_newlabelwjets_raw_probHgg", fj.ParticleNet_newlabelwjets_raw_probHgg)
            self.out.fillBranch(prefix + "ParticleNet_newlabelwjets_raw_probHmm", fj.ParticleNet_newlabelwjets_raw_probHmm)
            self.out.fillBranch(prefix + "ParticleNet_newlabelwjets_raw_probHqq", fj.ParticleNet_newlabelwjets_raw_probHqq)
            self.out.fillBranch(prefix + "ParticleNet_newlabelwjets_raw_probHte", fj.ParticleNet_newlabelwjets_raw_probHte)
            self.out.fillBranch(prefix + "ParticleNet_newlabelwjets_raw_probHtm", fj.ParticleNet_newlabelwjets_raw_probHtm)
            self.out.fillBranch(prefix + "ParticleNet_newlabelwjets_raw_probHtt", fj.ParticleNet_newlabelwjets_raw_probHtt)
            self.out.fillBranch(prefix + "ParticleNet_newlabelwjets_raw_probQCD0hf", fj.ParticleNet_newlabelwjets_raw_probQCD0hf)
            self.out.fillBranch(prefix + "ParticleNet_newlabelwjets_raw_probQCD1hf", fj.ParticleNet_newlabelwjets_raw_probQCD1hf)
            self.out.fillBranch(prefix + "ParticleNet_newlabelwjets_raw_probQCD2hf", fj.ParticleNet_newlabelwjets_raw_probQCD2hf)
            self.out.fillBranch(prefix + "ParticleNet_newlabelwjets_raw_probSingleTau", fj.ParticleNet_newlabelwjets_raw_probSingleTau)

            self.out.fillBranch(prefix + "ParticleNet_raw_masscorr", fj.ParticleNet_raw_masscorr)
            self.out.fillBranch(prefix + "ParticleNet_raw_probHbb", fj.ParticleNet_raw_probHbb)
            self.out.fillBranch(prefix + "ParticleNet_raw_probHcc", fj.ParticleNet_raw_probHcc)
            self.out.fillBranch(prefix + "ParticleNet_raw_probHgg", fj.ParticleNet_raw_probHgg)
            self.out.fillBranch(prefix + "ParticleNet_raw_probHqq", fj.ParticleNet_raw_probHqq)
            self.out.fillBranch(prefix + "ParticleNet_raw_probHte", fj.ParticleNet_raw_probHte)
            self.out.fillBranch(prefix + "ParticleNet_raw_probHtm", fj.ParticleNet_raw_probHtm)
            self.out.fillBranch(prefix + "ParticleNet_raw_probHtt", fj.ParticleNet_raw_probHtt)
            self.out.fillBranch(prefix + "ParticleNet_raw_probQCD0hf", fj.ParticleNet_raw_probQCD0hf)
            self.out.fillBranch(prefix + "ParticleNet_raw_probQCD1hf", fj.ParticleNet_raw_probQCD1hf)
            self.out.fillBranch(prefix + "ParticleNet_raw_probQCD2hf", fj.ParticleNet_raw_probQCD2hf)

            self.out.fillBranch(prefix + "globalParT3_QCD", fj.globalParT3_QCD)
            self.out.fillBranch(prefix + "globalParT3_TopbWev", fj.globalParT3_TopbWev)
            self.out.fillBranch(prefix + "globalParT3_TopbWmv", fj.globalParT3_TopbWmv)
            self.out.fillBranch(prefix + "globalParT3_TopbWq", fj.globalParT3_TopbWq)
            self.out.fillBranch(prefix + "globalParT3_TopbWqq", fj.globalParT3_TopbWqq)
            self.out.fillBranch(prefix + "globalParT3_TopbWtauhv", fj.globalParT3_TopbWtauhv)
            self.out.fillBranch(prefix + "globalParT3_WvsQCD", fj.globalParT3_WvsQCD)
            self.out.fillBranch(prefix + "globalParT3_XWW3q", fj.globalParT3_XWW3q)
            self.out.fillBranch(prefix + "globalParT3_XWW4q", fj.globalParT3_XWW4q)
            self.out.fillBranch(prefix + "globalParT3_XWWqqev", fj.globalParT3_XWWqqev)
            self.out.fillBranch(prefix + "globalParT3_XWWqqmv", fj.globalParT3_XWWqqmv)
            self.out.fillBranch(prefix + "globalParT3_Xbb", fj.globalParT3_Xbb)
            self.out.fillBranch(prefix + "globalParT3_Xcc", fj.globalParT3_Xcc)
            self.out.fillBranch(prefix + "globalParT3_Xcs", fj.globalParT3_Xcs)
            self.out.fillBranch(prefix + "globalParT3_Xqq", fj.globalParT3_Xqq)
            self.out.fillBranch(prefix + "globalParT3_Xtauhtaue", fj.globalParT3_Xtauhtaue)
            self.out.fillBranch(prefix + "globalParT3_Xtauhtauh", fj.globalParT3_Xtauhtauh)
            self.out.fillBranch(prefix + "globalParT3_Xtauhtaum", fj.globalParT3_Xtauhtaum)
            self.out.fillBranch(prefix + "globalParT3_massCorrGeneric", fj.globalParT3_massCorrGeneric)
            self.out.fillBranch(prefix + "globalParT3_massCorrX2p", fj.globalParT3_massCorrX2p)
            self.out.fillBranch(prefix + "globalParT3_withMassTopvsQCD", fj.globalParT3_withMassTopvsQCD)
            self.out.fillBranch(prefix + "globalParT3_withMassWvsQCD", fj.globalParT3_withMassWvsQCD)
            self.out.fillBranch(prefix + "globalParT3_withMassZvsQCD", fj.globalParT3_withMassZvsQCD)

            self.out.fillBranch(prefix + "particleNetLegacy_QCD", fj.particleNetLegacy_QCD)
            self.out.fillBranch(prefix + "particleNetLegacy_Xqq", fj.particleNetLegacy_Xqq)
            self.out.fillBranch(prefix + "particleNetLegacy_Xbb", fj.particleNetLegacy_Xbb)
            self.out.fillBranch(prefix + "particleNetLegacy_Xcc", fj.particleNetLegacy_Xcc)
            self.out.fillBranch(prefix + "particleNetLegacy_mass", fj.particleNetLegacy_mass)

            self.out.fillBranch(prefix + "particleNetWithMass_H4qvsQCD", fj.particleNetWithMass_H4qvsQCD)
            self.out.fillBranch(prefix + "particleNetWithMass_HbbvsQCD", fj.particleNetWithMass_HbbvsQCD)
            self.out.fillBranch(prefix + "particleNetWithMass_HccvsQCD", fj.particleNetWithMass_HccvsQCD)
            self.out.fillBranch(prefix + "particleNetWithMass_QCD", fj.particleNetWithMass_QCD)
            self.out.fillBranch(prefix + "particleNetWithMass_TvsQCD", fj.particleNetWithMass_TvsQCD)
            self.out.fillBranch(prefix + "particleNetWithMass_WvsQCD", fj.particleNetWithMass_WvsQCD)
            self.out.fillBranch(prefix + "particleNetWithMass_ZvsQCD", fj.particleNetWithMass_ZvsQCD)

            self.out.fillBranch(prefix + "particleNet_QCD", fj.particleNet_QCD)
            self.out.fillBranch(prefix + "particleNet_QCD0HF", fj.particleNet_QCD0HF)
            self.out.fillBranch(prefix + "particleNet_QCD1HF", fj.particleNet_QCD1HF)
            self.out.fillBranch(prefix + "particleNet_QCD2HF", fj.particleNet_QCD2HF)
            self.out.fillBranch(prefix + "particleNet_WVsQCD", fj.particleNet_WVsQCD)
            self.out.fillBranch(prefix + "particleNet_XbbVsQCD", fj.particleNet_XbbVsQCD)
            self.out.fillBranch(prefix + "particleNet_XccVsQCD", fj.particleNet_XccVsQCD)
            self.out.fillBranch(prefix + "particleNet_XggVsQCD", fj.particleNet_XggVsQCD)
            self.out.fillBranch(prefix + "particleNet_XqqVsQCD", fj.particleNet_XqqVsQCD)
            self.out.fillBranch(prefix + "particleNet_XteVsQCD", fj.particleNet_XteVsQCD)
            self.out.fillBranch(prefix + "particleNet_XtmVsQCD", fj.particleNet_XtmVsQCD)
            self.out.fillBranch(prefix + "particleNet_XttVsQCD", fj.particleNet_XttVsQCD)
            self.out.fillBranch(prefix + "particleNet_masscorr", fj.particleNet_massCorr)

            ParticleNet_newlabel_xtt = -math.log10(1 - fj.ParticleNet_newlabel_raw_probHtt + 1e-18)
            ParticleNet_newlabelwjets_xtt = -math.log10(1 - fj.ParticleNet_newlabelwjets_raw_probHtt + 1e-18)
            ParticleNet_xtt = -math.log10(1 - fj.ParticleNet_raw_probHtt + 1e-18)
            globalParT3_xtt = -math.log10(1 - fj.globalParT3_Xtauhtauh + 1e-18)
            particleNet_xttvsqcd = -math.log10(1 - fj.particleNet_XttVsQCD + 1e-18)

            ParticleNet_newlabel_xtm = -math.log10(1 - fj.ParticleNet_newlabel_raw_probHtm + 1e-18)
            ParticleNet_newlabelwjets_xtm = -math.log10(1 - fj.ParticleNet_newlabelwjets_raw_probHtm + 1e-18)
            ParticleNet_xtm = -math.log10(1 - fj.ParticleNet_raw_probHtm + 1e-18)
            globalParT3_xtm = -math.log10(1 - fj.globalParT3_Xtauhtaum + 1e-18)
            particleNet_xtmvsqcd = -math.log10(1 - fj.particleNet_XtmVsQCD + 1e-18)

            ParticleNet_newlabel_xte = -math.log10(1 - fj.ParticleNet_newlabel_raw_probHte + 1e-18)
            ParticleNet_newlabelwjets_xte = -math.log10(1 - fj.ParticleNet_newlabelwjets_raw_probHte + 1e-18)
            ParticleNet_xte = -math.log10(1 - fj.ParticleNet_raw_probHte + 1e-18)
            globalParT3_xte = -math.log10(1 - fj.globalParT3_Xtauhtaue + 1e-18)
            particleNet_xtevsqcd = -math.log10(1 - fj.particleNet_XteVsQCD + 1e-18)

            ParticleNet_newlabel_raw_probSingleTau = -math.log10(1 - fj.ParticleNet_newlabel_raw_probSingleTau + 1e-18)
            ParticleNet_newlabelwjets_raw_probSingleTau = -math.log10(1 - fj.ParticleNet_newlabelwjets_raw_probSingleTau + 1e-18)

            self.out.fillBranch(prefix + "nine_ParticleNet_newlabel_raw_probHte", ParticleNet_newlabel_xte)
            self.out.fillBranch(prefix + "nine_ParticleNet_newlabel_raw_probHtm", ParticleNet_newlabel_xtm)
            self.out.fillBranch(prefix + "nine_ParticleNet_newlabel_raw_probHtt", ParticleNet_newlabel_xtt)
            self.out.fillBranch(prefix + "nine_ParticleNet_newlabel_raw_probSingleTau", ParticleNet_newlabel_raw_probSingleTau)

            self.out.fillBranch(prefix + "nine_ParticleNet_newlabelwjets_raw_probHte", ParticleNet_newlabelwjets_xte)
            self.out.fillBranch(prefix + "nine_ParticleNet_newlabelwjets_raw_probHtm", ParticleNet_newlabelwjets_xtm)
            self.out.fillBranch(prefix + "nine_ParticleNet_newlabelwjets_raw_probHtt", ParticleNet_newlabelwjets_xtt)
            self.out.fillBranch(prefix + "nine_ParticleNet_newlabelwjets_raw_probSingleTau", ParticleNet_newlabelwjets_raw_probSingleTau)

            self.out.fillBranch(prefix + "nine_ParticleNet_raw_probHte", ParticleNet_xte)
            self.out.fillBranch(prefix + "nine_ParticleNet_raw_probHtm", ParticleNet_xtm)
            self.out.fillBranch(prefix + "nine_ParticleNet_raw_probHtt", ParticleNet_xtt)

            self.out.fillBranch(prefix + "nine_globalParT3_Xtauhtaue", globalParT3_xte)
            self.out.fillBranch(prefix + "nine_globalParT3_Xtauhtauh", globalParT3_xtm)
            self.out.fillBranch(prefix + "nine_globalParT3_Xtauhtaum", globalParT3_xtt)

            self.out.fillBranch(prefix + "nine_particleNet_XteVsQCD", particleNet_xtevsqcd)
            self.out.fillBranch(prefix + "nine_particleNet_XtmVsQCD", particleNet_xtmvsqcd)
            self.out.fillBranch(prefix + "nine_particleNet_XttVsQCD", particleNet_xttvsqcd)

            ParticleNet_newlabel_raw_probQCD = fj.ParticleNet_newlabel_raw_probQCD0hf + fj.ParticleNet_newlabel_raw_probQCD1hf + fj.ParticleNet_newlabel_raw_probQCD2hf
            ParticleNet_newlabel_sameflavor = sameflavor(fj.ParticleNet_newlabel_raw_probHtt,fj.ParticleNet_newlabel_raw_probHmm,fj.ParticleNet_newlabel_raw_probHee,fj.ParticleNet_newlabel_raw_probHgg,fj.ParticleNet_newlabel_raw_probHqq,fj.ParticleNet_newlabel_raw_probHcc,fj.ParticleNet_newlabel_raw_probHbb, ParticleNet_newlabel_raw_probQCD)

            ParticleNet_newlabelwjets_raw_probQCD = fj.ParticleNet_newlabelwjets_raw_probQCD0hf + fj.ParticleNet_newlabelwjets_raw_probQCD1hf + fj.ParticleNet_newlabelwjets_raw_probQCD2hf
            ParticleNet_newlabelwjets_sameflavor = sameflavor(fj.ParticleNet_newlabelwjets_raw_probHtt,fj.ParticleNet_newlabelwjets_raw_probHmm,fj.ParticleNet_newlabelwjets_raw_probHee,fj.ParticleNet_newlabelwjets_raw_probHgg,fj.ParticleNet_newlabelwjets_raw_probHqq,fj.ParticleNet_newlabelwjets_raw_probHcc,fj.ParticleNet_newlabelwjets_raw_probHbb, ParticleNet_newlabelwjets_raw_probQCD)

            self.out.fillBranch(prefix + "ParticleNet_newlabel_Httvssf", ParticleNet_newlabel_sameflavor)
            self.out.fillBranch(prefix + "ParticleNet_newlabelwjetsHttvssf", ParticleNet_newlabelwjets_sameflavor)

            ParticleNet_newlabel_emg = emg(fj.ParticleNet_newlabel_raw_probHtt,fj.ParticleNet_newlabel_raw_probHmm,fj.ParticleNet_newlabel_raw_probHee,fj.ParticleNet_newlabel_raw_probHgg)
            ParticleNet_newlabelwjets_emg = emg(fj.ParticleNet_newlabelwjets_raw_probHtt,fj.ParticleNet_newlabelwjets_raw_probHmm,fj.ParticleNet_newlabelwjets_raw_probHee,fj.ParticleNet_newlabelwjets_raw_probHgg)

            self.out.fillBranch(prefix + "ParticleNet_newlabel_Httvsemg", ParticleNet_newlabel_emg)
            self.out.fillBranch(prefix + "ParticleNet_newlabelwjetsHttvsemg", ParticleNet_newlabelwjets_emg)

            nine_newlabel_Httvssf = -math.log10(1 - ParticleNet_newlabel_sameflavor + 1e-18)
            nine_newlabelwjets_Httvssf = -math.log10(1 - ParticleNet_newlabelwjets_sameflavor + 1e-18)

            self.out.fillBranch(prefix + "nine_ParticleNet_newlabel_Httvssf", nine_newlabel_Httvssf)
            self.out.fillBranch(prefix + "nine_ParticleNet_newlabelwjetsHttvssf", nine_newlabelwjets_Httvssf)

            nine_newlabel_Httvsemg = -math.log10(1 - ParticleNet_newlabel_emg + 1e-18)
            nine_newlabelwjets_Httvsemg = -math.log10(1 - ParticleNet_newlabelwjets_emg + 1e-18)

            self.out.fillBranch(prefix + "nine_ParticleNet_newlabel_Httvsemg", nine_newlabel_Httvsemg)
            self.out.fillBranch(prefix + "nine_ParticleNet_newlabelwjetsHttvsemg", nine_newlabelwjets_Httvsemg)

            if self._opts['run_tagger']:
                self.out.fillBranch(prefix + "origParticleNetMD_XccVsQCD",
                                    convert_prob(fj, ['Xcc'], None, prefix='ParticleNetMD_prob'))
                self.out.fillBranch(prefix + "origParticleNetMD_XbbVsQCD",
                                    convert_prob(fj, ['Xbb'], None, prefix='ParticleNetMD_prob'))
