// ----------------------------------------------------------------------------
// openCARP is an open cardiac electrophysiology simulator.
//
// Copyright (C) 2020 openCARP project
//
// This program is licensed under the openCARP Academic Public License (APL)
// v1.0: You can use and redistribute it and/or modify it in non-commercial
// academic environments under the terms of APL as published by the openCARP
// project v1.0, or (at your option) any later version. Commercial use requires
// a commercial license (info@opencarp.org).
//
// This program is distributed without any warranty; see the openCARP APL for
// more details.
//
// You should have received a copy of the openCARP APL along with this program
// and can find it online: http://www.opencarp.org/license
// ----------------------------------------------------------------------------


/*
*
*  Authors: Jakub Tomek, Alfonso Bueno-Orovio, Elisa Passini, Xin Zhou, Ana Minchole, Oliver Britton, Chiara Bartolucci, Stefano Severi, Alvin Shrier, Laszlo Virag, Andras Varro, Blanca Rodriguez
*  Year: 2019
*  Title: Development, calibration, and validation of a novel human ventricular myocyte model in health, disease, and drug block
*  Journal: eLife
*  DOI: 10.7554/eLife.48890
*  Comment: 
*
*/
        

// DO NOT EDIT THIS SOURCE CODE FILE
// ANY CHANGES TO THIS FILE WILL BE OVERWRITTEN!!!!

#include "Tomek_edit.h"

#ifdef _OPENMP
#include <omp.h>
#endif



namespace limpet {

using ::opencarp::f_open;
using ::opencarp::FILE_SPEC;

Tomek_editIonType::Tomek_editIonType(bool plugin) : IonType(std::move(std::string("Tomek_edit")), plugin) {}

size_t Tomek_editIonType::params_size() const {
  return sizeof(struct Tomek_edit_Params);
}

size_t Tomek_editIonType::dlo_vector_size() const {

  return 1;
}

uint32_t Tomek_editIonType::reqdat() const {
  return Tomek_edit_REQDAT;
}

uint32_t Tomek_editIonType::moddat() const {
  return Tomek_edit_MODDAT;
}

void Tomek_editIonType::destroy(IonIfBase& imp_base) const {
  IonIfDerived& imp = static_cast<IonIfDerived&>(imp_base);
  imp.destroy_luts();
  // rarely need to do anything else
}

Target Tomek_editIonType::select_target(Target target) const {
  switch (target) {
    case Target::AUTO:
#   ifdef TOMEK_EDIT_MLIR_CUDA_GENERATED
      return Target::MLIR_CUDA;
#   elif defined(TOMEK_EDIT_MLIR_ROCM_GENERATED)
      return Target::MLIR_ROCM;
#   elif defined(TOMEK_EDIT_MLIR_CPU_GENERATED)
      return Target::MLIR_CPU;
#   elif defined(TOMEK_EDIT_CPU_GENERATED)
      return Target::CPU;
#   else
      return Target::UNKNOWN;
#   endif
#   ifdef TOMEK_EDIT_MLIR_CUDA_GENERATED
    case Target::MLIR_CUDA:
      return Target::MLIR_CUDA;
#   endif
#   ifdef TOMEK_EDIT_MLIR_ROCM_GENERATED
    case Target::MLIR_ROCM:
      return Target::MLIR_ROCM;
#   endif
#   ifdef TOMEK_EDIT_MLIR_CPU_GENERATED
    case Target::MLIR_CPU:
      return Target::MLIR_CPU;
#   endif
#   ifdef TOMEK_EDIT_CPU_GENERATED
    case Target::CPU:
      return Target::CPU;
#   endif
    default:
      return Target::UNKNOWN;
  }
}

void Tomek_editIonType::compute(Target target, int start, int end, IonIfBase& imp_base, GlobalData_t** data) const {
  IonIfDerived& imp = static_cast<IonIfDerived&>(imp_base);
  switch(target) {
    case Target::AUTO:
#   ifdef TOMEK_EDIT_MLIR_CUDA_GENERATED
      compute_Tomek_edit_mlir_gpu_cuda(start, end, imp, data);
#   elif defined(TOMEK_EDIT_MLIR_ROCM_GENERATED)
      compute_Tomek_edit_mlir_gpu_rocm(start, end, imp, data);
#   elif defined(TOMEK_EDIT_MLIR_CPU_GENERATED)
      compute_Tomek_edit_mlir_cpu(start, end, imp, data);
#   elif defined(TOMEK_EDIT_CPU_GENERATED)
      compute_Tomek_edit_cpu(start, end, imp, data);
#   else
#     error "Could not generate method Tomek_editIonType::compute."
#   endif
      break;
#   ifdef TOMEK_EDIT_MLIR_CUDA_GENERATED
    case Target::MLIR_CUDA:
      compute_Tomek_edit_mlir_gpu_cuda(start, end, imp, data);
      break;
#   endif
#   ifdef TOMEK_EDIT_MLIR_ROCM_GENERATED
    case Target::MLIR_ROCM:
      compute_Tomek_edit_mlir_gpu_rocm(start, end, imp, data);
      break;
#   endif
#   ifdef TOMEK_EDIT_MLIR_CPU_GENERATED
    case Target::MLIR_CPU:
      compute_Tomek_edit_mlir_cpu(start, end, imp, data);
      break;
#   endif
#   ifdef TOMEK_EDIT_CPU_GENERATED
    case Target::CPU:
      compute_Tomek_edit_cpu(start, end, imp, data);
      break;
#   endif
    default:
      throw std::runtime_error(std::string("Could not compute with the given target ") + get_string_from_target(target) + ".");
      break;
  }
}

// Define all constants
#define A_atp (GlobalData_t)(2.)
#define Aff (GlobalData_t)(0.6)
#define BSLmax (GlobalData_t)(1.124)
#define BSRmax (GlobalData_t)(0.047)
#define C1_init (GlobalData_t)(7.0344e-4)
#define C2_init (GlobalData_t)(8.5109e-4)
#define C3_init (GlobalData_t)(0.9981)
#define CaMKt_init (GlobalData_t)(0.0111)
#define Cai_init (GlobalData_t)(8.1583e-05)
#define Cajsr_init (GlobalData_t)(1.5214)
#define Cansr_init (GlobalData_t)(1.5211)
#define Cao (GlobalData_t)(1.8)
#define Cass_init (GlobalData_t)(7.0305e-5)
#define Cli (GlobalData_t)(24.0)
#define Clo (GlobalData_t)(150.0)
#define EKshift (GlobalData_t)(0.)
#define ENDO (GlobalData_t)(0.)
#define EPI (GlobalData_t)(1.)
#define F (GlobalData_t)(96485.)
#define Fjunc (GlobalData_t)(1.)
#define GClCa (GlobalData_t)(0.2843)
#define GClb (GlobalData_t)(1.98e-3)
#define GKb_b (GlobalData_t)(0.0189)
#define GKs_b (GlobalData_t)(0.0011)
#define GNa (GlobalData_t)(11.7802)
#define GpCa (GlobalData_t)(5e-04)
#define H (GlobalData_t)(1e-7)
#define ICaL_fractionSS (GlobalData_t)(0.8)
#define INaCa_fractionSS (GlobalData_t)(0.35)
#define I_init (GlobalData_t)(1.3289e-5)
#define Jrel_np_init (GlobalData_t)(1.6129e-22)
#define Jrel_p_init (GlobalData_t)(1.2475e-20)
#define K1m (GlobalData_t)(182.4)
#define K1p (GlobalData_t)(949.5)
#define K2m (GlobalData_t)(39.4)
#define K2n (GlobalData_t)(500.)
#define K2p (GlobalData_t)(687.2)
#define K3m (GlobalData_t)(79300.)
#define K3p (GlobalData_t)(1899.)
#define K4m (GlobalData_t)(40.)
#define K4p (GlobalData_t)(639.)
#define KCaoff (GlobalData_t)(5e3)
#define KCaon (GlobalData_t)(1.5e6)
#define KKi (GlobalData_t)(0.5)
#define KKo (GlobalData_t)(0.3582)
#define KNa1 (GlobalData_t)(15.)
#define KNa2 (GlobalData_t)(5.)
#define KNa3 (GlobalData_t)(88.12)
#define KNai0 (GlobalData_t)(9.073)
#define KNao0 (GlobalData_t)(27.78)
#define KNap (GlobalData_t)(224.)
#define K_atp (GlobalData_t)(0.25)
#define K_o_n (GlobalData_t)(5.)
#define Kasymm (GlobalData_t)(12.5)
#define KdClCa (GlobalData_t)(0.1)
#define Khp (GlobalData_t)(1.698e-7)
#define Ki_init (GlobalData_t)(142.3002)
#define KmBSL (GlobalData_t)(0.0087)
#define KmBSR (GlobalData_t)(0.00087)
#define KmCaAct (GlobalData_t)(150e-6)
#define KmCaM (GlobalData_t)(0.0015)
#define KmCaMK (GlobalData_t)(0.15)
#define KmCap (GlobalData_t)(0.0005)
#define Kmcmdn (GlobalData_t)(0.00238)
#define Kmcsqn (GlobalData_t)(0.8)
#define Kmgatp (GlobalData_t)(1.698e-7)
#define Kmn (GlobalData_t)(0.002)
#define Kmtrpn (GlobalData_t)(0.0005)
#define Ko (GlobalData_t)(5.0)
#define Kss_init (GlobalData_t)(142.3002)
#define KxKur (GlobalData_t)(292.)
#define L (GlobalData_t)(0.01)
#define MCELL (GlobalData_t)(2.)
#define MgADP (GlobalData_t)(0.05)
#define MgATP (GlobalData_t)(9.8)
#define Nai_init (GlobalData_t)(12.1025)
#define Nao (GlobalData_t)(140.0)
#define Nass_init (GlobalData_t)(12.1029)
#define O_init (GlobalData_t)(3.7585e-4)
#define PCa_b (GlobalData_t)(8.3757e-05)
#define PCab (GlobalData_t)(5.9194e-08)
#define PKNa (GlobalData_t)(0.01833)
#define PNab (GlobalData_t)(1.9239e-09)
#define R (GlobalData_t)(8314.)
#define T (GlobalData_t)(310.)
#define VShift (GlobalData_t)(0.)
#define V_init (GlobalData_t)(-88.7638)
#define aCaMK (GlobalData_t)(0.05)
#define a_init (GlobalData_t)(9.5098e-4)
#define alpha_1 (GlobalData_t)(0.154375)
#define ap_init (GlobalData_t)(4.8454e-4)
#define bCaMK (GlobalData_t)(0.00068)
#define beta_1 (GlobalData_t)(0.1911)
#define bt (GlobalData_t)(4.75)
#define cmdnmax_b (GlobalData_t)(0.05)
#define csqnmax (GlobalData_t)(10.)
#define d_init (GlobalData_t)(8.1084e-9)
#define delta (GlobalData_t)(-0.155)
#define dielConstant (GlobalData_t)(74.)
#define eP (GlobalData_t)(4.2)
#define fCaf_init (GlobalData_t)(1.0)
#define fCafp_init (GlobalData_t)(1.0)
#define fCas_init (GlobalData_t)(0.9999)
#define fKatp (GlobalData_t)(0.0)
#define ff_init (GlobalData_t)(1.0)
#define ffp_init (GlobalData_t)(1.0)
#define fs_init (GlobalData_t)(0.939)
#define gKatp (GlobalData_t)(4.3195)
#define hL_init (GlobalData_t)(0.5255)
#define hLp_init (GlobalData_t)(0.2872)
#define h_init (GlobalData_t)(0.8286)
#define hp_init (GlobalData_t)(0.6707)
#define iF_init (GlobalData_t)(0.9996)
#define iFp_init (GlobalData_t)(0.9996)
#define iS_init (GlobalData_t)(0.5936)
#define iSp_init (GlobalData_t)(0.6538)
#define jCa_init (GlobalData_t)(1.0)
#define j_init (GlobalData_t)(0.8284)
#define jp_init (GlobalData_t)(0.8281)
#define mL_init (GlobalData_t)(1.629e-4)
#define m_init (GlobalData_t)(8.0572e-4)
#define nCa_i_init (GlobalData_t)(0.0012)
#define nCa_ss_init (GlobalData_t)(6.6462e-4)
#define offset (GlobalData_t)(0.)
#define qCa (GlobalData_t)(0.167)
#define qNa (GlobalData_t)(0.5224)
#define rad (GlobalData_t)(0.0011)
#define tauCa (GlobalData_t)(0.2)
#define tauK (GlobalData_t)(2.0)
#define tauNa (GlobalData_t)(2.0)
#define tjCa (GlobalData_t)(75.)
#define trpnmax (GlobalData_t)(0.07)
#define wCa (GlobalData_t)(6e4)
#define wNa (GlobalData_t)(6e4)
#define wNaCa (GlobalData_t)(5e3)
#define xs1_init (GlobalData_t)(0.248)
#define xs2_init (GlobalData_t)(1.7707e-4)
#define zCa (GlobalData_t)(2.)
#define zK (GlobalData_t)(1.)
#define zNa (GlobalData_t)(1.)
#define zcl (GlobalData_t)(-1.)
#define Afs (GlobalData_t)((1.-(Aff)))
#define Ageo (GlobalData_t)(((((2.*3.14)*rad)*rad)+(((2.*3.14)*rad)*L)))
#define ECl (GlobalData_t)((((R*T)/(zcl*F))*(log((Clo/Cli)))))
#define Io (GlobalData_t)(((0.5*(((Nao+Ko)+Clo)+(4.*Cao)))/1000.))
#define K2_i (GlobalData_t)(KCaoff)
#define K2_ss (GlobalData_t)(KCaoff)
#define K5_i (GlobalData_t)(KCaoff)
#define K5_ss (GlobalData_t)(KCaoff)
#define Vcell (GlobalData_t)(((((1000.*3.14)*rad)*rad)*L))
#define a2 (GlobalData_t)(K2p)
#define a4 (GlobalData_t)((((K4p*MgATP)/Kmgatp)/(1.+(MgATP/Kmgatp))))
#define aKiK (GlobalData_t)((pow((Ko/K_o_n),0.24)))
#define a_rel (GlobalData_t)((0.5*bt))
#define b1 (GlobalData_t)((K1m*MgADP))
#define bKiK (GlobalData_t)((1./(1.+((A_atp/K_atp)*(A_atp/K_atp)))))
#define btp (GlobalData_t)((1.25*bt))
#define constA (GlobalData_t)((1.82e6*(pow((dielConstant*T),-1.5))))
#define h10_i (GlobalData_t)(((Kasymm+1.)+((Nao/KNa1)*(1.+(Nao/KNa2)))))
#define h10_ss (GlobalData_t)(((Kasymm+1.)+((Nao/KNa1)*(1.+(Nao/KNa2)))))
#define ACap (GlobalData_t)((2.*Ageo))
#define Vjsr (GlobalData_t)((0.0048*Vcell))
#define Vmyo (GlobalData_t)((0.68*Vcell))
#define Vnsr (GlobalData_t)((0.0552*Vcell))
#define Vss (GlobalData_t)((0.02*Vcell))
#define a_relp (GlobalData_t)((0.5*btp))
#define gamma_Cao (GlobalData_t)((exp((((-constA)*4.)*(((sqrt(Io))/(1.+(sqrt(Io))))-((0.3*Io)))))))
#define gamma_Ko (GlobalData_t)((exp(((-constA)*(((sqrt(Io))/(1.+(sqrt(Io))))-((0.3*Io)))))))
#define gamma_Nao (GlobalData_t)((exp(((-constA)*(((sqrt(Io))/(1.+(sqrt(Io))))-((0.3*Io)))))))
#define h11_i (GlobalData_t)(((Nao*Nao)/((h10_i*KNa1)*KNa2)))
#define h11_ss (GlobalData_t)(((Nao*Nao)/((h10_ss*KNa1)*KNa2)))
#define h12_i (GlobalData_t)((1./h10_i))
#define h12_ss (GlobalData_t)((1./h10_ss))
#define K1_i (GlobalData_t)(((h12_i*Cao)*KCaon))
#define K1_ss (GlobalData_t)(((h12_ss*Cao)*KCaon))



void Tomek_editIonType::initialize_params(IonIfBase& imp_base) const
{
  IonIfDerived& imp = static_cast<IonIfDerived&>(imp_base);
  cell_geom* region = &imp.cgeom();
  Tomek_edit_Params *p = imp.params();

  // Compute the regional constants
  {
    p->CaMKo = 0.05;
    p->Cajsr_half = 1.7;
    p->GK1_b = 0.6992;
    p->GKr_b = 0.0321;
    p->GNaL_b = 0.0279;
    p->Gncx_b = 0.0034;
    p->Gto_b = 0.16;
    p->Jrel_b = 1.5378;
    p->Jup_b = 1.0;
    p->PNaK_b = 15.4509;
    p->celltype = ENDO;
    if (0) ;
    else if (flag_set(p->flags, "ENDO")) p->celltype = ENDO;
    else if (flag_set(p->flags, "EPI")) p->celltype = EPI;
    else if (flag_set(p->flags, "MCELL")) p->celltype = MCELL;
    p->thL = 200.;
  }
  // Compute the regional initialization
  {
  }

}


// Define the parameters for the lookup tables
enum Tables {
  Cai_TAB,
  V_TAB,

  N_TABS
};

// Define the indices into the lookup tables.

    enum Rosenbrock {
    

      N_ROSEN
    };
enum Cai_TableIndex {
  BCai_idx,
  IpCa_idx,
  Jupnp_idx,
  Jupp_idx,
  KsCa_idx,
  allo_i_idx,
  NROWS_Cai
};

enum V_TableIndex {
  AfCaf_idx,
  AfCas_idx,
  AiF_idx,
  AiS_idx,
  IClb_idx,
  K3_i_idx,
  K3_ss_idx,
  K3pp_i_idx,
  K3pp_ss_idx,
  K8_i_idx,
  K8_ss_idx,
  KNai_idx,
  Vffrt_idx,
  Vfrt_idx,
  a3_idx,
  alpha_idx,
  alpha_2_idx,
  alpha_C2ToI_idx,
  alpha_i_idx,
  ass_idx,
  assp_idx,
  b2_idx,
  beta_idx,
  beta_2_idx,
  beta_ItoC2_idx,
  beta_i_idx,
  dss_idx,
  fCass_idx,
  fss_idx,
  hCa_idx,
  hLss_idx,
  hLssp_idx,
  hNa_idx,
  hss_idx,
  hssp_idx,
  iss_idx,
  jCass_idx,
  jss_idx,
  mLss_idx,
  mss_idx,
  ta_idx,
  td_idx,
  tfCaf_idx,
  tfCafp_idx,
  tfCas_idx,
  tff_idx,
  tffp_idx,
  tfs_idx,
  th_idx,
  tiF_idx,
  tiFp_idx,
  tiS_idx,
  tiSp_idx,
  tj_idx,
  tjp_idx,
  tm_idx,
  tmL_idx,
  txs1_idx,
  txs2_idx,
  xKb_idx,
  xs1ss_idx,
  xs2ss_idx,
  NROWS_V
};



void Tomek_editIonType::construct_tables(IonIfBase& imp_base) const
{
  IonIfDerived& imp = static_cast<IonIfDerived&>(imp_base);
  GlobalData_t dt = imp.get_dt() * 1e0;
  cell_geom* region = &imp.cgeom();
  Tomek_edit_Params *p = imp.params();

  imp.tables().resize(N_TABS);

  // Define the constants that depend on the parameters.
  double GK1 = ((p->celltype==1.) ? (p->GK1_b*1.2) : ((p->celltype==2.) ? (p->GK1_b*1.3) : p->GK1_b));
  double GKb = ((p->celltype==1.) ? (GKb_b*0.6) : GKb_b);
  double GKr = ((p->celltype==1.) ? (p->GKr_b*1.3) : ((p->celltype==2.) ? (p->GKr_b*0.8) : p->GKr_b));
  double GKs = ((p->celltype==1.) ? (GKs_b*1.4) : GKs_b);
  double GNaL = ((p->celltype==1.) ? (p->GNaL_b*0.6) : p->GNaL_b);
  double Gncx = ((p->celltype==1.) ? (p->Gncx_b*1.1) : ((p->celltype==2.) ? (p->Gncx_b*1.4) : p->Gncx_b));
  double Gto = ((p->celltype==1.) ? (p->Gto_b*2.) : ((p->celltype==2.) ? (p->Gto_b*2.) : p->Gto_b));
  double PCa = ((p->celltype==1.) ? (PCa_b*1.2) : ((p->celltype==2.) ? (PCa_b*2.) : PCa_b));
  double PNaK = ((p->celltype==1.) ? (p->PNaK_b*0.9) : ((p->celltype==2.) ? (p->PNaK_b*0.7) : p->PNaK_b));
  double cmdnmax = ((p->celltype==1.) ? (cmdnmax_b*1.3) : cmdnmax_b);
  double thLp = (3.*p->thL);
  double upSCale = ((p->celltype==1.) ? 1.3 : 1.);
  double PCaK = (3.574e-4*PCa);
  double PCaNa = (0.00125*PCa);
  double PCap = (1.1*PCa);
  double PCaKp = (3.574e-4*PCap);
  double PCaNap = (0.00125*PCap);
  
  // Create the Cai lookup table
  LUT* Cai_tab = &imp.tables()[Cai_TAB];
  LUT_alloc(Cai_tab, NROWS_Cai, 1e-6, 1e-2, 1e-6, "Tomek_edit Cai", imp.get_target());
  for (int __i=Cai_tab->mn_ind; __i<=Cai_tab->mx_ind; __i++) {
    double Cai = Cai_tab->res*__i;
    LUT_data_t* Cai_row = Cai_tab->tab[__i];
    Cai_row[BCai_idx] = (1./((1.+((cmdnmax*Kmcmdn)/((Kmcmdn+Cai)*(Kmcmdn+Cai))))+((trpnmax*Kmtrpn)/((Kmtrpn+Cai)*(Kmtrpn+Cai)))));
    Cai_row[IpCa_idx] = ((GpCa*Cai)/(KmCap+Cai));
    Cai_row[Jupnp_idx] = (((upSCale*0.005425)*Cai)/(Cai+0.00092));
    Cai_row[Jupp_idx] = ((((upSCale*2.75)*0.005425)*Cai)/((Cai+0.00092)-(0.00017)));
    Cai_row[KsCa_idx] = (1.+(0.6/(1.+(pow((3.8e-5/Cai),1.4)))));
    Cai_row[allo_i_idx] = (1./(1.+((KmCaAct/Cai)*(KmCaAct/Cai))));
  }
  check_LUT(Cai_tab);
  
  
  // Create the V lookup table
  LUT* V_tab = &imp.tables()[V_TAB];
  LUT_alloc(V_tab, NROWS_V, -1000, 1000, 1e-2, "Tomek_edit V", imp.get_target());
  for (int __i=V_tab->mn_ind; __i<=V_tab->mx_ind; __i++) {
    double V = V_tab->res*__i;
    LUT_data_t* V_row = V_tab->tab[__i];
    V_row[AfCaf_idx] = (0.3+(0.6/(1.+(exp(((V-(10.))/10.))))));
    V_row[AiF_idx] = (1./(1.+(exp((((V+EKshift)-(213.6))/151.2)))));
    V_row[IClb_idx] = (GClb*(V-(ECl)));
    V_row[Vffrt_idx] = (((V*F)*F)/(R*T));
    V_row[Vfrt_idx] = ((V*F)/(R*T));
    double ah = ((V>=-40.) ? 0. : (0.057*(exp(((-(V+80.))/6.8)))));
    double aj = ((V>=-40.) ? 0. : ((((-2.5428e4*(exp((0.2444*V))))-((6.948e-6*(exp((-0.04391*V))))))*(V+37.78))/(1.+(exp((0.311*(V+79.23)))))));
    V_row[ass_idx] = (1./(1.+(exp(((-((V+EKshift)-(14.34)))/14.82)))));
    V_row[assp_idx] = (1./(1.+(exp(((-((V+EKshift)-(24.34)))/14.82)))));
    double bh = ((V>=-40.) ? (0.77/(0.13*(1.+(exp(((-(V+10.66))/11.1)))))) : ((2.7*(exp((0.079*V))))+(3.1e5*(exp((0.3485*V))))));
    double bj = ((V>=-40.) ? ((0.6*(exp((0.057*V))))/(1.+(exp((-0.1*(V+32.)))))) : ((0.02424*(exp((-0.01052*V))))/(1.+(exp((-0.1378*(V+40.14)))))));
    double delta_epi = ((p->celltype==1.) ? (1.-((0.95/(1.+(exp((((V+EKshift)+70.)/5.))))))) : 1.);
    V_row[dss_idx] = ((V>=31.4978) ? 1. : (1.0763*(exp((-1.0070*(exp((-0.0829*V))))))));
    double dti_deVelop = (1.354+(1.e-4/((exp((((V+EKshift)-(167.4))/15.89)))+(exp(((-((V+EKshift)-(12.23)))/0.2154))))));
    double dti_recoVer = (1.-((0.5/(1.+(exp((((V+EKshift)+70.0)/20.0)))))));
    V_row[fss_idx] = (1./(1.+(exp(((V+19.58)/3.696)))));
    V_row[hLss_idx] = (1./(1.+(exp(((V+87.61)/7.488)))));
    V_row[hLssp_idx] = (1./(1.+(exp(((V+93.81)/7.488)))));
    V_row[hss_idx] = (1./((1.+(exp(((V+71.55)/7.43))))*(1.+(exp(((V+71.55)/7.43))))));
    V_row[hssp_idx] = (1./((1.+(exp(((V+77.55)/7.43))))*(1.+(exp(((V+77.55)/7.43))))));
    V_row[iss_idx] = (1./(1.+(exp((((V+EKshift)+43.94)/5.711)))));
    V_row[jCass_idx] = (1.0/(1.0+(exp(((V+18.08)/2.7916)))));
    V_row[mLss_idx] = (1./(1.+(exp(((-(V+42.85))/5.264)))));
    V_row[mss_idx] = (1./((1.+(exp(((-(V+56.86))/9.03))))*(1.+(exp(((-(V+56.86))/9.03))))));
    V_row[ta_idx] = (1.0515/((1./(1.2089*(1.+(exp(((-((V+EKshift)-(18.4099)))/29.3814))))))+(3.5/(1.+(exp((((V+EKshift)+100.)/29.3814)))))));
    V_row[td_idx] = ((offset+0.6)+(1./((exp((-0.05*((V+VShift)+6.))))+(exp((0.09*((V+VShift)+14.)))))));
    V_row[tfCaf_idx] = (7.+(1./((0.04*(exp(((-(V-(4.)))/7.))))+(0.04*(exp(((V-(4.))/7.)))))));
    V_row[tfCas_idx] = (100.+(1./((0.00012*(exp(((-V)/3.))))+(0.00012*(exp((V/7.)))))));
    V_row[tff_idx] = (7.+(1./((0.0045*(exp(((-(V+20.))/10.))))+(0.0045*(exp(((V+20.)/10.)))))));
    V_row[tfs_idx] = (1000.+(1./((0.000035*(exp(((-(V+5.))/4.))))+(0.000035*(exp(((V+5.)/6.)))))));
    double tiF_b = (4.562+(1./((0.3933*(exp(((-((V+EKshift)+100.))/100.))))+(0.08004*(exp((((V+EKshift)+50.)/16.59)))))));
    double tiS_b = (23.62+(1./((0.001416*(exp(((-((V+EKshift)+96.52))/59.05))))+(1.78e-8*(exp((((V+EKshift)+114.1)/8.079)))))));
    V_row[tm_idx] = ((0.1292*(exp((-(((V+45.79)/15.54)*((V+45.79)/15.54))))))+(0.06487*(exp((-(((V-(4.823))/51.12)*((V-(4.823))/51.12)))))));
    V_row[tmL_idx] = ((0.1292*(exp((-(((V+45.79)/15.54)*((V+45.79)/15.54))))))+(0.06487*(exp((-(((V-(4.823))/51.12)*((V-(4.823))/51.12)))))));
    V_row[txs1_idx] = (817.3+(1./((2.326e-4*(exp(((V+48.28)/17.8))))+(0.001292*(exp(((-(V+210.))/230.)))))));
    V_row[txs2_idx] = (1./((0.01*(exp(((V-(50.))/20.))))+(0.0193*(exp(((-(V+66.54))/31.))))));
    V_row[xKb_idx] = (1./(1.+(exp(((-(V-(10.8968)))/23.9871)))));
    V_row[xs1ss_idx] = (1./(1.+(exp(((-(V+11.6))/8.932)))));
    V_row[AfCas_idx] = (1.-(V_row[AfCaf_idx]));
    V_row[AiS_idx] = (1.-(V_row[AiF_idx]));
    V_row[KNai_idx] = (KNai0*(exp(((delta*V_row[Vfrt_idx])/3.))));
    double KNao = (KNao0*(exp((((1.-(delta))*V_row[Vfrt_idx])/3.))));
    V_row[alpha_idx] = (0.1161*(exp((0.2990*V_row[Vfrt_idx]))));
    V_row[alpha_2_idx] = (0.0578*(exp((0.9710*V_row[Vfrt_idx]))));
    V_row[alpha_C2ToI_idx] = (0.52e-4*(exp((1.525*V_row[Vfrt_idx]))));
    V_row[alpha_i_idx] = (0.2533*(exp((0.5953*V_row[Vfrt_idx]))));
    V_row[beta_idx] = (0.2442*(exp((-1.604*V_row[Vfrt_idx]))));
    V_row[beta_2_idx] = (0.349e-3*(exp((-1.062*V_row[Vfrt_idx]))));
    V_row[beta_i_idx] = (0.06525*(exp((-0.8209*V_row[Vfrt_idx]))));
    V_row[fCass_idx] = V_row[fss_idx];
    V_row[hCa_idx] = (exp((qCa*V_row[Vfrt_idx])));
    V_row[hNa_idx] = (exp((qNa*V_row[Vfrt_idx])));
    V_row[jss_idx] = V_row[hss_idx];
    V_row[tfCafp_idx] = (2.5*V_row[tfCaf_idx]);
    V_row[tffp_idx] = (2.5*V_row[tff_idx]);
    V_row[th_idx] = (1./(ah+bh));
    V_row[tiF_idx] = (tiF_b*delta_epi);
    V_row[tiS_idx] = (tiS_b*delta_epi);
    V_row[tj_idx] = (1./(aj+bj));
    V_row[xs2ss_idx] = V_row[xs1ss_idx];
    V_row[a3_idx] = ((K3p*((Ko/KKo)*(Ko/KKo)))/(((((1.+(Nao/KNao))*(1.+(Nao/KNao)))*(1.+(Nao/KNao)))+((1.+(Ko/KKo))*(1.+(Ko/KKo))))-(1.)));
    V_row[b2_idx] = ((K2m*(((Nao/KNao)*(Nao/KNao))*(Nao/KNao)))/(((((1.+(Nao/KNao))*(1.+(Nao/KNao)))*(1.+(Nao/KNao)))+((1.+(Ko/KKo))*(1.+(Ko/KKo))))-(1.)));
    V_row[beta_ItoC2_idx] = (((V_row[beta_2_idx]*V_row[beta_i_idx])*V_row[alpha_C2ToI_idx])/(V_row[alpha_2_idx]*V_row[alpha_i_idx]));
    double h7_i = (1.+((Nao/KNa3)*(1.+(1./V_row[hNa_idx]))));
    double h7_ss = (1.+((Nao/KNa3)*(1.+(1./V_row[hNa_idx]))));
    V_row[tiFp_idx] = ((dti_deVelop*dti_recoVer)*V_row[tiF_idx]);
    V_row[tiSp_idx] = ((dti_deVelop*dti_recoVer)*V_row[tiS_idx]);
    V_row[tjp_idx] = (1.46*V_row[tj_idx]);
    double h8_i = (Nao/((KNa3*V_row[hNa_idx])*h7_i));
    double h8_ss = (Nao/((KNa3*V_row[hNa_idx])*h7_ss));
    double h9_i = (1./h7_i);
    double h9_ss = (1./h7_ss);
    double K3p_i = (h9_i*wCa);
    double K3p_ss = (h9_ss*wCa);
    V_row[K3pp_i_idx] = (h8_i*wNaCa);
    V_row[K3pp_ss_idx] = (h8_ss*wNaCa);
    V_row[K8_i_idx] = ((h8_i*h11_i)*wNa);
    V_row[K8_ss_idx] = ((h8_ss*h11_ss)*wNa);
    V_row[K3_i_idx] = (K3p_i+V_row[K3pp_i_idx]);
    V_row[K3_ss_idx] = (K3p_ss+V_row[K3pp_ss_idx]);
  }
  check_LUT(V_tab);
  

}



void Tomek_editIonType::initialize_sv(IonIfBase& imp_base, GlobalData_t **impdata ) const
{
  IonIfDerived& imp = static_cast<IonIfDerived&>(imp_base);
  GlobalData_t dt = imp.get_dt() * 1e0;
  cell_geom *region = &imp.cgeom();
  Tomek_edit_Params *p = imp.params();

  Tomek_edit_state *sv_base = (Tomek_edit_state *)imp.sv_tab().data();
  GlobalData_t t = 0;

  IonIfDerived* IF = &imp;
  // Define the constants that depend on the parameters.
  double GK1 = ((p->celltype==1.) ? (p->GK1_b*1.2) : ((p->celltype==2.) ? (p->GK1_b*1.3) : p->GK1_b));
  double GKb = ((p->celltype==1.) ? (GKb_b*0.6) : GKb_b);
  double GKr = ((p->celltype==1.) ? (p->GKr_b*1.3) : ((p->celltype==2.) ? (p->GKr_b*0.8) : p->GKr_b));
  double GKs = ((p->celltype==1.) ? (GKs_b*1.4) : GKs_b);
  double GNaL = ((p->celltype==1.) ? (p->GNaL_b*0.6) : p->GNaL_b);
  double Gncx = ((p->celltype==1.) ? (p->Gncx_b*1.1) : ((p->celltype==2.) ? (p->Gncx_b*1.4) : p->Gncx_b));
  double Gto = ((p->celltype==1.) ? (p->Gto_b*2.) : ((p->celltype==2.) ? (p->Gto_b*2.) : p->Gto_b));
  double PCa = ((p->celltype==1.) ? (PCa_b*1.2) : ((p->celltype==2.) ? (PCa_b*2.) : PCa_b));
  double PNaK = ((p->celltype==1.) ? (p->PNaK_b*0.9) : ((p->celltype==2.) ? (p->PNaK_b*0.7) : p->PNaK_b));
  double cmdnmax = ((p->celltype==1.) ? (cmdnmax_b*1.3) : cmdnmax_b);
  double thLp = (3.*p->thL);
  double upSCale = ((p->celltype==1.) ? 1.3 : 1.);
  double PCaK = (3.574e-4*PCa);
  double PCaNa = (0.00125*PCa);
  double PCap = (1.1*PCa);
  double PCaKp = (3.574e-4*PCap);
  double PCaNap = (0.00125*PCap);
  //Prepare all the public arrays.
  GlobalData_t *Iion_ext = impdata[Iion];
  GlobalData_t *V_ext = impdata[Vm];
  //Prepare all the private functions.

  //set the initial values
  for(int __i=0; __i < imp.get_num_node(); __i+=1 ){
    Tomek_edit_state *sv = sv_base+__i / 1;
    //Initialize the external vars to their current values
    GlobalData_t Iion = Iion_ext[__i];
    GlobalData_t V = V_ext[__i];
    //Change the units of external variables as appropriate.
    sv->Cai *= 1e-3;
    
    
    // Initialize the rest of the nodal variables
    sv->C1 = C1_init;
    sv->C2 = C2_init;
    sv->C3 = C3_init;
    sv->CaMKt = CaMKt_init;
    sv->Cai = Cai_init;
    sv->Cajsr = Cajsr_init;
    sv->Cansr = Cansr_init;
    sv->Cass = Cass_init;
    sv->I = I_init;
    sv->Jrel_np = Jrel_np_init;
    sv->Jrel_p = Jrel_p_init;
    sv->Ki = Ki_init;
    sv->Kss = Kss_init;
    sv->Nai = Nai_init;
    sv->Nass = Nass_init;
    sv->O = O_init;
    V = V_init;
    sv->a = a_init;
    sv->ap = ap_init;
    sv->d = d_init;
    sv->fCaf = fCaf_init;
    sv->fCafp = fCafp_init;
    sv->fCas = fCas_init;
    sv->ff = ff_init;
    sv->ffp = ffp_init;
    sv->fs = fs_init;
    sv->h = h_init;
    sv->hL = hL_init;
    sv->hLp = hLp_init;
    sv->hp = hp_init;
    sv->iF = iF_init;
    sv->iFp = iFp_init;
    sv->iS = iS_init;
    sv->iSp = iSp_init;
    sv->j = j_init;
    sv->jCa = jCa_init;
    sv->jp = jp_init;
    sv->m = m_init;
    sv->mL = mL_init;
    sv->nCa_i = nCa_i_init;
    sv->nCa_ss = nCa_ss_init;
    sv->xs1 = xs1_init;
    sv->xs2 = xs2_init;
    double AfCaf = (0.3+(0.6/(1.+(exp(((V-(10.))/10.))))));
    double AiF = (1./(1.+(exp((((V+EKshift)-(213.6))/151.2)))));
    double CaMKb = ((p->CaMKo*(1.-(sv->CaMKt)))/(1.+(KmCaM/sv->Cass)));
    double EK = (((R*T)/(zK*F))*(log((Ko/sv->Ki))));
    double EKs = (((R*T)/(zK*F))*(log(((Ko+(PKNa*Nao))/(sv->Ki+(PKNa*sv->Nai))))));
    double ENa = (((R*T)/(zNa*F))*(log((Nao/sv->Nai))));
    double IClCa_junc = (((Fjunc*GClCa)/(1.+(KdClCa/sv->Cass)))*(V-(ECl)));
    double IClCa_sl = ((((1.-(Fjunc))*GClCa)/(1.+(KdClCa/sv->Cai)))*(V-(ECl)));
    double IClb = (GClb*(V-(ECl)));
    double Ii = ((0.5*(((sv->Nai+sv->Ki)+Cli)+(4.*sv->Cai)))/1000.);
    double IpCa = ((GpCa*sv->Cai)/(KmCap+sv->Cai));
    double Iss = ((0.5*(((sv->Nass+sv->Kss)+Cli)+(4.*sv->Cass)))/1000.);
    double KsCa = (1.+(0.6/(1.+(pow((3.8e-5/sv->Cai),1.4)))));
    double P = (eP/(((1.+(H/Khp))+(sv->Nai/KNap))+(sv->Ki/KxKur)));
    double Vffrt = (((V*F)*F)/(R*T));
    double Vfrt = ((V*F)/(R*T));
    double allo_i = (1./(1.+((KmCaAct/sv->Cai)*(KmCaAct/sv->Cai))));
    double allo_ss = (1./(1.+((KmCaAct/sv->Cass)*(KmCaAct/sv->Cass))));
    double f = ((Aff*sv->ff)+(Afs*sv->fs));
    double fp = ((Aff*sv->ffp)+(Afs*sv->fs));
    double h4_i = (1.+((sv->Nai/KNa1)*(1.+(sv->Nai/KNa2))));
    double h4_ss = (1.+((sv->Nass/KNa1)*(1.+(sv->Nass/KNa2))));
    double xKb = (1./(1.+(exp(((-(V-(10.8968)))/23.9871)))));
    double AfCas = (1.-(AfCaf));
    double AiS = (1.-(AiF));
    double CaMKa = (CaMKb+sv->CaMKt);
    double IClCa = (IClCa_junc+IClCa_sl);
    double IKATP = ((((fKatp*gKatp)*aKiK)*bKiK)*(V-(EK)));
    double IKb = ((GKb*xKb)*(V-(EK)));
    double IKr = (((GKr*(sqrt((Ko/5.))))*sv->O)*(V-(EK)));
    double IKs = ((((GKs*KsCa)*sv->xs1)*sv->xs2)*(V-(EKs)));
    double INab = (((PNab*Vffrt)*((sv->Nai*(exp(Vfrt)))-(Nao)))/((exp(Vfrt))-(1.)));
    double KNai = (KNai0*(exp(((delta*Vfrt)/3.))));
    double KNao = (KNao0*(exp((((1.-(delta))*Vfrt)/3.))));
    double aK1 = (4.094/(1.+(exp((0.1217*((V-(EK))-(49.934)))))));
    double b3 = (((K3m*P)*H)/(1.+(MgATP/Kmgatp)));
    double bK1 = (((15.72*(exp((0.0674*((V-(EK))-(3.257))))))+(exp((0.0618*((V-(EK))-(594.31))))))/(1.+(exp((-0.1629*((V-(EK))+14.207))))));
    double gamma_Cai = (exp((((-constA)*4.)*(((sqrt(Ii))/(1.+(sqrt(Ii))))-((0.3*Ii))))));
    double gamma_Cass = (exp((((-constA)*4.)*(((sqrt(Iss))/(1.+(sqrt(Iss))))-((0.3*Iss))))));
    double gamma_Ki = (exp(((-constA)*(((sqrt(Ii))/(1.+(sqrt(Ii))))-((0.3*Ii))))));
    double gamma_Kss = (exp(((-constA)*(((sqrt(Iss))/(1.+(sqrt(Iss))))-((0.3*Iss))))));
    double gamma_Nai = (exp(((-constA)*(((sqrt(Ii))/(1.+(sqrt(Ii))))-((0.3*Ii))))));
    double gamma_Nass = (exp(((-constA)*(((sqrt(Iss))/(1.+(sqrt(Iss))))-((0.3*Iss))))));
    double h5_i = ((sv->Nai*sv->Nai)/((h4_i*KNa1)*KNa2));
    double h5_ss = ((sv->Nass*sv->Nass)/((h4_ss*KNa1)*KNa2));
    double h6_i = (1./h4_i);
    double h6_ss = (1./h4_ss);
    double hCa = (exp((qCa*Vfrt)));
    double hNa = (exp((qNa*Vfrt)));
    double ICab = ((((PCab*4.)*Vffrt)*(((gamma_Cai*sv->Cai)*(exp((2.*Vfrt))))-((gamma_Cao*Cao))))/((exp((2.*Vfrt)))-(1.)));
    double K1ss = (aK1/(aK1+bK1));
    double K6_i = ((h6_i*sv->Cai)*KCaon);
    double K6_ss = ((h6_ss*sv->Cass)*KCaon);
    double PhiCaK_i = ((Vffrt*(((gamma_Ki*sv->Ki)*(exp(Vfrt)))-((gamma_Ko*Ko))))/((exp(Vfrt))-(1.)));
    double PhiCaK_ss = ((Vffrt*(((gamma_Kss*sv->Kss)*(exp(Vfrt)))-((gamma_Ko*Ko))))/((exp(Vfrt))-(1.)));
    double PhiCaL_i = (((4.*Vffrt)*(((gamma_Cai*sv->Cai)*(exp((2.*Vfrt))))-((gamma_Cao*Cao))))/((exp((2.*Vfrt)))-(1.)));
    double PhiCaL_ss = (((4.*Vffrt)*(((gamma_Cass*sv->Cass)*(exp((2.*Vfrt))))-((gamma_Cao*Cao))))/((exp((2.*Vfrt)))-(1.)));
    double PhiCaNa_i = ((Vffrt*(((gamma_Nai*sv->Nai)*(exp(Vfrt)))-((gamma_Nao*Nao))))/((exp(Vfrt))-(1.)));
    double PhiCaNa_ss = ((Vffrt*(((gamma_Nass*sv->Nass)*(exp(Vfrt)))-((gamma_Nao*Nao))))/((exp(Vfrt))-(1.)));
    double a1 = ((K1p*(((sv->Nai/KNai)*(sv->Nai/KNai))*(sv->Nai/KNai)))/(((((1.+(sv->Nai/KNai))*(1.+(sv->Nai/KNai)))*(1.+(sv->Nai/KNai)))+((1.+(sv->Ki/KKi))*(1.+(sv->Ki/KKi))))-(1.)));
    double a3 = ((K3p*((Ko/KKo)*(Ko/KKo)))/(((((1.+(Nao/KNao))*(1.+(Nao/KNao)))*(1.+(Nao/KNao)))+((1.+(Ko/KKo))*(1.+(Ko/KKo))))-(1.)));
    double b2 = ((K2m*(((Nao/KNao)*(Nao/KNao))*(Nao/KNao)))/(((((1.+(Nao/KNao))*(1.+(Nao/KNao)))*(1.+(Nao/KNao)))+((1.+(Ko/KKo))*(1.+(Ko/KKo))))-(1.)));
    double b4 = ((K4m*((sv->Ki/KKi)*(sv->Ki/KKi)))/(((((1.+(sv->Nai/KNai))*(1.+(sv->Nai/KNai)))*(1.+(sv->Nai/KNai)))+((1.+(sv->Ki/KKi))*(1.+(sv->Ki/KKi))))-(1.)));
    double fCa = ((AfCaf*sv->fCaf)+(AfCas*sv->fCas));
    double fCap = ((AfCaf*sv->fCafp)+(AfCas*sv->fCas));
    double fICaLp = (1./(1.+(KmCaMK/CaMKa)));
    double fINaLp = (1./(1.+(KmCaMK/CaMKa)));
    double fINap = (1./(1.+(KmCaMK/CaMKa)));
    double fItop = (1./(1.+(KmCaMK/CaMKa)));
    double h1_i = (1.+((sv->Nai/KNa3)*(1.+hNa)));
    double h1_ss = (1.+((sv->Nass/KNa3)*(1.+hNa)));
    double h7_i = (1.+((Nao/KNa3)*(1.+(1./hNa))));
    double h7_ss = (1.+((Nao/KNa3)*(1.+(1./hNa))));
    double i_t = ((AiF*sv->iF)+(AiS*sv->iS));
    double ip = ((AiF*sv->iFp)+(AiS*sv->iSp));
    double ICaK_i = ((1.-(ICaL_fractionSS))*((((((1.-(fICaLp))*PCaK)*PhiCaK_i)*sv->d)*((f*(1.-(sv->nCa_i)))+((sv->jCa*fCa)*sv->nCa_i)))+((((fICaLp*PCaKp)*PhiCaK_i)*sv->d)*((fp*(1.-(sv->nCa_i)))+((sv->jCa*fCap)*sv->nCa_i)))));
    double ICaK_ss = (ICaL_fractionSS*((((((1.-(fICaLp))*PCaK)*PhiCaK_ss)*sv->d)*((f*(1.-(sv->nCa_ss)))+((sv->jCa*fCa)*sv->nCa_ss)))+((((fICaLp*PCaKp)*PhiCaK_ss)*sv->d)*((fp*(1.-(sv->nCa_ss)))+((sv->jCa*fCap)*sv->nCa_ss)))));
    double ICaL_i = ((1.-(ICaL_fractionSS))*((((((1.-(fICaLp))*PCa)*PhiCaL_i)*sv->d)*((f*(1.-(sv->nCa_i)))+((sv->jCa*fCa)*sv->nCa_i)))+((((fICaLp*PCap)*PhiCaL_i)*sv->d)*((fp*(1.-(sv->nCa_i)))+((sv->jCa*fCap)*sv->nCa_i)))));
    double ICaL_ss = (ICaL_fractionSS*((((((1.-(fICaLp))*PCa)*PhiCaL_ss)*sv->d)*((f*(1.-(sv->nCa_ss)))+((sv->jCa*fCa)*sv->nCa_ss)))+((((fICaLp*PCap)*PhiCaL_ss)*sv->d)*((fp*(1.-(sv->nCa_ss)))+((sv->jCa*fCap)*sv->nCa_ss)))));
    double ICaNa_i = ((1.-(ICaL_fractionSS))*((((((1.-(fICaLp))*PCaNa)*PhiCaNa_i)*sv->d)*((f*(1.-(sv->nCa_i)))+((sv->jCa*fCa)*sv->nCa_i)))+((((fICaLp*PCaNap)*PhiCaNa_i)*sv->d)*((fp*(1.-(sv->nCa_i)))+((sv->jCa*fCap)*sv->nCa_i)))));
    double ICaNa_ss = (ICaL_fractionSS*((((((1.-(fICaLp))*PCaNa)*PhiCaNa_ss)*sv->d)*((f*(1.-(sv->nCa_ss)))+((sv->jCa*fCa)*sv->nCa_ss)))+((((fICaLp*PCaNap)*PhiCaNa_ss)*sv->d)*((fp*(1.-(sv->nCa_ss)))+((sv->jCa*fCap)*sv->nCa_ss)))));
    double IK1 = (((GK1*(sqrt((Ko/5.))))*K1ss)*(V-(EK)));
    double INa = (((GNa*(V-(ENa)))*((sv->m*sv->m)*sv->m))*((((1.-(fINap))*sv->h)*sv->j)+((fINap*sv->hp)*sv->jp)));
    double INaL = (((GNaL*(V-(ENa)))*sv->mL)*(((1.-(fINaLp))*sv->hL)+(fINaLp*sv->hLp)));
    double Ito = ((Gto*(V-(EK)))*((((1.-(fItop))*sv->a)*i_t)+((fItop*sv->ap)*ip)));
    double h2_i = ((sv->Nai*hNa)/(KNa3*h1_i));
    double h2_ss = ((sv->Nass*hNa)/(KNa3*h1_ss));
    double h3_i = (1./h1_i);
    double h3_ss = (1./h1_ss);
    double h8_i = (Nao/((KNa3*hNa)*h7_i));
    double h8_ss = (Nao/((KNa3*hNa)*h7_ss));
    double h9_i = (1./h7_i);
    double h9_ss = (1./h7_ss);
    double x1 = (((((a4*a1)*a2)+((b2*b4)*b3))+((a2*b4)*b3))+((b3*a1)*a2));
    double x2 = (((((b2*b1)*b4)+((a1*a2)*a3))+((a3*b1)*b4))+((a2*a3)*b4));
    double x3 = (((((a2*a3)*a4)+((b3*b2)*b1))+((b2*b1)*a4))+((a3*a4)*b1));
    double x4 = (((((b4*b3)*b2)+((a3*a4)*a1))+((b2*a4)*a1))+((b3*b2)*a1));
    double E1 = (x1/(((x1+x2)+x3)+x4));
    double E2 = (x2/(((x1+x2)+x3)+x4));
    double E3 = (x3/(((x1+x2)+x3)+x4));
    double E4 = (x4/(((x1+x2)+x3)+x4));
    double ICaK = (ICaK_ss+ICaK_i);
    double ICaL = (ICaL_ss+ICaL_i);
    double ICaNa = (ICaNa_ss+ICaNa_i);
    double K3p_i = (h9_i*wCa);
    double K3p_ss = (h9_ss*wCa);
    double K3pp_i = (h8_i*wNaCa);
    double K3pp_ss = (h8_ss*wNaCa);
    double K4p_i = ((h3_i*wCa)/hCa);
    double K4p_ss = ((h3_ss*wCa)/hCa);
    double K4pp_i = (h2_i*wNaCa);
    double K4pp_ss = (h2_ss*wNaCa);
    double K7_i = ((h5_i*h2_i)*wNa);
    double K7_ss = ((h5_ss*h2_ss)*wNa);
    double K8_i = ((h8_i*h11_i)*wNa);
    double K8_ss = ((h8_ss*h11_ss)*wNa);
    double JNaKK = (2.*((E4*b1)-((E3*a1))));
    double JNaKNa = (3.*((E1*a3)-((E2*b3))));
    double K3_i = (K3p_i+K3pp_i);
    double K3_ss = (K3p_ss+K3pp_ss);
    double K4_i = (K4p_i+K4pp_i);
    double K4_ss = (K4p_ss+K4pp_ss);
    double INaK = (PNaK*((zNa*JNaKNa)+(zK*JNaKK)));
    double x1_i = (((K2_i*K4_i)*(K7_i+K6_i))+((K5_i*K7_i)*(K2_i+K3_i)));
    double x1_ss = (((K2_ss*K4_ss)*(K7_ss+K6_ss))+((K5_ss*K7_ss)*(K2_ss+K3_ss)));
    double x2_i = (((K1_i*K7_i)*(K4_i+K5_i))+((K4_i*K6_i)*(K1_i+K8_i)));
    double x2_ss = (((K1_ss*K7_ss)*(K4_ss+K5_ss))+((K4_ss*K6_ss)*(K1_ss+K8_ss)));
    double x3_i = (((K1_i*K3_i)*(K7_i+K6_i))+((K8_i*K6_i)*(K2_i+K3_i)));
    double x3_ss = (((K1_ss*K3_ss)*(K7_ss+K6_ss))+((K8_ss*K6_ss)*(K2_ss+K3_ss)));
    double x4_i = (((K2_i*K8_i)*(K4_i+K5_i))+((K3_i*K5_i)*(K1_i+K8_i)));
    double x4_ss = (((K2_ss*K8_ss)*(K4_ss+K5_ss))+((K3_ss*K5_ss)*(K1_ss+K8_ss)));
    double E1_i = (x1_i/(((x1_i+x2_i)+x3_i)+x4_i));
    double E1_ss = (x1_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
    double E2_i = (x2_i/(((x1_i+x2_i)+x3_i)+x4_i));
    double E2_ss = (x2_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
    double E3_i = (x3_i/(((x1_i+x2_i)+x3_i)+x4_i));
    double E3_ss = (x3_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
    double E4_i = (x4_i/(((x1_i+x2_i)+x3_i)+x4_i));
    double E4_ss = (x4_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
    double JncxCa_i = ((E2_i*K2_i)-((E1_i*K1_i)));
    double JncxCa_ss = ((E2_ss*K2_ss)-((E1_ss*K1_ss)));
    double JncxNa_i = (((3.*((E4_i*K7_i)-((E1_i*K8_i))))+(E3_i*K4pp_i))-((E2_i*K3pp_i)));
    double JncxNa_ss = (((3.*((E4_ss*K7_ss)-((E1_ss*K8_ss))))+(E3_ss*K4pp_ss))-((E2_ss*K3pp_ss)));
    double INaCa_i = ((((1.-(INaCa_fractionSS))*Gncx)*allo_i)*((zNa*JncxNa_i)+(zCa*JncxCa_i)));
    double INaCa_ss = (((INaCa_fractionSS*Gncx)*allo_ss)*((zNa*JncxNa_ss)+(zCa*JncxCa_ss)));
    Iion = ((((((((((((((((((INa+INaL)+Ito)+ICaL)+ICaNa)+ICaK)+IKr)+IKs)+IK1)+INaCa_i)+INaCa_ss)+INaK)+INab)+IKb)+IpCa)+ICab)+IClCa)+IClb)+IKATP);
    //Change the units of external variables as appropriate.
    sv->Cai *= 1e3;
    
    
    //Save all external vars
    Iion_ext[__i] = Iion;
    V_ext[__i] = V;
  }

}

/** compute the  current
 *
 * param start   index of first node
 * param end     index of last node
 * param IF      IMP
 * param plgdata external data needed by IMP
 */
#ifdef TOMEK_EDIT_CPU_GENERATED
extern "C" {
void compute_Tomek_edit_cpu(int start, int end, IonIfBase& imp_base, GlobalData_t **impdata )
{
  Tomek_editIonType::IonIfDerived& imp = static_cast<Tomek_editIonType::IonIfDerived&>(imp_base);
  GlobalData_t dt = imp.get_dt()*1e0;
  cell_geom *region = &imp.cgeom();
  Tomek_edit_Params *p  = imp.params();
  Tomek_edit_state *sv_base = (Tomek_edit_state *)imp.sv_tab().data();

  GlobalData_t t = imp.get_tstp().cnt*dt;

  Tomek_editIonType::IonIfDerived* IF = &imp;

  // Define the constants that depend on the parameters.
  GlobalData_t GK1 = ((p->celltype==1.) ? (p->GK1_b*1.2) : ((p->celltype==2.) ? (p->GK1_b*1.3) : p->GK1_b));
  GlobalData_t GKb = ((p->celltype==1.) ? (GKb_b*0.6) : GKb_b);
  GlobalData_t GKr = ((p->celltype==1.) ? (p->GKr_b*1.3) : ((p->celltype==2.) ? (p->GKr_b*0.8) : p->GKr_b));
  GlobalData_t GKs = ((p->celltype==1.) ? (GKs_b*1.4) : GKs_b);
  GlobalData_t GNaL = ((p->celltype==1.) ? (p->GNaL_b*0.6) : p->GNaL_b);
  GlobalData_t Gncx = ((p->celltype==1.) ? (p->Gncx_b*1.1) : ((p->celltype==2.) ? (p->Gncx_b*1.4) : p->Gncx_b));
  GlobalData_t Gto = ((p->celltype==1.) ? (p->Gto_b*2.) : ((p->celltype==2.) ? (p->Gto_b*2.) : p->Gto_b));
  GlobalData_t PCa = ((p->celltype==1.) ? (PCa_b*1.2) : ((p->celltype==2.) ? (PCa_b*2.) : PCa_b));
  GlobalData_t PNaK = ((p->celltype==1.) ? (p->PNaK_b*0.9) : ((p->celltype==2.) ? (p->PNaK_b*0.7) : p->PNaK_b));
  GlobalData_t cmdnmax = ((p->celltype==1.) ? (cmdnmax_b*1.3) : cmdnmax_b);
  GlobalData_t thLp = (3.*p->thL);
  GlobalData_t upSCale = ((p->celltype==1.) ? 1.3 : 1.);
  GlobalData_t PCaK = (3.574e-4*PCa);
  GlobalData_t PCaNa = (0.00125*PCa);
  GlobalData_t PCap = (1.1*PCa);
  GlobalData_t PCaKp = (3.574e-4*PCap);
  GlobalData_t PCaNap = (0.00125*PCap);
  //Prepare all the public arrays.
  GlobalData_t *Iion_ext = impdata[Iion];
  GlobalData_t *V_ext = impdata[Vm];
  //Prepare all the private functions.

#pragma omp parallel for schedule(static)
  for (int __i=(start / 1) * 1; __i<end; __i+=1) {
    Tomek_edit_state *sv = sv_base+__i / 1;
                    
    //Initialize the external vars to their current values
    GlobalData_t Iion = Iion_ext[__i];
    GlobalData_t V = V_ext[__i];
    //Change the units of external variables as appropriate.
    sv->Cai *= 1e-3;
    
    
    //Compute lookup tables for things that have already been defined.
    LUT_data_t Cai_row[NROWS_Cai];
    LUT_interpRow(&IF->tables()[Cai_TAB], sv->Cai, __i, Cai_row);
    LUT_data_t V_row[NROWS_V];
    LUT_interpRow(&IF->tables()[V_TAB], V, __i, V_row);
    
    
    //Compute storevars and external modvars
    GlobalData_t CaMKb = ((p->CaMKo*(1.-(sv->CaMKt)))/(1.+(KmCaM/sv->Cass)));
    GlobalData_t EK = (((R*T)/(zK*F))*(log((Ko/sv->Ki))));
    GlobalData_t EKs = (((R*T)/(zK*F))*(log(((Ko+(PKNa*Nao))/(sv->Ki+(PKNa*sv->Nai))))));
    GlobalData_t ENa = (((R*T)/(zNa*F))*(log((Nao/sv->Nai))));
    GlobalData_t IClCa_junc = (((Fjunc*GClCa)/(1.+(KdClCa/sv->Cass)))*(V-(ECl)));
    GlobalData_t IClCa_sl = ((((1.-(Fjunc))*GClCa)/(1.+(KdClCa/sv->Cai)))*(V-(ECl)));
    GlobalData_t Ii = ((0.5*(((sv->Nai+sv->Ki)+Cli)+(4.*sv->Cai)))/1000.);
    GlobalData_t Iss = ((0.5*(((sv->Nass+sv->Kss)+Cli)+(4.*sv->Cass)))/1000.);
    GlobalData_t P = (eP/(((1.+(H/Khp))+(sv->Nai/KNap))+(sv->Ki/KxKur)));
    GlobalData_t allo_ss = (1./(1.+((KmCaAct/sv->Cass)*(KmCaAct/sv->Cass))));
    GlobalData_t f = ((Aff*sv->ff)+(Afs*sv->fs));
    GlobalData_t fp = ((Aff*sv->ffp)+(Afs*sv->fs));
    GlobalData_t h4_i = (1.+((sv->Nai/KNa1)*(1.+(sv->Nai/KNa2))));
    GlobalData_t h4_ss = (1.+((sv->Nass/KNa1)*(1.+(sv->Nass/KNa2))));
    GlobalData_t CaMKa = (CaMKb+sv->CaMKt);
    GlobalData_t IClCa = (IClCa_junc+IClCa_sl);
    GlobalData_t IKATP = ((((fKatp*gKatp)*aKiK)*bKiK)*(V-(EK)));
    GlobalData_t IKb = ((GKb*V_row[xKb_idx])*(V-(EK)));
    GlobalData_t IKr = (((GKr*(sqrt((Ko/5.))))*sv->O)*(V-(EK)));
    GlobalData_t IKs = ((((GKs*Cai_row[KsCa_idx])*sv->xs1)*sv->xs2)*(V-(EKs)));
    GlobalData_t INab = (((PNab*V_row[Vffrt_idx])*((sv->Nai*(exp(V_row[Vfrt_idx])))-(Nao)))/((exp(V_row[Vfrt_idx]))-(1.)));
    GlobalData_t aK1 = (4.094/(1.+(exp((0.1217*((V-(EK))-(49.934)))))));
    GlobalData_t b3 = (((K3m*P)*H)/(1.+(MgATP/Kmgatp)));
    GlobalData_t bK1 = (((15.72*(exp((0.0674*((V-(EK))-(3.257))))))+(exp((0.0618*((V-(EK))-(594.31))))))/(1.+(exp((-0.1629*((V-(EK))+14.207))))));
    GlobalData_t gamma_Cai = (exp((((-constA)*4.)*(((sqrt(Ii))/(1.+(sqrt(Ii))))-((0.3*Ii))))));
    GlobalData_t gamma_Cass = (exp((((-constA)*4.)*(((sqrt(Iss))/(1.+(sqrt(Iss))))-((0.3*Iss))))));
    GlobalData_t gamma_Ki = (exp(((-constA)*(((sqrt(Ii))/(1.+(sqrt(Ii))))-((0.3*Ii))))));
    GlobalData_t gamma_Kss = (exp(((-constA)*(((sqrt(Iss))/(1.+(sqrt(Iss))))-((0.3*Iss))))));
    GlobalData_t gamma_Nai = (exp(((-constA)*(((sqrt(Ii))/(1.+(sqrt(Ii))))-((0.3*Ii))))));
    GlobalData_t gamma_Nass = (exp(((-constA)*(((sqrt(Iss))/(1.+(sqrt(Iss))))-((0.3*Iss))))));
    GlobalData_t h5_i = ((sv->Nai*sv->Nai)/((h4_i*KNa1)*KNa2));
    GlobalData_t h5_ss = ((sv->Nass*sv->Nass)/((h4_ss*KNa1)*KNa2));
    GlobalData_t h6_i = (1./h4_i);
    GlobalData_t h6_ss = (1./h4_ss);
    GlobalData_t ICab = ((((PCab*4.)*V_row[Vffrt_idx])*(((gamma_Cai*sv->Cai)*(exp((2.*V_row[Vfrt_idx]))))-((gamma_Cao*Cao))))/((exp((2.*V_row[Vfrt_idx])))-(1.)));
    GlobalData_t K1ss = (aK1/(aK1+bK1));
    GlobalData_t K6_i = ((h6_i*sv->Cai)*KCaon);
    GlobalData_t K6_ss = ((h6_ss*sv->Cass)*KCaon);
    GlobalData_t PhiCaK_i = ((V_row[Vffrt_idx]*(((gamma_Ki*sv->Ki)*(exp(V_row[Vfrt_idx])))-((gamma_Ko*Ko))))/((exp(V_row[Vfrt_idx]))-(1.)));
    GlobalData_t PhiCaK_ss = ((V_row[Vffrt_idx]*(((gamma_Kss*sv->Kss)*(exp(V_row[Vfrt_idx])))-((gamma_Ko*Ko))))/((exp(V_row[Vfrt_idx]))-(1.)));
    GlobalData_t PhiCaL_i = (((4.*V_row[Vffrt_idx])*(((gamma_Cai*sv->Cai)*(exp((2.*V_row[Vfrt_idx]))))-((gamma_Cao*Cao))))/((exp((2.*V_row[Vfrt_idx])))-(1.)));
    GlobalData_t PhiCaL_ss = (((4.*V_row[Vffrt_idx])*(((gamma_Cass*sv->Cass)*(exp((2.*V_row[Vfrt_idx]))))-((gamma_Cao*Cao))))/((exp((2.*V_row[Vfrt_idx])))-(1.)));
    GlobalData_t PhiCaNa_i = ((V_row[Vffrt_idx]*(((gamma_Nai*sv->Nai)*(exp(V_row[Vfrt_idx])))-((gamma_Nao*Nao))))/((exp(V_row[Vfrt_idx]))-(1.)));
    GlobalData_t PhiCaNa_ss = ((V_row[Vffrt_idx]*(((gamma_Nass*sv->Nass)*(exp(V_row[Vfrt_idx])))-((gamma_Nao*Nao))))/((exp(V_row[Vfrt_idx]))-(1.)));
    GlobalData_t a1 = ((K1p*(((sv->Nai/V_row[KNai_idx])*(sv->Nai/V_row[KNai_idx]))*(sv->Nai/V_row[KNai_idx])))/(((((1.+(sv->Nai/V_row[KNai_idx]))*(1.+(sv->Nai/V_row[KNai_idx])))*(1.+(sv->Nai/V_row[KNai_idx])))+((1.+(sv->Ki/KKi))*(1.+(sv->Ki/KKi))))-(1.)));
    GlobalData_t b4 = ((K4m*((sv->Ki/KKi)*(sv->Ki/KKi)))/(((((1.+(sv->Nai/V_row[KNai_idx]))*(1.+(sv->Nai/V_row[KNai_idx])))*(1.+(sv->Nai/V_row[KNai_idx])))+((1.+(sv->Ki/KKi))*(1.+(sv->Ki/KKi))))-(1.)));
    GlobalData_t fCa = ((V_row[AfCaf_idx]*sv->fCaf)+(V_row[AfCas_idx]*sv->fCas));
    GlobalData_t fCap = ((V_row[AfCaf_idx]*sv->fCafp)+(V_row[AfCas_idx]*sv->fCas));
    GlobalData_t fICaLp = (1./(1.+(KmCaMK/CaMKa)));
    GlobalData_t fINaLp = (1./(1.+(KmCaMK/CaMKa)));
    GlobalData_t fINap = (1./(1.+(KmCaMK/CaMKa)));
    GlobalData_t fItop = (1./(1.+(KmCaMK/CaMKa)));
    GlobalData_t h1_i = (1.+((sv->Nai/KNa3)*(1.+V_row[hNa_idx])));
    GlobalData_t h1_ss = (1.+((sv->Nass/KNa3)*(1.+V_row[hNa_idx])));
    GlobalData_t i_t = ((V_row[AiF_idx]*sv->iF)+(V_row[AiS_idx]*sv->iS));
    GlobalData_t ip = ((V_row[AiF_idx]*sv->iFp)+(V_row[AiS_idx]*sv->iSp));
    GlobalData_t ICaK_i = ((1.-(ICaL_fractionSS))*((((((1.-(fICaLp))*PCaK)*PhiCaK_i)*sv->d)*((f*(1.-(sv->nCa_i)))+((sv->jCa*fCa)*sv->nCa_i)))+((((fICaLp*PCaKp)*PhiCaK_i)*sv->d)*((fp*(1.-(sv->nCa_i)))+((sv->jCa*fCap)*sv->nCa_i)))));
    GlobalData_t ICaK_ss = (ICaL_fractionSS*((((((1.-(fICaLp))*PCaK)*PhiCaK_ss)*sv->d)*((f*(1.-(sv->nCa_ss)))+((sv->jCa*fCa)*sv->nCa_ss)))+((((fICaLp*PCaKp)*PhiCaK_ss)*sv->d)*((fp*(1.-(sv->nCa_ss)))+((sv->jCa*fCap)*sv->nCa_ss)))));
    GlobalData_t ICaL_i = ((1.-(ICaL_fractionSS))*((((((1.-(fICaLp))*PCa)*PhiCaL_i)*sv->d)*((f*(1.-(sv->nCa_i)))+((sv->jCa*fCa)*sv->nCa_i)))+((((fICaLp*PCap)*PhiCaL_i)*sv->d)*((fp*(1.-(sv->nCa_i)))+((sv->jCa*fCap)*sv->nCa_i)))));
    GlobalData_t ICaL_ss = (ICaL_fractionSS*((((((1.-(fICaLp))*PCa)*PhiCaL_ss)*sv->d)*((f*(1.-(sv->nCa_ss)))+((sv->jCa*fCa)*sv->nCa_ss)))+((((fICaLp*PCap)*PhiCaL_ss)*sv->d)*((fp*(1.-(sv->nCa_ss)))+((sv->jCa*fCap)*sv->nCa_ss)))));
    GlobalData_t ICaNa_i = ((1.-(ICaL_fractionSS))*((((((1.-(fICaLp))*PCaNa)*PhiCaNa_i)*sv->d)*((f*(1.-(sv->nCa_i)))+((sv->jCa*fCa)*sv->nCa_i)))+((((fICaLp*PCaNap)*PhiCaNa_i)*sv->d)*((fp*(1.-(sv->nCa_i)))+((sv->jCa*fCap)*sv->nCa_i)))));
    GlobalData_t ICaNa_ss = (ICaL_fractionSS*((((((1.-(fICaLp))*PCaNa)*PhiCaNa_ss)*sv->d)*((f*(1.-(sv->nCa_ss)))+((sv->jCa*fCa)*sv->nCa_ss)))+((((fICaLp*PCaNap)*PhiCaNa_ss)*sv->d)*((fp*(1.-(sv->nCa_ss)))+((sv->jCa*fCap)*sv->nCa_ss)))));
    GlobalData_t IK1 = (((GK1*(sqrt((Ko/5.))))*K1ss)*(V-(EK)));
    GlobalData_t INa = (((GNa*(V-(ENa)))*((sv->m*sv->m)*sv->m))*((((1.-(fINap))*sv->h)*sv->j)+((fINap*sv->hp)*sv->jp)));
    GlobalData_t INaL = (((GNaL*(V-(ENa)))*sv->mL)*(((1.-(fINaLp))*sv->hL)+(fINaLp*sv->hLp)));
    GlobalData_t Ito = ((Gto*(V-(EK)))*((((1.-(fItop))*sv->a)*i_t)+((fItop*sv->ap)*ip)));
    GlobalData_t h2_i = ((sv->Nai*V_row[hNa_idx])/(KNa3*h1_i));
    GlobalData_t h2_ss = ((sv->Nass*V_row[hNa_idx])/(KNa3*h1_ss));
    GlobalData_t h3_i = (1./h1_i);
    GlobalData_t h3_ss = (1./h1_ss);
    GlobalData_t x1 = (((((a4*a1)*a2)+((V_row[b2_idx]*b4)*b3))+((a2*b4)*b3))+((b3*a1)*a2));
    GlobalData_t x2 = (((((V_row[b2_idx]*b1)*b4)+((a1*a2)*V_row[a3_idx]))+((V_row[a3_idx]*b1)*b4))+((a2*V_row[a3_idx])*b4));
    GlobalData_t x3 = (((((a2*V_row[a3_idx])*a4)+((b3*V_row[b2_idx])*b1))+((V_row[b2_idx]*b1)*a4))+((V_row[a3_idx]*a4)*b1));
    GlobalData_t x4 = (((((b4*b3)*V_row[b2_idx])+((V_row[a3_idx]*a4)*a1))+((V_row[b2_idx]*a4)*a1))+((b3*V_row[b2_idx])*a1));
    GlobalData_t E1 = (x1/(((x1+x2)+x3)+x4));
    GlobalData_t E2 = (x2/(((x1+x2)+x3)+x4));
    GlobalData_t E3 = (x3/(((x1+x2)+x3)+x4));
    GlobalData_t E4 = (x4/(((x1+x2)+x3)+x4));
    GlobalData_t ICaK = (ICaK_ss+ICaK_i);
    GlobalData_t ICaL = (ICaL_ss+ICaL_i);
    GlobalData_t ICaNa = (ICaNa_ss+ICaNa_i);
    GlobalData_t K4p_i = ((h3_i*wCa)/V_row[hCa_idx]);
    GlobalData_t K4p_ss = ((h3_ss*wCa)/V_row[hCa_idx]);
    GlobalData_t K4pp_i = (h2_i*wNaCa);
    GlobalData_t K4pp_ss = (h2_ss*wNaCa);
    GlobalData_t K7_i = ((h5_i*h2_i)*wNa);
    GlobalData_t K7_ss = ((h5_ss*h2_ss)*wNa);
    GlobalData_t JNaKK = (2.*((E4*b1)-((E3*a1))));
    GlobalData_t JNaKNa = (3.*((E1*V_row[a3_idx])-((E2*b3))));
    GlobalData_t K4_i = (K4p_i+K4pp_i);
    GlobalData_t K4_ss = (K4p_ss+K4pp_ss);
    GlobalData_t INaK = (PNaK*((zNa*JNaKNa)+(zK*JNaKK)));
    GlobalData_t x1_i = (((K2_i*K4_i)*(K7_i+K6_i))+((K5_i*K7_i)*(K2_i+V_row[K3_i_idx])));
    GlobalData_t x1_ss = (((K2_ss*K4_ss)*(K7_ss+K6_ss))+((K5_ss*K7_ss)*(K2_ss+V_row[K3_ss_idx])));
    GlobalData_t x2_i = (((K1_i*K7_i)*(K4_i+K5_i))+((K4_i*K6_i)*(K1_i+V_row[K8_i_idx])));
    GlobalData_t x2_ss = (((K1_ss*K7_ss)*(K4_ss+K5_ss))+((K4_ss*K6_ss)*(K1_ss+V_row[K8_ss_idx])));
    GlobalData_t x3_i = (((K1_i*V_row[K3_i_idx])*(K7_i+K6_i))+((V_row[K8_i_idx]*K6_i)*(K2_i+V_row[K3_i_idx])));
    GlobalData_t x3_ss = (((K1_ss*V_row[K3_ss_idx])*(K7_ss+K6_ss))+((V_row[K8_ss_idx]*K6_ss)*(K2_ss+V_row[K3_ss_idx])));
    GlobalData_t x4_i = (((K2_i*V_row[K8_i_idx])*(K4_i+K5_i))+((V_row[K3_i_idx]*K5_i)*(K1_i+V_row[K8_i_idx])));
    GlobalData_t x4_ss = (((K2_ss*V_row[K8_ss_idx])*(K4_ss+K5_ss))+((V_row[K3_ss_idx]*K5_ss)*(K1_ss+V_row[K8_ss_idx])));
    GlobalData_t E1_i = (x1_i/(((x1_i+x2_i)+x3_i)+x4_i));
    GlobalData_t E1_ss = (x1_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
    GlobalData_t E2_i = (x2_i/(((x1_i+x2_i)+x3_i)+x4_i));
    GlobalData_t E2_ss = (x2_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
    GlobalData_t E3_i = (x3_i/(((x1_i+x2_i)+x3_i)+x4_i));
    GlobalData_t E3_ss = (x3_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
    GlobalData_t E4_i = (x4_i/(((x1_i+x2_i)+x3_i)+x4_i));
    GlobalData_t E4_ss = (x4_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
    GlobalData_t JncxCa_i = ((E2_i*K2_i)-((E1_i*K1_i)));
    GlobalData_t JncxCa_ss = ((E2_ss*K2_ss)-((E1_ss*K1_ss)));
    GlobalData_t JncxNa_i = (((3.*((E4_i*K7_i)-((E1_i*V_row[K8_i_idx]))))+(E3_i*K4pp_i))-((E2_i*V_row[K3pp_i_idx])));
    GlobalData_t JncxNa_ss = (((3.*((E4_ss*K7_ss)-((E1_ss*V_row[K8_ss_idx]))))+(E3_ss*K4pp_ss))-((E2_ss*V_row[K3pp_ss_idx])));
    GlobalData_t INaCa_i = ((((1.-(INaCa_fractionSS))*Gncx)*Cai_row[allo_i_idx])*((zNa*JncxNa_i)+(zCa*JncxCa_i)));
    GlobalData_t INaCa_ss = (((INaCa_fractionSS*Gncx)*allo_ss)*((zNa*JncxNa_ss)+(zCa*JncxCa_ss)));
    Iion = ((((((((((((((((((INa+INaL)+Ito)+ICaL)+ICaNa)+ICaK)+IKr)+IKs)+IK1)+INaCa_i)+INaCa_ss)+INaK)+INab)+IKb)+Cai_row[IpCa_idx])+ICab)+IClCa)+V_row[IClb_idx])+IKATP);
    
    
    //Complete Forward Euler Update
    GlobalData_t BCajsr = (1./(1.+((csqnmax*Kmcsqn)/((Kmcsqn+sv->Cajsr)*(Kmcsqn+sv->Cajsr)))));
    GlobalData_t BCass = (1./((1.+((BSRmax*KmBSR)/((KmBSR+sv->Cass)*(KmBSR+sv->Cass))))+((BSLmax*KmBSL)/((KmBSL+sv->Cass)*(KmBSL+sv->Cass)))));
    GlobalData_t Jdiff = ((sv->Cass-(sv->Cai))/tauCa);
    GlobalData_t JdiffK = ((sv->Kss-(sv->Ki))/tauK);
    GlobalData_t JdiffNa = ((sv->Nass-(sv->Nai))/tauNa);
    GlobalData_t JleaK = ((0.0048825*sv->Cansr)/15.);
    GlobalData_t Jrel_inf_b = (((-a_rel)*ICaL_ss)/(1.+((((((((p->Cajsr_half/sv->Cajsr)*(p->Cajsr_half/sv->Cajsr))*(p->Cajsr_half/sv->Cajsr))*(p->Cajsr_half/sv->Cajsr))*(p->Cajsr_half/sv->Cajsr))*(p->Cajsr_half/sv->Cajsr))*(p->Cajsr_half/sv->Cajsr))*(p->Cajsr_half/sv->Cajsr))));
    GlobalData_t Jrel_infp_b = (((-a_relp)*ICaL_ss)/(1.+((((((((p->Cajsr_half/sv->Cajsr)*(p->Cajsr_half/sv->Cajsr))*(p->Cajsr_half/sv->Cajsr))*(p->Cajsr_half/sv->Cajsr))*(p->Cajsr_half/sv->Cajsr))*(p->Cajsr_half/sv->Cajsr))*(p->Cajsr_half/sv->Cajsr))*(p->Cajsr_half/sv->Cajsr))));
    GlobalData_t Jtr = ((sv->Cansr-(sv->Cajsr))/60.);
    GlobalData_t Km2n = sv->jCa;
    GlobalData_t diff_CaMKt = (((aCaMK*CaMKb)*(CaMKb+sv->CaMKt))-((bCaMK*sv->CaMKt)));
    GlobalData_t fJrelp = (1./(1.+(KmCaMK/CaMKa)));
    GlobalData_t fJupp = (1./(1.+(KmCaMK/CaMKa)));
    GlobalData_t tau_rel_b = (bt/(1.+(0.0123/sv->Cajsr)));
    GlobalData_t tau_relp_b = (btp/(1.+(0.0123/sv->Cajsr)));
    GlobalData_t Jrel = (p->Jrel_b*(((1.-(fJrelp))*sv->Jrel_np)+(fJrelp*sv->Jrel_p)));
    GlobalData_t Jrel_inf = ((p->celltype==2.) ? (Jrel_inf_b*1.7) : Jrel_inf_b);
    GlobalData_t Jrel_infp = ((p->celltype==2.) ? (Jrel_infp_b*1.7) : Jrel_infp_b);
    GlobalData_t Jup = (p->Jup_b*((((1.-(fJupp))*Cai_row[Jupnp_idx])+(fJupp*Cai_row[Jupp_idx]))-(JleaK)));
    GlobalData_t anCa_i = (1./((K2n/Km2n)+((((1.+(Kmn/sv->Cai))*(1.+(Kmn/sv->Cai)))*(1.+(Kmn/sv->Cai)))*(1.+(Kmn/sv->Cai)))));
    GlobalData_t anCa_ss = (1./((K2n/Km2n)+((((1.+(Kmn/sv->Cass))*(1.+(Kmn/sv->Cass)))*(1.+(Kmn/sv->Cass)))*(1.+(Kmn/sv->Cass)))));
    GlobalData_t diff_Ki = ((((-(((((((Ito+IKr)+IKs)+IK1)+IKb)+IKATP)-((2.*INaK)))+ICaK_i))*ACap)/(F*Vmyo))+((JdiffK*Vss)/Vmyo));
    GlobalData_t diff_Kss = ((((-ICaK_ss)*ACap)/(F*Vss))-(JdiffK));
    GlobalData_t diff_Nai = ((((-(((((INa+INaL)+(3.*INaCa_i))+ICaNa_i)+(3.*INaK))+INab))*ACap)/(F*Vmyo))+((JdiffNa*Vss)/Vmyo));
    GlobalData_t diff_Nass = ((((-(ICaNa_ss+(3.*INaCa_ss)))*ACap)/(F*Vss))-(JdiffNa));
    GlobalData_t diff_a = ((V_row[ass_idx]-(sv->a))/V_row[ta_idx]);
    GlobalData_t diff_ap = ((V_row[assp_idx]-(sv->ap))/V_row[ta_idx]);
    GlobalData_t diff_d = ((V_row[dss_idx]-(sv->d))/V_row[td_idx]);
    GlobalData_t diff_ff = ((V_row[fss_idx]-(sv->ff))/V_row[tff_idx]);
    GlobalData_t diff_fs = ((V_row[fss_idx]-(sv->fs))/V_row[tfs_idx]);
    GlobalData_t diff_hL = ((V_row[hLss_idx]-(sv->hL))/p->thL);
    GlobalData_t diff_hLp = ((V_row[hLssp_idx]-(sv->hLp))/thLp);
    GlobalData_t diff_jCa = ((V_row[jCass_idx]-(sv->jCa))/tjCa);
    GlobalData_t diff_m = ((V_row[mss_idx]-(sv->m))/V_row[tm_idx]);
    GlobalData_t diff_mL = ((V_row[mLss_idx]-(sv->mL))/V_row[tmL_idx]);
    GlobalData_t diff_xs1 = ((V_row[xs1ss_idx]-(sv->xs1))/V_row[txs1_idx]);
    GlobalData_t tau_rel = ((tau_rel_b<0.001) ? 0.001 : tau_rel_b);
    GlobalData_t tau_relp = ((tau_relp_b<0.001) ? 0.001 : tau_relp_b);
    GlobalData_t diff_C2 = (((V_row[alpha_idx]*sv->C3)+(beta_1*sv->C1))-(((V_row[beta_idx]+alpha_1)*sv->C2)));
    GlobalData_t diff_C3 = ((V_row[beta_idx]*sv->C2)-((V_row[alpha_idx]*sv->C3)));
    GlobalData_t diff_Cai = (Cai_row[BCai_idx]*(((((-(((ICaL_i+Cai_row[IpCa_idx])+ICab)-((2.*INaCa_i))))*ACap)/((2.*F)*Vmyo))-(((Jup*Vnsr)/Vmyo)))+((Jdiff*Vss)/Vmyo)));
    GlobalData_t diff_Cajsr = (BCajsr*(Jtr-(Jrel)));
    GlobalData_t diff_Cansr = (Jup-(((Jtr*Vjsr)/Vnsr)));
    GlobalData_t diff_Cass = (BCass*(((((-(ICaL_ss-((2.*INaCa_ss))))*ACap)/((2.*F)*Vss))+((Jrel*Vjsr)/Vss))-(Jdiff)));
    GlobalData_t diff_Jrel_np = ((Jrel_inf-(sv->Jrel_np))/tau_rel);
    GlobalData_t diff_Jrel_p = ((Jrel_infp-(sv->Jrel_p))/tau_relp);
    GlobalData_t diff_O = (((V_row[alpha_2_idx]*sv->C1)+(V_row[beta_i_idx]*sv->I))-(((V_row[beta_2_idx]+V_row[alpha_i_idx])*sv->O)));
    GlobalData_t diff_fCaf = ((V_row[fCass_idx]-(sv->fCaf))/V_row[tfCaf_idx]);
    GlobalData_t diff_fCafp = ((V_row[fCass_idx]-(sv->fCafp))/V_row[tfCafp_idx]);
    GlobalData_t diff_fCas = ((V_row[fCass_idx]-(sv->fCas))/V_row[tfCas_idx]);
    GlobalData_t diff_ffp = ((V_row[fss_idx]-(sv->ffp))/V_row[tffp_idx]);
    GlobalData_t diff_h = ((V_row[hss_idx]-(sv->h))/V_row[th_idx]);
    GlobalData_t diff_hp = ((V_row[hssp_idx]-(sv->hp))/V_row[th_idx]);
    GlobalData_t diff_iF = ((V_row[iss_idx]-(sv->iF))/V_row[tiF_idx]);
    GlobalData_t diff_iS = ((V_row[iss_idx]-(sv->iS))/V_row[tiS_idx]);
    GlobalData_t diff_j = ((V_row[jss_idx]-(sv->j))/V_row[tj_idx]);
    GlobalData_t diff_nCa_i = ((anCa_i*K2n)-((sv->nCa_i*Km2n)));
    GlobalData_t diff_nCa_ss = ((anCa_ss*K2n)-((sv->nCa_ss*Km2n)));
    GlobalData_t diff_xs2 = ((V_row[xs2ss_idx]-(sv->xs2))/V_row[txs2_idx]);
    GlobalData_t diff_C1 = ((((alpha_1*sv->C2)+(V_row[beta_2_idx]*sv->O))+(V_row[beta_ItoC2_idx]*sv->I))-((((beta_1+V_row[alpha_2_idx])+V_row[alpha_C2ToI_idx])*sv->C1)));
    GlobalData_t diff_I = (((V_row[alpha_C2ToI_idx]*sv->C1)+(V_row[alpha_i_idx]*sv->O))-(((V_row[beta_ItoC2_idx]+V_row[beta_i_idx])*sv->I)));
    GlobalData_t diff_iFp = ((V_row[iss_idx]-(sv->iFp))/V_row[tiFp_idx]);
    GlobalData_t diff_iSp = ((V_row[iss_idx]-(sv->iSp))/V_row[tiSp_idx]);
    GlobalData_t diff_jp = ((V_row[jss_idx]-(sv->jp))/V_row[tjp_idx]);
    GlobalData_t C1_new = sv->C1+diff_C1*dt;
    GlobalData_t C2_new = sv->C2+diff_C2*dt;
    GlobalData_t C3_new = sv->C3+diff_C3*dt;
    GlobalData_t CaMKt_new = sv->CaMKt+diff_CaMKt*dt;
    GlobalData_t Cai_new = sv->Cai+diff_Cai*dt;
    GlobalData_t Cajsr_new = sv->Cajsr+diff_Cajsr*dt;
    GlobalData_t Cansr_new = sv->Cansr+diff_Cansr*dt;
    GlobalData_t Cass_new = sv->Cass+diff_Cass*dt;
    GlobalData_t I_new = sv->I+diff_I*dt;
    GlobalData_t Jrel_np_new = sv->Jrel_np+diff_Jrel_np*dt;
    GlobalData_t Jrel_p_new = sv->Jrel_p+diff_Jrel_p*dt;
    GlobalData_t Ki_new = sv->Ki+diff_Ki*dt;
    GlobalData_t Kss_new = sv->Kss+diff_Kss*dt;
    GlobalData_t Nai_new = sv->Nai+diff_Nai*dt;
    GlobalData_t Nass_new = sv->Nass+diff_Nass*dt;
    GlobalData_t O_new = sv->O+diff_O*dt;
    GlobalData_t a_new = sv->a+diff_a*dt;
    GlobalData_t ap_new = sv->ap+diff_ap*dt;
    GlobalData_t d_new = sv->d+diff_d*dt;
    GlobalData_t fCaf_new = sv->fCaf+diff_fCaf*dt;
    GlobalData_t fCafp_new = sv->fCafp+diff_fCafp*dt;
    GlobalData_t fCas_new = sv->fCas+diff_fCas*dt;
    GlobalData_t ff_new = sv->ff+diff_ff*dt;
    GlobalData_t ffp_new = sv->ffp+diff_ffp*dt;
    GlobalData_t fs_new = sv->fs+diff_fs*dt;
    GlobalData_t h_new = sv->h+diff_h*dt;
    GlobalData_t hL_new = sv->hL+diff_hL*dt;
    GlobalData_t hLp_new = sv->hLp+diff_hLp*dt;
    GlobalData_t hp_new = sv->hp+diff_hp*dt;
    GlobalData_t iF_new = sv->iF+diff_iF*dt;
    GlobalData_t iFp_new = sv->iFp+diff_iFp*dt;
    GlobalData_t iS_new = sv->iS+diff_iS*dt;
    GlobalData_t iSp_new = sv->iSp+diff_iSp*dt;
    GlobalData_t j_new = sv->j+diff_j*dt;
    GlobalData_t jCa_new = sv->jCa+diff_jCa*dt;
    GlobalData_t jp_new = sv->jp+diff_jp*dt;
    GlobalData_t m_new = sv->m+diff_m*dt;
    GlobalData_t mL_new = sv->mL+diff_mL*dt;
    GlobalData_t nCa_i_new = sv->nCa_i+diff_nCa_i*dt;
    GlobalData_t nCa_ss_new = sv->nCa_ss+diff_nCa_ss*dt;
    GlobalData_t xs1_new = sv->xs1+diff_xs1*dt;
    GlobalData_t xs2_new = sv->xs2+diff_xs2*dt;
    
    
    //Complete Rush Larsen Update
    
    
    //Complete RK2 Update
    
    
    //Complete RK4 Update
    
    
    //Complete Sundnes Update
    
    
    //Complete Markov Backward Euler method
    
    
    //Complete Rosenbrock Update
    
    
    //Complete Cvode Update
    
    
    //Finish the update
    if (C1_new < 0) { IIF_warn(__i, "C1 < 0, clamping to 0"); sv->C1 = 0; }
    else if (C1_new > 1) { IIF_warn(__i, "C1 > 1, clamping to 1"); sv->C1 = 1; }
    else {sv->C1 = C1_new;}
    if (C2_new < 0) { IIF_warn(__i, "C2 < 0, clamping to 0"); sv->C2 = 0; }
    else if (C2_new > 1) { IIF_warn(__i, "C2 > 1, clamping to 1"); sv->C2 = 1; }
    else {sv->C2 = C2_new;}
    if (C3_new < 0) { IIF_warn(__i, "C3 < 0, clamping to 0"); sv->C3 = 0; }
    else if (C3_new > 1) { IIF_warn(__i, "C3 > 1, clamping to 1"); sv->C3 = 1; }
    else {sv->C3 = C3_new;}
    if (CaMKt_new < 1e-9) { IIF_warn(__i, "CaMKt < 1e-9, clamping to 1e-9"); sv->CaMKt = 1e-9; }
    else {sv->CaMKt = CaMKt_new;}
    if (Cai_new < 1e-9) { IIF_warn(__i, "Cai < 1e-9, clamping to 1e-9"); sv->Cai = 1e-9; }
    else {sv->Cai = Cai_new;}
    if (Cajsr_new < 1e-9) { IIF_warn(__i, "Cajsr < 1e-9, clamping to 1e-9"); sv->Cajsr = 1e-9; }
    else {sv->Cajsr = Cajsr_new;}
    if (Cansr_new < 1e-9) { IIF_warn(__i, "Cansr < 1e-9, clamping to 1e-9"); sv->Cansr = 1e-9; }
    else {sv->Cansr = Cansr_new;}
    if (Cass_new < 1e-9) { IIF_warn(__i, "Cass < 1e-9, clamping to 1e-9"); sv->Cass = 1e-9; }
    else {sv->Cass = Cass_new;}
    if (I_new < 0) { IIF_warn(__i, "I < 0, clamping to 0"); sv->I = 0; }
    else if (I_new > 1) { IIF_warn(__i, "I > 1, clamping to 1"); sv->I = 1; }
    else {sv->I = I_new;}
    Iion = Iion;
    if (Jrel_np_new < 0) { IIF_warn(__i, "Jrel_np < 0, clamping to 0"); sv->Jrel_np = 0; }
    else if (Jrel_np_new > 1) { IIF_warn(__i, "Jrel_np > 1, clamping to 1"); sv->Jrel_np = 1; }
    else {sv->Jrel_np = Jrel_np_new;}
    if (Jrel_p_new < 0) { IIF_warn(__i, "Jrel_p < 0, clamping to 0"); sv->Jrel_p = 0; }
    else if (Jrel_p_new > 1) { IIF_warn(__i, "Jrel_p > 1, clamping to 1"); sv->Jrel_p = 1; }
    else {sv->Jrel_p = Jrel_p_new;}
    if (Ki_new < 1e-9) { IIF_warn(__i, "Ki < 1e-9, clamping to 1e-9"); sv->Ki = 1e-9; }
    else {sv->Ki = Ki_new;}
    if (Kss_new < 1e-9) { IIF_warn(__i, "Kss < 1e-9, clamping to 1e-9"); sv->Kss = 1e-9; }
    else {sv->Kss = Kss_new;}
    if (Nai_new < 1e-9) { IIF_warn(__i, "Nai < 1e-9, clamping to 1e-9"); sv->Nai = 1e-9; }
    else {sv->Nai = Nai_new;}
    if (Nass_new < 1e-9) { IIF_warn(__i, "Nass < 1e-9, clamping to 1e-9"); sv->Nass = 1e-9; }
    else {sv->Nass = Nass_new;}
    if (O_new < 0) { IIF_warn(__i, "O < 0, clamping to 0"); sv->O = 0; }
    else if (O_new > 1) { IIF_warn(__i, "O > 1, clamping to 1"); sv->O = 1; }
    else {sv->O = O_new;}
    if (a_new < 0) { IIF_warn(__i, "a < 0, clamping to 0"); sv->a = 0; }
    else if (a_new > 1) { IIF_warn(__i, "a > 1, clamping to 1"); sv->a = 1; }
    else {sv->a = a_new;}
    if (ap_new < 0) { IIF_warn(__i, "ap < 0, clamping to 0"); sv->ap = 0; }
    else if (ap_new > 1) { IIF_warn(__i, "ap > 1, clamping to 1"); sv->ap = 1; }
    else {sv->ap = ap_new;}
    if (d_new < 0) { IIF_warn(__i, "d < 0, clamping to 0"); sv->d = 0; }
    else if (d_new > 1) { IIF_warn(__i, "d > 1, clamping to 1"); sv->d = 1; }
    else {sv->d = d_new;}
    if (fCaf_new < 0) { IIF_warn(__i, "fCaf < 0, clamping to 0"); sv->fCaf = 0; }
    else if (fCaf_new > 1) { IIF_warn(__i, "fCaf > 1, clamping to 1"); sv->fCaf = 1; }
    else {sv->fCaf = fCaf_new;}
    if (fCafp_new < 0) { IIF_warn(__i, "fCafp < 0, clamping to 0"); sv->fCafp = 0; }
    else if (fCafp_new > 1) { IIF_warn(__i, "fCafp > 1, clamping to 1"); sv->fCafp = 1; }
    else {sv->fCafp = fCafp_new;}
    if (fCas_new < 0) { IIF_warn(__i, "fCas < 0, clamping to 0"); sv->fCas = 0; }
    else if (fCas_new > 1) { IIF_warn(__i, "fCas > 1, clamping to 1"); sv->fCas = 1; }
    else {sv->fCas = fCas_new;}
    if (ff_new < 0) { IIF_warn(__i, "ff < 0, clamping to 0"); sv->ff = 0; }
    else if (ff_new > 1) { IIF_warn(__i, "ff > 1, clamping to 1"); sv->ff = 1; }
    else {sv->ff = ff_new;}
    if (ffp_new < 0) { IIF_warn(__i, "ffp < 0, clamping to 0"); sv->ffp = 0; }
    else if (ffp_new > 1) { IIF_warn(__i, "ffp > 1, clamping to 1"); sv->ffp = 1; }
    else {sv->ffp = ffp_new;}
    if (fs_new < 0) { IIF_warn(__i, "fs < 0, clamping to 0"); sv->fs = 0; }
    else if (fs_new > 1) { IIF_warn(__i, "fs > 1, clamping to 1"); sv->fs = 1; }
    else {sv->fs = fs_new;}
    if (h_new < 0) { IIF_warn(__i, "h < 0, clamping to 0"); sv->h = 0; }
    else if (h_new > 1) { IIF_warn(__i, "h > 1, clamping to 1"); sv->h = 1; }
    else {sv->h = h_new;}
    if (hL_new < 0) { IIF_warn(__i, "hL < 0, clamping to 0"); sv->hL = 0; }
    else if (hL_new > 1) { IIF_warn(__i, "hL > 1, clamping to 1"); sv->hL = 1; }
    else {sv->hL = hL_new;}
    if (hLp_new < 0) { IIF_warn(__i, "hLp < 0, clamping to 0"); sv->hLp = 0; }
    else if (hLp_new > 1) { IIF_warn(__i, "hLp > 1, clamping to 1"); sv->hLp = 1; }
    else {sv->hLp = hLp_new;}
    if (hp_new < 0) { IIF_warn(__i, "hp < 0, clamping to 0"); sv->hp = 0; }
    else if (hp_new > 1) { IIF_warn(__i, "hp > 1, clamping to 1"); sv->hp = 1; }
    else {sv->hp = hp_new;}
    if (iF_new < 0) { IIF_warn(__i, "iF < 0, clamping to 0"); sv->iF = 0; }
    else if (iF_new > 1) { IIF_warn(__i, "iF > 1, clamping to 1"); sv->iF = 1; }
    else {sv->iF = iF_new;}
    if (iFp_new < 0) { IIF_warn(__i, "iFp < 0, clamping to 0"); sv->iFp = 0; }
    else if (iFp_new > 1) { IIF_warn(__i, "iFp > 1, clamping to 1"); sv->iFp = 1; }
    else {sv->iFp = iFp_new;}
    if (iS_new < 0) { IIF_warn(__i, "iS < 0, clamping to 0"); sv->iS = 0; }
    else if (iS_new > 1) { IIF_warn(__i, "iS > 1, clamping to 1"); sv->iS = 1; }
    else {sv->iS = iS_new;}
    if (iSp_new < 0) { IIF_warn(__i, "iSp < 0, clamping to 0"); sv->iSp = 0; }
    else if (iSp_new > 1) { IIF_warn(__i, "iSp > 1, clamping to 1"); sv->iSp = 1; }
    else {sv->iSp = iSp_new;}
    if (j_new < 0) { IIF_warn(__i, "j < 0, clamping to 0"); sv->j = 0; }
    else if (j_new > 1) { IIF_warn(__i, "j > 1, clamping to 1"); sv->j = 1; }
    else {sv->j = j_new;}
    if (jCa_new < 0) { IIF_warn(__i, "jCa < 0, clamping to 0"); sv->jCa = 0; }
    else if (jCa_new > 1) { IIF_warn(__i, "jCa > 1, clamping to 1"); sv->jCa = 1; }
    else {sv->jCa = jCa_new;}
    if (jp_new < 0) { IIF_warn(__i, "jp < 0, clamping to 0"); sv->jp = 0; }
    else if (jp_new > 1) { IIF_warn(__i, "jp > 1, clamping to 1"); sv->jp = 1; }
    else {sv->jp = jp_new;}
    if (m_new < 0) { IIF_warn(__i, "m < 0, clamping to 0"); sv->m = 0; }
    else if (m_new > 1) { IIF_warn(__i, "m > 1, clamping to 1"); sv->m = 1; }
    else {sv->m = m_new;}
    if (mL_new < 0) { IIF_warn(__i, "mL < 0, clamping to 0"); sv->mL = 0; }
    else if (mL_new > 1) { IIF_warn(__i, "mL > 1, clamping to 1"); sv->mL = 1; }
    else {sv->mL = mL_new;}
    if (nCa_i_new < 0) { IIF_warn(__i, "nCa_i < 0, clamping to 0"); sv->nCa_i = 0; }
    else if (nCa_i_new > 1) { IIF_warn(__i, "nCa_i > 1, clamping to 1"); sv->nCa_i = 1; }
    else {sv->nCa_i = nCa_i_new;}
    if (nCa_ss_new < 0) { IIF_warn(__i, "nCa_ss < 0, clamping to 0"); sv->nCa_ss = 0; }
    else if (nCa_ss_new > 1) { IIF_warn(__i, "nCa_ss > 1, clamping to 1"); sv->nCa_ss = 1; }
    else {sv->nCa_ss = nCa_ss_new;}
    if (xs1_new < 0) { IIF_warn(__i, "xs1 < 0, clamping to 0"); sv->xs1 = 0; }
    else if (xs1_new > 1) { IIF_warn(__i, "xs1 > 1, clamping to 1"); sv->xs1 = 1; }
    else {sv->xs1 = xs1_new;}
    if (xs2_new < 0) { IIF_warn(__i, "xs2 < 0, clamping to 0"); sv->xs2 = 0; }
    else if (xs2_new > 1) { IIF_warn(__i, "xs2 > 1, clamping to 1"); sv->xs2 = 1; }
    else {sv->xs2 = xs2_new;}
    //Change the units of external variables as appropriate.
    sv->Cai *= 1e3;
    
    
    //Save all external vars
    Iion_ext[__i] = Iion;
    V_ext[__i] = V;

  }

            }
}
#endif // TOMEK_EDIT_CPU_GENERATED

bool Tomek_editIonType::has_trace() const {
    return true;
}

void Tomek_editIonType::trace(IonIfBase& imp_base, int node, FILE* file, GlobalData_t** impdata) const
{
  IonIfDerived& imp = static_cast<IonIfDerived&>(imp_base);
  static bool first = true;
  if (first) {
    first = false;
    FILE_SPEC theader = f_open("Tomek_edit_trace_header.txt","wt");
    fprintf(theader->fd,
        "Iion\n"
        "V\n"
      );

    f_close(theader);
  }

  GlobalData_t dt = imp.get_dt() * 1e0;
  cell_geom *region = &imp.cgeom();
  Tomek_edit_Params *p  = imp.params();

  Tomek_edit_state *sv_base = (Tomek_edit_state *)imp.sv_tab().data();

  Tomek_edit_state *sv = sv_base+node;
  int __i = node;

  GlobalData_t t = imp.get_tstp().cnt * dt;
  IonIfBase* IF = &imp;
  // Define the constants that depend on the parameters.
  GlobalData_t GK1 = ((p->celltype==1.) ? (p->GK1_b*1.2) : ((p->celltype==2.) ? (p->GK1_b*1.3) : p->GK1_b));
  GlobalData_t GKb = ((p->celltype==1.) ? (GKb_b*0.6) : GKb_b);
  GlobalData_t GKr = ((p->celltype==1.) ? (p->GKr_b*1.3) : ((p->celltype==2.) ? (p->GKr_b*0.8) : p->GKr_b));
  GlobalData_t GKs = ((p->celltype==1.) ? (GKs_b*1.4) : GKs_b);
  GlobalData_t GNaL = ((p->celltype==1.) ? (p->GNaL_b*0.6) : p->GNaL_b);
  GlobalData_t Gncx = ((p->celltype==1.) ? (p->Gncx_b*1.1) : ((p->celltype==2.) ? (p->Gncx_b*1.4) : p->Gncx_b));
  GlobalData_t Gto = ((p->celltype==1.) ? (p->Gto_b*2.) : ((p->celltype==2.) ? (p->Gto_b*2.) : p->Gto_b));
  GlobalData_t PCa = ((p->celltype==1.) ? (PCa_b*1.2) : ((p->celltype==2.) ? (PCa_b*2.) : PCa_b));
  GlobalData_t PNaK = ((p->celltype==1.) ? (p->PNaK_b*0.9) : ((p->celltype==2.) ? (p->PNaK_b*0.7) : p->PNaK_b));
  GlobalData_t cmdnmax = ((p->celltype==1.) ? (cmdnmax_b*1.3) : cmdnmax_b);
  GlobalData_t thLp = (3.*p->thL);
  GlobalData_t upSCale = ((p->celltype==1.) ? 1.3 : 1.);
  GlobalData_t PCaK = (3.574e-4*PCa);
  GlobalData_t PCaNa = (0.00125*PCa);
  GlobalData_t PCap = (1.1*PCa);
  GlobalData_t PCaKp = (3.574e-4*PCap);
  GlobalData_t PCaNap = (0.00125*PCap);
  //Prepare all the public arrays.
  GlobalData_t *Iion_ext = impdata[Iion];
  GlobalData_t *V_ext = impdata[Vm];
  //Prepare all the private functions.
  //Initialize the external vars to their current values
  GlobalData_t Iion = Iion_ext[__i];
  GlobalData_t V = V_ext[__i];
  //Change the units of external variables as appropriate.
  sv->Cai *= 1e-3;
  
  
  GlobalData_t AfCaf = (0.3+(0.6/(1.+(exp(((V-(10.))/10.))))));
  GlobalData_t AiF = (1./(1.+(exp((((V+EKshift)-(213.6))/151.2)))));
  GlobalData_t CaMKb = ((p->CaMKo*(1.-(sv->CaMKt)))/(1.+(KmCaM/sv->Cass)));
  GlobalData_t EK = (((R*T)/(zK*F))*(log((Ko/sv->Ki))));
  GlobalData_t EKs = (((R*T)/(zK*F))*(log(((Ko+(PKNa*Nao))/(sv->Ki+(PKNa*sv->Nai))))));
  GlobalData_t ENa = (((R*T)/(zNa*F))*(log((Nao/sv->Nai))));
  GlobalData_t IClCa_junc = (((Fjunc*GClCa)/(1.+(KdClCa/sv->Cass)))*(V-(ECl)));
  GlobalData_t IClCa_sl = ((((1.-(Fjunc))*GClCa)/(1.+(KdClCa/sv->Cai)))*(V-(ECl)));
  GlobalData_t IClb = (GClb*(V-(ECl)));
  GlobalData_t Ii = ((0.5*(((sv->Nai+sv->Ki)+Cli)+(4.*sv->Cai)))/1000.);
  GlobalData_t IpCa = ((GpCa*sv->Cai)/(KmCap+sv->Cai));
  GlobalData_t Iss = ((0.5*(((sv->Nass+sv->Kss)+Cli)+(4.*sv->Cass)))/1000.);
  GlobalData_t KsCa = (1.+(0.6/(1.+(pow((3.8e-5/sv->Cai),1.4)))));
  GlobalData_t P = (eP/(((1.+(H/Khp))+(sv->Nai/KNap))+(sv->Ki/KxKur)));
  GlobalData_t Vffrt = (((V*F)*F)/(R*T));
  GlobalData_t Vfrt = ((V*F)/(R*T));
  GlobalData_t allo_i = (1./(1.+((KmCaAct/sv->Cai)*(KmCaAct/sv->Cai))));
  GlobalData_t allo_ss = (1./(1.+((KmCaAct/sv->Cass)*(KmCaAct/sv->Cass))));
  GlobalData_t f = ((Aff*sv->ff)+(Afs*sv->fs));
  GlobalData_t fp = ((Aff*sv->ffp)+(Afs*sv->fs));
  GlobalData_t h4_i = (1.+((sv->Nai/KNa1)*(1.+(sv->Nai/KNa2))));
  GlobalData_t h4_ss = (1.+((sv->Nass/KNa1)*(1.+(sv->Nass/KNa2))));
  GlobalData_t xKb = (1./(1.+(exp(((-(V-(10.8968)))/23.9871)))));
  GlobalData_t AfCas = (1.-(AfCaf));
  GlobalData_t AiS = (1.-(AiF));
  GlobalData_t CaMKa = (CaMKb+sv->CaMKt);
  GlobalData_t IClCa = (IClCa_junc+IClCa_sl);
  GlobalData_t IKATP = ((((fKatp*gKatp)*aKiK)*bKiK)*(V-(EK)));
  GlobalData_t IKb = ((GKb*xKb)*(V-(EK)));
  GlobalData_t IKr = (((GKr*(sqrt((Ko/5.))))*sv->O)*(V-(EK)));
  GlobalData_t IKs = ((((GKs*KsCa)*sv->xs1)*sv->xs2)*(V-(EKs)));
  GlobalData_t INab = (((PNab*Vffrt)*((sv->Nai*(exp(Vfrt)))-(Nao)))/((exp(Vfrt))-(1.)));
  GlobalData_t KNai = (KNai0*(exp(((delta*Vfrt)/3.))));
  GlobalData_t KNao = (KNao0*(exp((((1.-(delta))*Vfrt)/3.))));
  GlobalData_t aK1 = (4.094/(1.+(exp((0.1217*((V-(EK))-(49.934)))))));
  GlobalData_t b3 = (((K3m*P)*H)/(1.+(MgATP/Kmgatp)));
  GlobalData_t bK1 = (((15.72*(exp((0.0674*((V-(EK))-(3.257))))))+(exp((0.0618*((V-(EK))-(594.31))))))/(1.+(exp((-0.1629*((V-(EK))+14.207))))));
  GlobalData_t gamma_Cai = (exp((((-constA)*4.)*(((sqrt(Ii))/(1.+(sqrt(Ii))))-((0.3*Ii))))));
  GlobalData_t gamma_Cass = (exp((((-constA)*4.)*(((sqrt(Iss))/(1.+(sqrt(Iss))))-((0.3*Iss))))));
  GlobalData_t gamma_Ki = (exp(((-constA)*(((sqrt(Ii))/(1.+(sqrt(Ii))))-((0.3*Ii))))));
  GlobalData_t gamma_Kss = (exp(((-constA)*(((sqrt(Iss))/(1.+(sqrt(Iss))))-((0.3*Iss))))));
  GlobalData_t gamma_Nai = (exp(((-constA)*(((sqrt(Ii))/(1.+(sqrt(Ii))))-((0.3*Ii))))));
  GlobalData_t gamma_Nass = (exp(((-constA)*(((sqrt(Iss))/(1.+(sqrt(Iss))))-((0.3*Iss))))));
  GlobalData_t h5_i = ((sv->Nai*sv->Nai)/((h4_i*KNa1)*KNa2));
  GlobalData_t h5_ss = ((sv->Nass*sv->Nass)/((h4_ss*KNa1)*KNa2));
  GlobalData_t h6_i = (1./h4_i);
  GlobalData_t h6_ss = (1./h4_ss);
  GlobalData_t hCa = (exp((qCa*Vfrt)));
  GlobalData_t hNa = (exp((qNa*Vfrt)));
  GlobalData_t ICab = ((((PCab*4.)*Vffrt)*(((gamma_Cai*sv->Cai)*(exp((2.*Vfrt))))-((gamma_Cao*Cao))))/((exp((2.*Vfrt)))-(1.)));
  GlobalData_t K1ss = (aK1/(aK1+bK1));
  GlobalData_t K6_i = ((h6_i*sv->Cai)*KCaon);
  GlobalData_t K6_ss = ((h6_ss*sv->Cass)*KCaon);
  GlobalData_t PhiCaK_i = ((Vffrt*(((gamma_Ki*sv->Ki)*(exp(Vfrt)))-((gamma_Ko*Ko))))/((exp(Vfrt))-(1.)));
  GlobalData_t PhiCaK_ss = ((Vffrt*(((gamma_Kss*sv->Kss)*(exp(Vfrt)))-((gamma_Ko*Ko))))/((exp(Vfrt))-(1.)));
  GlobalData_t PhiCaL_i = (((4.*Vffrt)*(((gamma_Cai*sv->Cai)*(exp((2.*Vfrt))))-((gamma_Cao*Cao))))/((exp((2.*Vfrt)))-(1.)));
  GlobalData_t PhiCaL_ss = (((4.*Vffrt)*(((gamma_Cass*sv->Cass)*(exp((2.*Vfrt))))-((gamma_Cao*Cao))))/((exp((2.*Vfrt)))-(1.)));
  GlobalData_t PhiCaNa_i = ((Vffrt*(((gamma_Nai*sv->Nai)*(exp(Vfrt)))-((gamma_Nao*Nao))))/((exp(Vfrt))-(1.)));
  GlobalData_t PhiCaNa_ss = ((Vffrt*(((gamma_Nass*sv->Nass)*(exp(Vfrt)))-((gamma_Nao*Nao))))/((exp(Vfrt))-(1.)));
  GlobalData_t a1 = ((K1p*(((sv->Nai/KNai)*(sv->Nai/KNai))*(sv->Nai/KNai)))/(((((1.+(sv->Nai/KNai))*(1.+(sv->Nai/KNai)))*(1.+(sv->Nai/KNai)))+((1.+(sv->Ki/KKi))*(1.+(sv->Ki/KKi))))-(1.)));
  GlobalData_t a3 = ((K3p*((Ko/KKo)*(Ko/KKo)))/(((((1.+(Nao/KNao))*(1.+(Nao/KNao)))*(1.+(Nao/KNao)))+((1.+(Ko/KKo))*(1.+(Ko/KKo))))-(1.)));
  GlobalData_t b2 = ((K2m*(((Nao/KNao)*(Nao/KNao))*(Nao/KNao)))/(((((1.+(Nao/KNao))*(1.+(Nao/KNao)))*(1.+(Nao/KNao)))+((1.+(Ko/KKo))*(1.+(Ko/KKo))))-(1.)));
  GlobalData_t b4 = ((K4m*((sv->Ki/KKi)*(sv->Ki/KKi)))/(((((1.+(sv->Nai/KNai))*(1.+(sv->Nai/KNai)))*(1.+(sv->Nai/KNai)))+((1.+(sv->Ki/KKi))*(1.+(sv->Ki/KKi))))-(1.)));
  GlobalData_t fCa = ((AfCaf*sv->fCaf)+(AfCas*sv->fCas));
  GlobalData_t fCap = ((AfCaf*sv->fCafp)+(AfCas*sv->fCas));
  GlobalData_t fICaLp = (1./(1.+(KmCaMK/CaMKa)));
  GlobalData_t fINaLp = (1./(1.+(KmCaMK/CaMKa)));
  GlobalData_t fINap = (1./(1.+(KmCaMK/CaMKa)));
  GlobalData_t fItop = (1./(1.+(KmCaMK/CaMKa)));
  GlobalData_t h1_i = (1.+((sv->Nai/KNa3)*(1.+hNa)));
  GlobalData_t h1_ss = (1.+((sv->Nass/KNa3)*(1.+hNa)));
  GlobalData_t h7_i = (1.+((Nao/KNa3)*(1.+(1./hNa))));
  GlobalData_t h7_ss = (1.+((Nao/KNa3)*(1.+(1./hNa))));
  GlobalData_t i_t = ((AiF*sv->iF)+(AiS*sv->iS));
  GlobalData_t ip = ((AiF*sv->iFp)+(AiS*sv->iSp));
  GlobalData_t ICaK_i = ((1.-(ICaL_fractionSS))*((((((1.-(fICaLp))*PCaK)*PhiCaK_i)*sv->d)*((f*(1.-(sv->nCa_i)))+((sv->jCa*fCa)*sv->nCa_i)))+((((fICaLp*PCaKp)*PhiCaK_i)*sv->d)*((fp*(1.-(sv->nCa_i)))+((sv->jCa*fCap)*sv->nCa_i)))));
  GlobalData_t ICaK_ss = (ICaL_fractionSS*((((((1.-(fICaLp))*PCaK)*PhiCaK_ss)*sv->d)*((f*(1.-(sv->nCa_ss)))+((sv->jCa*fCa)*sv->nCa_ss)))+((((fICaLp*PCaKp)*PhiCaK_ss)*sv->d)*((fp*(1.-(sv->nCa_ss)))+((sv->jCa*fCap)*sv->nCa_ss)))));
  GlobalData_t ICaL_i = ((1.-(ICaL_fractionSS))*((((((1.-(fICaLp))*PCa)*PhiCaL_i)*sv->d)*((f*(1.-(sv->nCa_i)))+((sv->jCa*fCa)*sv->nCa_i)))+((((fICaLp*PCap)*PhiCaL_i)*sv->d)*((fp*(1.-(sv->nCa_i)))+((sv->jCa*fCap)*sv->nCa_i)))));
  GlobalData_t ICaL_ss = (ICaL_fractionSS*((((((1.-(fICaLp))*PCa)*PhiCaL_ss)*sv->d)*((f*(1.-(sv->nCa_ss)))+((sv->jCa*fCa)*sv->nCa_ss)))+((((fICaLp*PCap)*PhiCaL_ss)*sv->d)*((fp*(1.-(sv->nCa_ss)))+((sv->jCa*fCap)*sv->nCa_ss)))));
  GlobalData_t ICaNa_i = ((1.-(ICaL_fractionSS))*((((((1.-(fICaLp))*PCaNa)*PhiCaNa_i)*sv->d)*((f*(1.-(sv->nCa_i)))+((sv->jCa*fCa)*sv->nCa_i)))+((((fICaLp*PCaNap)*PhiCaNa_i)*sv->d)*((fp*(1.-(sv->nCa_i)))+((sv->jCa*fCap)*sv->nCa_i)))));
  GlobalData_t ICaNa_ss = (ICaL_fractionSS*((((((1.-(fICaLp))*PCaNa)*PhiCaNa_ss)*sv->d)*((f*(1.-(sv->nCa_ss)))+((sv->jCa*fCa)*sv->nCa_ss)))+((((fICaLp*PCaNap)*PhiCaNa_ss)*sv->d)*((fp*(1.-(sv->nCa_ss)))+((sv->jCa*fCap)*sv->nCa_ss)))));
  GlobalData_t IK1 = (((GK1*(sqrt((Ko/5.))))*K1ss)*(V-(EK)));
  GlobalData_t INa = (((GNa*(V-(ENa)))*((sv->m*sv->m)*sv->m))*((((1.-(fINap))*sv->h)*sv->j)+((fINap*sv->hp)*sv->jp)));
  GlobalData_t INaL = (((GNaL*(V-(ENa)))*sv->mL)*(((1.-(fINaLp))*sv->hL)+(fINaLp*sv->hLp)));
  GlobalData_t Ito = ((Gto*(V-(EK)))*((((1.-(fItop))*sv->a)*i_t)+((fItop*sv->ap)*ip)));
  GlobalData_t h2_i = ((sv->Nai*hNa)/(KNa3*h1_i));
  GlobalData_t h2_ss = ((sv->Nass*hNa)/(KNa3*h1_ss));
  GlobalData_t h3_i = (1./h1_i);
  GlobalData_t h3_ss = (1./h1_ss);
  GlobalData_t h8_i = (Nao/((KNa3*hNa)*h7_i));
  GlobalData_t h8_ss = (Nao/((KNa3*hNa)*h7_ss));
  GlobalData_t h9_i = (1./h7_i);
  GlobalData_t h9_ss = (1./h7_ss);
  GlobalData_t x1 = (((((a4*a1)*a2)+((b2*b4)*b3))+((a2*b4)*b3))+((b3*a1)*a2));
  GlobalData_t x2 = (((((b2*b1)*b4)+((a1*a2)*a3))+((a3*b1)*b4))+((a2*a3)*b4));
  GlobalData_t x3 = (((((a2*a3)*a4)+((b3*b2)*b1))+((b2*b1)*a4))+((a3*a4)*b1));
  GlobalData_t x4 = (((((b4*b3)*b2)+((a3*a4)*a1))+((b2*a4)*a1))+((b3*b2)*a1));
  GlobalData_t E1 = (x1/(((x1+x2)+x3)+x4));
  GlobalData_t E2 = (x2/(((x1+x2)+x3)+x4));
  GlobalData_t E3 = (x3/(((x1+x2)+x3)+x4));
  GlobalData_t E4 = (x4/(((x1+x2)+x3)+x4));
  GlobalData_t ICaK = (ICaK_ss+ICaK_i);
  GlobalData_t ICaL = (ICaL_ss+ICaL_i);
  GlobalData_t ICaNa = (ICaNa_ss+ICaNa_i);
  GlobalData_t K3p_i = (h9_i*wCa);
  GlobalData_t K3p_ss = (h9_ss*wCa);
  GlobalData_t K3pp_i = (h8_i*wNaCa);
  GlobalData_t K3pp_ss = (h8_ss*wNaCa);
  GlobalData_t K4p_i = ((h3_i*wCa)/hCa);
  GlobalData_t K4p_ss = ((h3_ss*wCa)/hCa);
  GlobalData_t K4pp_i = (h2_i*wNaCa);
  GlobalData_t K4pp_ss = (h2_ss*wNaCa);
  GlobalData_t K7_i = ((h5_i*h2_i)*wNa);
  GlobalData_t K7_ss = ((h5_ss*h2_ss)*wNa);
  GlobalData_t K8_i = ((h8_i*h11_i)*wNa);
  GlobalData_t K8_ss = ((h8_ss*h11_ss)*wNa);
  GlobalData_t JNaKK = (2.*((E4*b1)-((E3*a1))));
  GlobalData_t JNaKNa = (3.*((E1*a3)-((E2*b3))));
  GlobalData_t K3_i = (K3p_i+K3pp_i);
  GlobalData_t K3_ss = (K3p_ss+K3pp_ss);
  GlobalData_t K4_i = (K4p_i+K4pp_i);
  GlobalData_t K4_ss = (K4p_ss+K4pp_ss);
  GlobalData_t INaK = (PNaK*((zNa*JNaKNa)+(zK*JNaKK)));
  GlobalData_t x1_i = (((K2_i*K4_i)*(K7_i+K6_i))+((K5_i*K7_i)*(K2_i+K3_i)));
  GlobalData_t x1_ss = (((K2_ss*K4_ss)*(K7_ss+K6_ss))+((K5_ss*K7_ss)*(K2_ss+K3_ss)));
  GlobalData_t x2_i = (((K1_i*K7_i)*(K4_i+K5_i))+((K4_i*K6_i)*(K1_i+K8_i)));
  GlobalData_t x2_ss = (((K1_ss*K7_ss)*(K4_ss+K5_ss))+((K4_ss*K6_ss)*(K1_ss+K8_ss)));
  GlobalData_t x3_i = (((K1_i*K3_i)*(K7_i+K6_i))+((K8_i*K6_i)*(K2_i+K3_i)));
  GlobalData_t x3_ss = (((K1_ss*K3_ss)*(K7_ss+K6_ss))+((K8_ss*K6_ss)*(K2_ss+K3_ss)));
  GlobalData_t x4_i = (((K2_i*K8_i)*(K4_i+K5_i))+((K3_i*K5_i)*(K1_i+K8_i)));
  GlobalData_t x4_ss = (((K2_ss*K8_ss)*(K4_ss+K5_ss))+((K3_ss*K5_ss)*(K1_ss+K8_ss)));
  GlobalData_t E1_i = (x1_i/(((x1_i+x2_i)+x3_i)+x4_i));
  GlobalData_t E1_ss = (x1_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
  GlobalData_t E2_i = (x2_i/(((x1_i+x2_i)+x3_i)+x4_i));
  GlobalData_t E2_ss = (x2_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
  GlobalData_t E3_i = (x3_i/(((x1_i+x2_i)+x3_i)+x4_i));
  GlobalData_t E3_ss = (x3_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
  GlobalData_t E4_i = (x4_i/(((x1_i+x2_i)+x3_i)+x4_i));
  GlobalData_t E4_ss = (x4_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
  GlobalData_t JncxCa_i = ((E2_i*K2_i)-((E1_i*K1_i)));
  GlobalData_t JncxCa_ss = ((E2_ss*K2_ss)-((E1_ss*K1_ss)));
  GlobalData_t JncxNa_i = (((3.*((E4_i*K7_i)-((E1_i*K8_i))))+(E3_i*K4pp_i))-((E2_i*K3pp_i)));
  GlobalData_t JncxNa_ss = (((3.*((E4_ss*K7_ss)-((E1_ss*K8_ss))))+(E3_ss*K4pp_ss))-((E2_ss*K3pp_ss)));
  GlobalData_t INaCa_i = ((((1.-(INaCa_fractionSS))*Gncx)*allo_i)*((zNa*JncxNa_i)+(zCa*JncxCa_i)));
  GlobalData_t INaCa_ss = (((INaCa_fractionSS*Gncx)*allo_ss)*((zNa*JncxNa_ss)+(zCa*JncxCa_ss)));
  Iion = ((((((((((((((((((INa+INaL)+Ito)+ICaL)+ICaNa)+ICaK)+IKr)+IKs)+IK1)+INaCa_i)+INaCa_ss)+INaK)+INab)+IKb)+IpCa)+ICab)+IClCa)+IClb)+IKATP);
  //Output the desired variables
  fprintf(file, "%4.12f\t", Iion);
  fprintf(file, "%4.12f\t", V);
  //Change the units of external variables as appropriate.
  sv->Cai *= 1e3;
  
  

}
IonIfBase* Tomek_editIonType::make_ion_if(Target target, int num_node, const std::vector<std::reference_wrapper<IonType>>& plugins) const {
        // Place the allocated IonIf in managed memory if a GPU target exists for this model
        // otherwise, place it in main RAM
    IonIfDerived* ptr;
    if (this->select_target(Target::MLIR_ROCM) == Target::MLIR_ROCM) {
        ptr = allocate_on_target<IonIfDerived>(Target::MLIR_ROCM, 1, true);
    }
    else if (this->select_target(Target::MLIR_CUDA) == Target::MLIR_CUDA) {
        ptr = allocate_on_target<IonIfDerived>(Target::MLIR_CUDA, 1, true);
    }
    else {
        ptr = allocate_on_target<IonIfDerived>(Target::MLIR_CPU, 1, true);
    }
    // Using placement new to place the object in the correct memory
    return new(ptr) IonIfDerived(*this, this->select_target(target),
    num_node, plugins);
}

void Tomek_editIonType::destroy_ion_if(IonIfBase *imp) const {
    // Call destructor and deallocate manually because the object might
    // be located on GPU (delete won't work in this case)
    imp->~IonIfBase();
    IonIfDerived* ptr = static_cast<IonIfDerived *>(imp);
    if (this->select_target(Target::MLIR_ROCM) == Target::MLIR_ROCM) {
        deallocate_on_target<IonIfDerived>(Target::MLIR_ROCM, ptr);
    }
    else if (this->select_target(Target::MLIR_CUDA) == Target::MLIR_CUDA) {
        deallocate_on_target<IonIfDerived>(Target::MLIR_CUDA, ptr);
    }
    else {
        deallocate_on_target<IonIfDerived>(Target::MLIR_CPU, ptr);
    }
}

}  // namespace limpet
        