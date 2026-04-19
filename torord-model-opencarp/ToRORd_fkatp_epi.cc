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

*
*/
        

// DO NOT EDIT THIS SOURCE CODE FILE
// ANY CHANGES TO THIS FILE WILL BE OVERWRITTEN!!!!

#include "ToRORd_fkatp_epi.h"

#ifdef _OPENMP
#include <omp.h>
#endif



#ifdef USE_CVODE
#include <nvector/nvector_serial.h>
#include <cvode/cvode.h>
#include <cvode/cvode_diag.h>
#endif

namespace limpet {

using ::opencarp::f_open;
using ::opencarp::FILE_SPEC;

ToRORd_fkatp_epiIonType::ToRORd_fkatp_epiIonType(bool plugin) : IonType(std::move(std::string("ToRORd_fkatp_epi")), plugin) {}

size_t ToRORd_fkatp_epiIonType::params_size() const {
  return sizeof(struct ToRORd_fkatp_epi_Params);
}

size_t ToRORd_fkatp_epiIonType::dlo_vector_size() const {

  return 1;
}

uint32_t ToRORd_fkatp_epiIonType::reqdat() const {
  return ToRORd_fkatp_epi_REQDAT;
}

uint32_t ToRORd_fkatp_epiIonType::moddat() const {
  return ToRORd_fkatp_epi_MODDAT;
}

void ToRORd_fkatp_epiIonType::destroy(IonIfBase& imp_base) const {
  IonIfDerived& imp = static_cast<IonIfDerived&>(imp_base);
  imp.destroy_luts();
  // rarely need to do anything else
}

Target ToRORd_fkatp_epiIonType::select_target(Target target) const {
  switch (target) {
    case Target::AUTO:
#   ifdef TORORD_FKATP_EPI_MLIR_CUDA_GENERATED
      return Target::MLIR_CUDA;
#   elif defined(TORORD_FKATP_EPI_MLIR_ROCM_GENERATED)
      return Target::MLIR_ROCM;
#   elif defined(TORORD_FKATP_EPI_MLIR_CPU_GENERATED)
      return Target::MLIR_CPU;
#   elif defined(TORORD_FKATP_EPI_CPU_GENERATED)
      return Target::CPU;
#   else
      return Target::UNKNOWN;
#   endif
#   ifdef TORORD_FKATP_EPI_MLIR_CUDA_GENERATED
    case Target::MLIR_CUDA:
      return Target::MLIR_CUDA;
#   endif
#   ifdef TORORD_FKATP_EPI_MLIR_ROCM_GENERATED
    case Target::MLIR_ROCM:
      return Target::MLIR_ROCM;
#   endif
#   ifdef TORORD_FKATP_EPI_MLIR_CPU_GENERATED
    case Target::MLIR_CPU:
      return Target::MLIR_CPU;
#   endif
#   ifdef TORORD_FKATP_EPI_CPU_GENERATED
    case Target::CPU:
      return Target::CPU;
#   endif
    default:
      return Target::UNKNOWN;
  }
}

void ToRORd_fkatp_epiIonType::compute(Target target, int start, int end, IonIfBase& imp_base, GlobalData_t** data) const {
  IonIfDerived& imp = static_cast<IonIfDerived&>(imp_base);
  switch(target) {
    case Target::AUTO:
#   ifdef TORORD_FKATP_EPI_MLIR_CUDA_GENERATED
      compute_ToRORd_fkatp_epi_mlir_gpu_cuda(start, end, imp, data);
#   elif defined(TORORD_FKATP_EPI_MLIR_ROCM_GENERATED)
      compute_ToRORd_fkatp_epi_mlir_gpu_rocm(start, end, imp, data);
#   elif defined(TORORD_FKATP_EPI_MLIR_CPU_GENERATED)
      compute_ToRORd_fkatp_epi_mlir_cpu(start, end, imp, data);
#   elif defined(TORORD_FKATP_EPI_CPU_GENERATED)
      compute_ToRORd_fkatp_epi_cpu(start, end, imp, data);
#   else
#     error "Could not generate method ToRORd_fkatp_epiIonType::compute."
#   endif
      break;
#   ifdef TORORD_FKATP_EPI_MLIR_CUDA_GENERATED
    case Target::MLIR_CUDA:
      compute_ToRORd_fkatp_epi_mlir_gpu_cuda(start, end, imp, data);
      break;
#   endif
#   ifdef TORORD_FKATP_EPI_MLIR_ROCM_GENERATED
    case Target::MLIR_ROCM:
      compute_ToRORd_fkatp_epi_mlir_gpu_rocm(start, end, imp, data);
      break;
#   endif
#   ifdef TORORD_FKATP_EPI_MLIR_CPU_GENERATED
    case Target::MLIR_CPU:
      compute_ToRORd_fkatp_epi_mlir_cpu(start, end, imp, data);
      break;
#   endif
#   ifdef TORORD_FKATP_EPI_CPU_GENERATED
    case Target::CPU:
      compute_ToRORd_fkatp_epi_cpu(start, end, imp, data);
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
#define C1_init (GlobalData_t)(0.00067941)
#define C2_init (GlobalData_t)(0.00082869)
#define C3_init (GlobalData_t)(0.9982)
#define CaMKo (GlobalData_t)(0.05)
#define CaMKt_init (GlobalData_t)(0.0129)
#define EKshift (GlobalData_t)(0.)
#define F (GlobalData_t)(96485.)
#define Fjunc (GlobalData_t)(1.)
#define GClCa (GlobalData_t)(0.2843)
#define GClb (GlobalData_t)(1.98e-3)
#define GK1_b (GlobalData_t)(0.6992)
#define GKb_b (GlobalData_t)(0.0189)
#define GKr_b (GlobalData_t)(0.0321)
#define GKs_b (GlobalData_t)(0.0011)
#define GNa (GlobalData_t)(11.7802)
#define GNaL_b (GlobalData_t)(0.0279)
#define Gncx_b (GlobalData_t)(0.0034)
#define GpCa (GlobalData_t)(5e-04)
#define Gto_b (GlobalData_t)(0.16)
#define H (GlobalData_t)(1e-7)
#define ICaL_fractionSS (GlobalData_t)(0.8)
#define INaCa_fractionSS (GlobalData_t)(0.35)
#define I_init (GlobalData_t)(9.5416e-06)
#define Jrel_b (GlobalData_t)(1.5378)
#define Jrel_np_init (GlobalData_t)(2.8189e-24)
#define Jrel_p_init (GlobalData_t)(0.)
#define Jup_b (GlobalData_t)(1.0)
#define K_atp (GlobalData_t)(0.25)
#define K_o_n (GlobalData_t)(5.)
#define KdClCa (GlobalData_t)(0.1)
#define Khp (GlobalData_t)(1.698e-7)
#define Kki (GlobalData_t)(0.5)
#define Kko (GlobalData_t)(0.3582)
#define KmBSL (GlobalData_t)(0.0087)
#define KmBSR (GlobalData_t)(0.00087)
#define KmCaAct (GlobalData_t)(150e-6)
#define KmCaM (GlobalData_t)(0.0015)
#define KmCaMK (GlobalData_t)(0.15)
#define KmCap (GlobalData_t)(0.0005)
#define Kmgatp (GlobalData_t)(1.698e-7)
#define Kmn (GlobalData_t)(0.002)
#define Knai0 (GlobalData_t)(9.073)
#define Knao0 (GlobalData_t)(27.78)
#define Knap (GlobalData_t)(224.)
#define Kxkur (GlobalData_t)(292.)
#define L (GlobalData_t)(0.01)
#define MgADP (GlobalData_t)(0.05)
#define MgATP (GlobalData_t)(9.8)
#define O_init (GlobalData_t)(0.00027561)
#define PCa_b (GlobalData_t)(8.3757e-05)
#define PCab (GlobalData_t)(5.9194e-08)
#define PKNa (GlobalData_t)(0.01833)
#define PNab (GlobalData_t)(1.9239e-09)
#define Pnak_b (GlobalData_t)(15.4509)
#define R (GlobalData_t)(8314.)
#define T (GlobalData_t)(310.)
#define aCaMK (GlobalData_t)(0.05)
#define a_init (GlobalData_t)(0.00092716)
#define alpha_1 (GlobalData_t)(0.154375)
#define ap_init (GlobalData_t)(0.0004724)
#define bCaMK (GlobalData_t)(0.00068)
#define beta_1 (GlobalData_t)(0.1911)
#define bt (GlobalData_t)(4.75)
#define cai_init (GlobalData_t)(6.6309e-05)
#define cajsr_half (GlobalData_t)(1.7)
#define cajsr_init (GlobalData_t)(1.8102)
#define cansr_init (GlobalData_t)(1.8119)
#define cao (GlobalData_t)(1.8)
#define cass_init (GlobalData_t)(5.7672e-05)
#define celltype (GlobalData_t)(1.)
#define cli (GlobalData_t)(24.0)
#define clo (GlobalData_t)(150.0)
#define cmdnmax_b (GlobalData_t)(0.05)
#define csqnmax (GlobalData_t)(10.)
#define d_init (GlobalData_t)(0.)
#define delta (GlobalData_t)(-0.155)
#define dielConstant (GlobalData_t)(74.)
#define eP (GlobalData_t)(4.2)
#define fcaf_init (GlobalData_t)(1.0)
#define fcafp_init (GlobalData_t)(1.0)
#define fcas_init (GlobalData_t)(0.9999)
#define ff_init (GlobalData_t)(1.0)
#define ffp_init (GlobalData_t)(1.0)
#define fkatp (GlobalData_t)(0.0)
#define fs_init (GlobalData_t)(0.9485)
#define gkatp (GlobalData_t)(4.3195)
#define hL_init (GlobalData_t)(0.5401)
#define hLp_init (GlobalData_t)(0.3034)
#define h_init (GlobalData_t)(0.8360)
#define hp_init (GlobalData_t)(0.6828)
#define iF_init (GlobalData_t)(0.9996)
#define iFp_init (GlobalData_t)(0.9996)
#define iS_init (GlobalData_t)(0.9996)
#define iSp_init (GlobalData_t)(0.9996)
#define i_Stim_Amplitude (GlobalData_t)(-53.)
#define i_Stim_End (GlobalData_t)(100000000000000000.)
#define i_Stim_Period (GlobalData_t)(1000.)
#define i_Stim_PulseDuration (GlobalData_t)(1.0)
#define i_Stim_Start (GlobalData_t)(0.)
#define j_init (GlobalData_t)(0.8359)
#define jca_init (GlobalData_t)(1.0)
#define jp_init (GlobalData_t)(0.8357)
#define k1m (GlobalData_t)(182.4)
#define k1p (GlobalData_t)(949.5)
#define k2m (GlobalData_t)(39.4)
#define k2n (GlobalData_t)(500.)
#define k2p (GlobalData_t)(687.2)
#define k3m (GlobalData_t)(79300.)
#define k3p (GlobalData_t)(1899.)
#define k4m (GlobalData_t)(40.)
#define k4p (GlobalData_t)(639.)
#define kasymm (GlobalData_t)(12.5)
#define kcaoff (GlobalData_t)(5e3)
#define kcaon (GlobalData_t)(1.5e6)
#define ki_init (GlobalData_t)(142.6951)
#define kmcmdn (GlobalData_t)(0.00238)
#define kmcsqn (GlobalData_t)(0.8)
#define kmtrpn (GlobalData_t)(0.0005)
#define kna1 (GlobalData_t)(15.)
#define kna2 (GlobalData_t)(5.)
#define kna3 (GlobalData_t)(88.12)
#define ko (GlobalData_t)(5.0)
#define kss_init (GlobalData_t)(142.6951)
#define mL_init (GlobalData_t)(0.00015166)
#define m_init (GlobalData_t)(0.00074303)
#define nai_init (GlobalData_t)(12.8363)
#define nao (GlobalData_t)(140.0)
#define nass_init (GlobalData_t)(12.8366)
#define nca_i_init (GlobalData_t)(0.00053006)
#define nca_ss_init (GlobalData_t)(0.00030853)
#define offset (GlobalData_t)(0.)
#define qca (GlobalData_t)(0.167)
#define qna (GlobalData_t)(0.5224)
#define rad (GlobalData_t)(0.0011)
#define tauCa (GlobalData_t)(0.2)
#define tauK (GlobalData_t)(2.0)
#define tauNa (GlobalData_t)(2.0)
#define thL (GlobalData_t)(200.)
#define tjca (GlobalData_t)(75.)
#define trpnmax (GlobalData_t)(0.07)
#define vShift (GlobalData_t)(0.)
#define v_init (GlobalData_t)(-89.1400)
#define wca (GlobalData_t)(6e4)
#define wna (GlobalData_t)(6e4)
#define wnaca (GlobalData_t)(5e3)
#define xs1_init (GlobalData_t)(0.2309)
#define xs2_init (GlobalData_t)(0.00016975)
#define zca (GlobalData_t)(2.)
#define zcl (GlobalData_t)(-1.)
#define zk (GlobalData_t)(1.)
#define zna (GlobalData_t)(1.)
#define Afs (GlobalData_t)((1.-(Aff)))
#define Ageo (GlobalData_t)(((((2.*3.14)*rad)*rad)+(((2.*3.14)*rad)*L)))
#define ECl (GlobalData_t)((((R*T)/(zcl*F))*(log((clo/cli)))))
#define GK1 (GlobalData_t)(((celltype==1.) ? (GK1_b*1.2) : ((celltype==2.) ? (GK1_b*1.3) : GK1_b)))
#define GKb (GlobalData_t)(((celltype==1.) ? (GKb_b*0.6) : GKb_b))
#define GKr (GlobalData_t)(((celltype==1.) ? (GKr_b*1.3) : ((celltype==2.) ? (GKr_b*0.8) : GKr_b)))
#define GKs (GlobalData_t)(((celltype==1.) ? (GKs_b*1.4) : GKs_b))
#define GNaL (GlobalData_t)(((celltype==1.) ? (GNaL_b*0.6) : GNaL_b))
#define Gncx (GlobalData_t)(((celltype==1.) ? (Gncx_b*1.1) : ((celltype==2.) ? (Gncx_b*1.4) : Gncx_b)))
#define Gto (GlobalData_t)(((celltype==1.) ? (Gto_b*2.) : ((celltype==2.) ? (Gto_b*2.) : Gto_b)))
#define Io (GlobalData_t)(((0.5*(((nao+ko)+clo)+(4.*cao)))/1000.))
#define PCa (GlobalData_t)(((celltype==1.) ? (PCa_b*1.2) : ((celltype==2.) ? (PCa_b*2.) : PCa_b)))
#define Pnak (GlobalData_t)(((celltype==1.) ? (Pnak_b*0.9) : ((celltype==2.) ? (Pnak_b*0.7) : Pnak_b)))
#define a2 (GlobalData_t)(k2p)
#define a4 (GlobalData_t)((((k4p*MgATP)/Kmgatp)/(1.+(MgATP/Kmgatp))))
#define a_rel (GlobalData_t)((0.5*bt))
#define akik (GlobalData_t)((pow((ko/K_o_n),0.24)))
#define b1 (GlobalData_t)((k1m*MgADP))
#define bkik (GlobalData_t)((1./(1.+((A_atp/K_atp)*(A_atp/K_atp)))))
#define btp (GlobalData_t)((1.25*bt))
#define cmdnmax (GlobalData_t)(((celltype==1.) ? (cmdnmax_b*1.3) : cmdnmax_b))
#define constA (GlobalData_t)((1.82e6*(pow((dielConstant*T),-1.5))))
#define h10_i (GlobalData_t)(((kasymm+1.)+((nao/kna1)*(1.+(nao/kna2)))))
#define h10_ss (GlobalData_t)(((kasymm+1.)+((nao/kna1)*(1.+(nao/kna2)))))
#define k2_i (GlobalData_t)(kcaoff)
#define k2_ss (GlobalData_t)(kcaoff)
#define k5_i (GlobalData_t)(kcaoff)
#define k5_ss (GlobalData_t)(kcaoff)
#define thLp (GlobalData_t)((3.*thL))
#define upScale (GlobalData_t)(((celltype==1.) ? 1.3 : 1.))
#define vcell (GlobalData_t)(((((1000.*3.14)*rad)*rad)*L))
#define Acap (GlobalData_t)((2.*Ageo))
#define PCaK (GlobalData_t)((3.574e-4*PCa))
#define PCaNa (GlobalData_t)((0.00125*PCa))
#define PCap (GlobalData_t)((1.1*PCa))
#define a_relp (GlobalData_t)((0.5*btp))
#define gamma_cao (GlobalData_t)((exp((((-constA)*4.)*(((sqrt(Io))/(1.+(sqrt(Io))))-((0.3*Io)))))))
#define gamma_ko (GlobalData_t)((exp(((-constA)*(((sqrt(Io))/(1.+(sqrt(Io))))-((0.3*Io)))))))
#define gamma_nao (GlobalData_t)((exp(((-constA)*(((sqrt(Io))/(1.+(sqrt(Io))))-((0.3*Io)))))))
#define h11_i (GlobalData_t)(((nao*nao)/((h10_i*kna1)*kna2)))
#define h11_ss (GlobalData_t)(((nao*nao)/((h10_ss*kna1)*kna2)))
#define h12_i (GlobalData_t)((1./h10_i))
#define h12_ss (GlobalData_t)((1./h10_ss))
#define vjsr (GlobalData_t)((0.0048*vcell))
#define vmyo (GlobalData_t)((0.68*vcell))
#define vnsr (GlobalData_t)((0.0552*vcell))
#define vss (GlobalData_t)((0.02*vcell))
#define PCaKp (GlobalData_t)((3.574e-4*PCap))
#define PCaNap (GlobalData_t)((0.00125*PCap))
#define k1_i (GlobalData_t)(((h12_i*cao)*kcaon))
#define k1_ss (GlobalData_t)(((h12_ss*cao)*kcaon))



void ToRORd_fkatp_epiIonType::initialize_params(IonIfBase& imp_base) const
{
  IonIfDerived& imp = static_cast<IonIfDerived&>(imp_base);
  cell_geom* region = &imp.cgeom();
  ToRORd_fkatp_epi_Params *p = imp.params();

  // Compute the regional constants
  {

#ifndef USE_CVODE
        //log_msg(NULL,5,0,"ToRORd_fkatp_epi needs CVODE. Recompilation needed.");
        exit(1);
#endif
  }
  // Compute the regional initialization
  {
  }

}


// Define the parameters for the lookup tables
enum Tables {

  N_TABS
};

// Define the indices into the lookup tables.

    enum Rosenbrock {
    

      N_ROSEN
    };


void ToRORd_fkatp_epiIonType::construct_tables(IonIfBase& imp_base) const
{
  IonIfDerived& imp = static_cast<IonIfDerived&>(imp_base);
  GlobalData_t dt = imp.get_dt() * 1e0;
  cell_geom* region = &imp.cgeom();
  ToRORd_fkatp_epi_Params *p = imp.params();

  imp.tables().resize(N_TABS);

  // Define the constants that depend on the parameters.

}

enum Cvode_Vars {
  CVODE_C1,
  CVODE_C2,
  CVODE_C3,
  CVODE_CaMKt,
  CVODE_I,
  CVODE_Jrel_np,
  CVODE_Jrel_p,
  CVODE_O,
  CVODE_a,
  CVODE_ap,
  CVODE_cai,
  CVODE_cajsr,
  CVODE_cansr,
  CVODE_cass,
  CVODE_d,
  CVODE_fcaf,
  CVODE_fcafp,
  CVODE_fcas,
  CVODE_ff,
  CVODE_ffp,
  CVODE_fs,
  CVODE_h,
  CVODE_hL,
  CVODE_hLp,
  CVODE_hp,
  CVODE_iF,
  CVODE_iFp,
  CVODE_iS,
  CVODE_iSp,
  CVODE_j,
  CVODE_jca,
  CVODE_jp,
  CVODE_ki,
  CVODE_kss,
  CVODE_m,
  CVODE_mL,
  CVODE_nai,
  CVODE_nass,
  CVODE_nca_i,
  CVODE_nca_ss,
  CVODE_xs1,
  CVODE_xs2,

  N_CVODE
};

#ifdef USE_CVODE
#if SUNDIALS_VERSION_MAJOR >= 7
int ToRORd_fkatp_epi_internal_rhs_function(sunrealtype t, N_Vector CVODE, N_Vector CVODE_dot, void* user_data);
#else
int ToRORd_fkatp_epi_internal_rhs_function(realtype t, N_Vector CVODE, N_Vector CVODE_dot, void* user_data);
#endif
#endif

#ifndef LIMPET_CVODE_RTOL
#define LIMPET_CVODE_RTOL 1e-5
#endif

//disable absolute tolerances for all gates.
#ifndef LIMPET_CVODE_ATOL
#define LIMPET_CVODE_ATOL 1e-16
#endif




void ToRORd_fkatp_epiIonType::initialize_sv(IonIfBase& imp_base, GlobalData_t **impdata ) const
{
  IonIfDerived& imp = static_cast<IonIfDerived&>(imp_base);
  GlobalData_t dt = imp.get_dt() * 1e0;
  cell_geom *region = &imp.cgeom();
  ToRORd_fkatp_epi_Params *p = imp.params();

  ToRORd_fkatp_epi_state *sv_base = (ToRORd_fkatp_epi_state *)imp.sv_tab().data();
  GlobalData_t t = 0;

  IonIfDerived* IF = &imp;
  // Define the constants that depend on the parameters.
  //Prepare all the public arrays.
  GlobalData_t *Iion_ext = impdata[Iion];
  GlobalData_t *v_ext = impdata[Vm];
  //Prepare all the private functions.

  //set the initial values
  for(int __i=0; __i < imp.get_num_node(); __i+=1 ){
    ToRORd_fkatp_epi_state *sv = sv_base+__i / 1;
    //Initialize the external vars to their current values
    GlobalData_t Iion = Iion_ext[__i];
    GlobalData_t v = v_ext[__i];
    //Change the units of external variables as appropriate.
    
    
    // Initialize the rest of the nodal variables
    sv->C1 = C1_init;
    sv->C2 = C2_init;
    sv->C3 = C3_init;
    sv->CaMKt = CaMKt_init;
    sv->I = I_init;
    sv->Jrel_np = Jrel_np_init;
    sv->Jrel_p = Jrel_p_init;
    sv->O = O_init;
    sv->a = a_init;
    sv->ap = ap_init;
    sv->cai = cai_init;
    sv->cajsr = cajsr_init;
    sv->cansr = cansr_init;
    sv->cass = cass_init;
    sv->d = d_init;
    sv->fcaf = fcaf_init;
    sv->fcafp = fcafp_init;
    sv->fcas = fcas_init;
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
    sv->jca = jca_init;
    sv->jp = jp_init;
    sv->ki = ki_init;
    sv->kss = kss_init;
    sv->m = m_init;
    sv->mL = mL_init;
    sv->nai = nai_init;
    sv->nass = nass_init;
    sv->nca_i = nca_i_init;
    sv->nca_ss = nca_ss_init;
    v = v_init;
    sv->xs1 = xs1_init;
    sv->xs2 = xs2_init;
    double Afcaf = (0.3+(0.6/(1.+(exp(((v-(10.))/10.))))));
    double AiF = (1./(1.+(exp((((v+EKshift)-(213.6))/151.2)))));
    double CaMKb = ((CaMKo*(1.-(sv->CaMKt)))/(1.+(KmCaM/sv->cass)));
    double EK = (((R*T)/(zk*F))*(log((ko/sv->ki))));
    double EKs = (((R*T)/(zk*F))*(log(((ko+(PKNa*nao))/(sv->ki+(PKNa*sv->nai))))));
    double ENa = (((R*T)/(zna*F))*(log((nao/sv->nai))));
    double IClCa_junc = (((Fjunc*GClCa)/(1.+(KdClCa/sv->cass)))*(v-(ECl)));
    double IClCa_sl = ((((1.-(Fjunc))*GClCa)/(1.+(KdClCa/sv->cai)))*(v-(ECl)));
    double IClb = (GClb*(v-(ECl)));
    double Ii = ((0.5*(((sv->nai+sv->ki)+cli)+(4.*sv->cai)))/1000.);
    double IpCa = ((GpCa*sv->cai)/(KmCap+sv->cai));
    double Iss = ((0.5*(((sv->nass+sv->kss)+cli)+(4.*sv->cass)))/1000.);
    double KsCa = (1.+(0.6/(1.+(pow((3.8e-5/sv->cai),1.4)))));
    double P = (eP/(((1.+(H/Khp))+(sv->nai/Knap))+(sv->ki/Kxkur)));
    double allo_i = (1./(1.+((KmCaAct/sv->cai)*(KmCaAct/sv->cai))));
    double allo_ss = (1./(1.+((KmCaAct/sv->cass)*(KmCaAct/sv->cass))));
    double f = ((Aff*sv->ff)+(Afs*sv->fs));
    double fp = ((Aff*sv->ffp)+(Afs*sv->fs));
    double h4_i = (1.+((sv->nai/kna1)*(1.+(sv->nai/kna2))));
    double h4_ss = (1.+((sv->nass/kna1)*(1.+(sv->nass/kna2))));
    double vffrt = (((v*F)*F)/(R*T));
    double vfrt = ((v*F)/(R*T));
    double xkb = (1./(1.+(exp(((-(v-(10.8968)))/23.9871)))));
    double Afcas = (1.-(Afcaf));
    double AiS = (1.-(AiF));
    double CaMKa = (CaMKb+sv->CaMKt);
    double IClCa = (IClCa_junc+IClCa_sl);
    double IKb = ((GKb*xkb)*(v-(EK)));
    double IKr = (((GKr*(sqrt((ko/5.))))*sv->O)*(v-(EK)));
    double IKs = ((((GKs*KsCa)*sv->xs1)*sv->xs2)*(v-(EKs)));
    double INab = (((PNab*vffrt)*((sv->nai*(exp(vfrt)))-(nao)))/((exp(vfrt))-(1.)));
    double I_katp = ((((fkatp*gkatp)*akik)*bkik)*(v-(EK)));
    double Knai = (Knai0*(exp(((delta*vfrt)/3.))));
    double Knao = (Knao0*(exp((((1.-(delta))*vfrt)/3.))));
    double aK1 = (4.094/(1.+(exp((0.1217*((v-(EK))-(49.934)))))));
    double b3 = (((k3m*P)*H)/(1.+(MgATP/Kmgatp)));
    double bK1 = (((15.72*(exp((0.0674*((v-(EK))-(3.257))))))+(exp((0.0618*((v-(EK))-(594.31))))))/(1.+(exp((-0.1629*((v-(EK))+14.207))))));
    double gamma_cai = (exp((((-constA)*4.)*(((sqrt(Ii))/(1.+(sqrt(Ii))))-((0.3*Ii))))));
    double gamma_cass = (exp((((-constA)*4.)*(((sqrt(Iss))/(1.+(sqrt(Iss))))-((0.3*Iss))))));
    double gamma_ki = (exp(((-constA)*(((sqrt(Ii))/(1.+(sqrt(Ii))))-((0.3*Ii))))));
    double gamma_kss = (exp(((-constA)*(((sqrt(Iss))/(1.+(sqrt(Iss))))-((0.3*Iss))))));
    double gamma_nai = (exp(((-constA)*(((sqrt(Ii))/(1.+(sqrt(Ii))))-((0.3*Ii))))));
    double gamma_nass = (exp(((-constA)*(((sqrt(Iss))/(1.+(sqrt(Iss))))-((0.3*Iss))))));
    double h5_i = ((sv->nai*sv->nai)/((h4_i*kna1)*kna2));
    double h5_ss = ((sv->nass*sv->nass)/((h4_ss*kna1)*kna2));
    double h6_i = (1./h4_i);
    double h6_ss = (1./h4_ss);
    double hca = (exp((qca*vfrt)));
    double hna = (exp((qna*vfrt)));
    double ICab = ((((PCab*4.)*vffrt)*(((gamma_cai*sv->cai)*(exp((2.*vfrt))))-((gamma_cao*cao))))/((exp((2.*vfrt)))-(1.)));
    double K1ss = (aK1/(aK1+bK1));
    double PhiCaK_i = ((vffrt*(((gamma_ki*sv->ki)*(exp(vfrt)))-((gamma_ko*ko))))/((exp(vfrt))-(1.)));
    double PhiCaK_ss = ((vffrt*(((gamma_kss*sv->kss)*(exp(vfrt)))-((gamma_ko*ko))))/((exp(vfrt))-(1.)));
    double PhiCaL_i = (((4.*vffrt)*(((gamma_cai*sv->cai)*(exp((2.*vfrt))))-((gamma_cao*cao))))/((exp((2.*vfrt)))-(1.)));
    double PhiCaL_ss = (((4.*vffrt)*(((gamma_cass*sv->cass)*(exp((2.*vfrt))))-((gamma_cao*cao))))/((exp((2.*vfrt)))-(1.)));
    double PhiCaNa_i = ((vffrt*(((gamma_nai*sv->nai)*(exp(vfrt)))-((gamma_nao*nao))))/((exp(vfrt))-(1.)));
    double PhiCaNa_ss = ((vffrt*(((gamma_nass*sv->nass)*(exp(vfrt)))-((gamma_nao*nao))))/((exp(vfrt))-(1.)));
    double a1 = ((k1p*(((sv->nai/Knai)*(sv->nai/Knai))*(sv->nai/Knai)))/(((((1.+(sv->nai/Knai))*(1.+(sv->nai/Knai)))*(1.+(sv->nai/Knai)))+((1.+(sv->ki/Kki))*(1.+(sv->ki/Kki))))-(1.)));
    double a3 = ((k3p*((ko/Kko)*(ko/Kko)))/(((((1.+(nao/Knao))*(1.+(nao/Knao)))*(1.+(nao/Knao)))+((1.+(ko/Kko))*(1.+(ko/Kko))))-(1.)));
    double b2 = ((k2m*(((nao/Knao)*(nao/Knao))*(nao/Knao)))/(((((1.+(nao/Knao))*(1.+(nao/Knao)))*(1.+(nao/Knao)))+((1.+(ko/Kko))*(1.+(ko/Kko))))-(1.)));
    double b4 = ((k4m*((sv->ki/Kki)*(sv->ki/Kki)))/(((((1.+(sv->nai/Knai))*(1.+(sv->nai/Knai)))*(1.+(sv->nai/Knai)))+((1.+(sv->ki/Kki))*(1.+(sv->ki/Kki))))-(1.)));
    double fICaLp = (1./(1.+(KmCaMK/CaMKa)));
    double fINaLp = (1./(1.+(KmCaMK/CaMKa)));
    double fINap = (1./(1.+(KmCaMK/CaMKa)));
    double fItop = (1./(1.+(KmCaMK/CaMKa)));
    double fca = ((Afcaf*sv->fcaf)+(Afcas*sv->fcas));
    double fcap = ((Afcaf*sv->fcafp)+(Afcas*sv->fcas));
    double h1_i = (1.+((sv->nai/kna3)*(1.+hna)));
    double h1_ss = (1.+((sv->nass/kna3)*(1.+hna)));
    double h7_i = (1.+((nao/kna3)*(1.+(1./hna))));
    double h7_ss = (1.+((nao/kna3)*(1.+(1./hna))));
    double i_gate = ((AiF*sv->iF)+(AiS*sv->iS));
    double ip = ((AiF*sv->iFp)+(AiS*sv->iSp));
    double k6_i = ((h6_i*sv->cai)*kcaon);
    double k6_ss = ((h6_ss*sv->cass)*kcaon);
    double ICaK_i = ((1.-(ICaL_fractionSS))*((((((1.-(fICaLp))*PCaK)*PhiCaK_i)*sv->d)*((f*(1.-(sv->nca_i)))+((sv->jca*fca)*sv->nca_i)))+((((fICaLp*PCaKp)*PhiCaK_i)*sv->d)*((fp*(1.-(sv->nca_i)))+((sv->jca*fcap)*sv->nca_i)))));
    double ICaK_ss = (ICaL_fractionSS*((((((1.-(fICaLp))*PCaK)*PhiCaK_ss)*sv->d)*((f*(1.-(sv->nca_ss)))+((sv->jca*fca)*sv->nca_ss)))+((((fICaLp*PCaKp)*PhiCaK_ss)*sv->d)*((fp*(1.-(sv->nca_ss)))+((sv->jca*fcap)*sv->nca_ss)))));
    double ICaL_i = ((1.-(ICaL_fractionSS))*((((((1.-(fICaLp))*PCa)*PhiCaL_i)*sv->d)*((f*(1.-(sv->nca_i)))+((sv->jca*fca)*sv->nca_i)))+((((fICaLp*PCap)*PhiCaL_i)*sv->d)*((fp*(1.-(sv->nca_i)))+((sv->jca*fcap)*sv->nca_i)))));
    double ICaL_ss = (ICaL_fractionSS*((((((1.-(fICaLp))*PCa)*PhiCaL_ss)*sv->d)*((f*(1.-(sv->nca_ss)))+((sv->jca*fca)*sv->nca_ss)))+((((fICaLp*PCap)*PhiCaL_ss)*sv->d)*((fp*(1.-(sv->nca_ss)))+((sv->jca*fcap)*sv->nca_ss)))));
    double ICaNa_i = ((1.-(ICaL_fractionSS))*((((((1.-(fICaLp))*PCaNa)*PhiCaNa_i)*sv->d)*((f*(1.-(sv->nca_i)))+((sv->jca*fca)*sv->nca_i)))+((((fICaLp*PCaNap)*PhiCaNa_i)*sv->d)*((fp*(1.-(sv->nca_i)))+((sv->jca*fcap)*sv->nca_i)))));
    double ICaNa_ss = (ICaL_fractionSS*((((((1.-(fICaLp))*PCaNa)*PhiCaNa_ss)*sv->d)*((f*(1.-(sv->nca_ss)))+((sv->jca*fca)*sv->nca_ss)))+((((fICaLp*PCaNap)*PhiCaNa_ss)*sv->d)*((fp*(1.-(sv->nca_ss)))+((sv->jca*fcap)*sv->nca_ss)))));
    double IK1 = (((GK1*(sqrt((ko/5.))))*K1ss)*(v-(EK)));
    double INa = (((GNa*(v-(ENa)))*((sv->m*sv->m)*sv->m))*((((1.-(fINap))*sv->h)*sv->j)+((fINap*sv->hp)*sv->jp)));
    double INaL = (((GNaL*(v-(ENa)))*sv->mL)*(((1.-(fINaLp))*sv->hL)+(fINaLp*sv->hLp)));
    double Ito = ((Gto*(v-(EK)))*((((1.-(fItop))*sv->a)*i_gate)+((fItop*sv->ap)*ip)));
    double h2_i = ((sv->nai*hna)/(kna3*h1_i));
    double h2_ss = ((sv->nass*hna)/(kna3*h1_ss));
    double h3_i = (1./h1_i);
    double h3_ss = (1./h1_ss);
    double h8_i = (nao/((kna3*hna)*h7_i));
    double h8_ss = (nao/((kna3*hna)*h7_ss));
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
    double k3p_i = (h9_i*wca);
    double k3p_ss = (h9_ss*wca);
    double k3pp_i = (h8_i*wnaca);
    double k3pp_ss = (h8_ss*wnaca);
    double k4p_i = ((h3_i*wca)/hca);
    double k4p_ss = ((h3_ss*wca)/hca);
    double k4pp_i = (h2_i*wnaca);
    double k4pp_ss = (h2_ss*wnaca);
    double k7_i = ((h5_i*h2_i)*wna);
    double k7_ss = ((h5_ss*h2_ss)*wna);
    double k8_i = ((h8_i*h11_i)*wna);
    double k8_ss = ((h8_ss*h11_ss)*wna);
    double JnakK = (2.*((E4*b1)-((E3*a1))));
    double JnakNa = (3.*((E1*a3)-((E2*b3))));
    double k3_i = (k3p_i+k3pp_i);
    double k3_ss = (k3p_ss+k3pp_ss);
    double k4_i = (k4p_i+k4pp_i);
    double k4_ss = (k4p_ss+k4pp_ss);
    double INaK = (Pnak*((zna*JnakNa)+(zk*JnakK)));
    double x1_i = (((k2_i*k4_i)*(k7_i+k6_i))+((k5_i*k7_i)*(k2_i+k3_i)));
    double x1_ss = (((k2_ss*k4_ss)*(k7_ss+k6_ss))+((k5_ss*k7_ss)*(k2_ss+k3_ss)));
    double x2_i = (((k1_i*k7_i)*(k4_i+k5_i))+((k4_i*k6_i)*(k1_i+k8_i)));
    double x2_ss = (((k1_ss*k7_ss)*(k4_ss+k5_ss))+((k4_ss*k6_ss)*(k1_ss+k8_ss)));
    double x3_i = (((k1_i*k3_i)*(k7_i+k6_i))+((k8_i*k6_i)*(k2_i+k3_i)));
    double x3_ss = (((k1_ss*k3_ss)*(k7_ss+k6_ss))+((k8_ss*k6_ss)*(k2_ss+k3_ss)));
    double x4_i = (((k2_i*k8_i)*(k4_i+k5_i))+((k3_i*k5_i)*(k1_i+k8_i)));
    double x4_ss = (((k2_ss*k8_ss)*(k4_ss+k5_ss))+((k3_ss*k5_ss)*(k1_ss+k8_ss)));
    double E1_i = (x1_i/(((x1_i+x2_i)+x3_i)+x4_i));
    double E1_ss = (x1_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
    double E2_i = (x2_i/(((x1_i+x2_i)+x3_i)+x4_i));
    double E2_ss = (x2_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
    double E3_i = (x3_i/(((x1_i+x2_i)+x3_i)+x4_i));
    double E3_ss = (x3_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
    double E4_i = (x4_i/(((x1_i+x2_i)+x3_i)+x4_i));
    double E4_ss = (x4_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
    double JncxCa_i = ((E2_i*k2_i)-((E1_i*k1_i)));
    double JncxCa_ss = ((E2_ss*k2_ss)-((E1_ss*k1_ss)));
    double JncxNa_i = (((3.*((E4_i*k7_i)-((E1_i*k8_i))))+(E3_i*k4pp_i))-((E2_i*k3pp_i)));
    double JncxNa_ss = (((3.*((E4_ss*k7_ss)-((E1_ss*k8_ss))))+(E3_ss*k4pp_ss))-((E2_ss*k3pp_ss)));
    double INaCa_i = ((((1.-(INaCa_fractionSS))*Gncx)*allo_i)*((zna*JncxNa_i)+(zca*JncxCa_i)));
    double INaCa_ss = (((INaCa_fractionSS*Gncx)*allo_ss)*((zna*JncxNa_ss)+(zca*JncxCa_ss)));
    Iion = (-(-((((((((((((((((((INa+INaL)+Ito)+ICaL)+ICaNa)+ICaK)+IKr)+IKs)+IK1)+INaCa_i)+INaCa_ss)+INaK)+INab)+IKb)+IpCa)+ICab)+IClCa)+IClb)+I_katp)));
    //Change the units of external variables as appropriate.
    
    
    //Save all external vars
    Iion_ext[__i] = Iion;
    v_ext[__i] = v;
  }

  //setup CVODE
  {
#ifdef USE_CVODE
#if SUNDIALS_VERSION_MAJOR >= 6
    SUNContext sunctx;
    SUNContext_Create(SUN_COMM_NULL, &sunctx);
    void* cvode_mem = CVodeCreate(CV_BDF, sunctx);
#elif SUNDIALS_VERSION_MAJOR >= 4
    void* cvode_mem = CVodeCreate(CV_BDF);
#else
    void* cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
#endif
    assert(cvode_mem != NULL);
#if SUNDIALS_VERSION_MAJOR >= 6
    N_Vector cvode_y = N_VNew_Serial(N_CVODE, sunctx);
#else
    N_Vector cvode_y = N_VNew_Serial(N_CVODE);
#endif
    int flag = CVodeInit(cvode_mem, ToRORd_fkatp_epi_internal_rhs_function, 0, cvode_y);
    assert(flag == CV_SUCCESS);
    N_VDestroy(cvode_y);

    flag = CVodeSStolerances(cvode_mem, LIMPET_CVODE_RTOL, LIMPET_CVODE_ATOL);
    assert(flag == CV_SUCCESS);
    flag = CVodeSetMaxStep(cvode_mem, dt);
    assert(flag == CV_SUCCESS);
    ToRORd_fkatp_epi_Private_cpu* userdata = (ToRORd_fkatp_epi_Private_cpu*)(imp.ion_private().data());
    userdata->cvode_mem = cvode_mem;
#if SUNDIALS_VERSION_MAJOR >= 6
    userdata->sunctx_ptr = malloc(sizeof(SUNContext));
    memcpy(userdata->sunctx_ptr, &sunctx, sizeof(SUNContext));
#endif
    flag = CVodeSetUserData(cvode_mem, userdata);
    assert(flag == CV_SUCCESS);
    flag = CVDiag(cvode_mem);
    assert(flag == CV_SUCCESS);
#endif
  }

}

/** compute the  current
 *
 * param start   index of first node
 * param end     index of last node
 * param IF      IMP
 * param plgdata external data needed by IMP
 */
#ifdef TORORD_FKATP_EPI_CPU_GENERATED
extern "C" {
void compute_ToRORd_fkatp_epi_cpu(int start, int end, IonIfBase& imp_base, GlobalData_t **impdata )
{
  ToRORd_fkatp_epiIonType::IonIfDerived& imp = static_cast<ToRORd_fkatp_epiIonType::IonIfDerived&>(imp_base);
  GlobalData_t dt = imp.get_dt()*1e0;
  cell_geom *region = &imp.cgeom();
  ToRORd_fkatp_epi_Params *p  = imp.params();
  ToRORd_fkatp_epi_state *sv_base = (ToRORd_fkatp_epi_state *)imp.sv_tab().data();

  GlobalData_t t = imp.get_tstp().cnt*dt;

  ToRORd_fkatp_epiIonType::IonIfDerived* IF = &imp;

  // Define the constants that depend on the parameters.
  
  //Initializing the userdata structures.
  ToRORd_fkatp_epiIonType::private_type* ion_private = imp.ion_private().data();
  int nthread = 1;
  #ifdef _OPENMP
  nthread = omp_get_max_threads();
  #endif
  for( int j=0; j<nthread; j++ ) {
  }
  
  #ifdef USE_CVODE
  #if SUNDIALS_VERSION_MAJOR >= 7
  sunrealtype CVODE_array[N_CVODE];
  N_Vector CVODE = N_VMake_Serial(N_CVODE, CVODE_array, *((SUNContext*)((ToRORd_fkatp_epiIonType::private_type*)(imp.ion_private().data()))->sunctx_ptr));
  #else
  realtype CVODE_array[N_CVODE];
  N_Vector CVODE = N_VMake_Serial(N_CVODE, CVODE_array);
  #endif
  #endif
  void* cvode_mem = ((ToRORd_fkatp_epiIonType::private_type*)(imp.ion_private().data()))->cvode_mem;
                      
  //Prepare all the public arrays.
  GlobalData_t *Iion_ext = impdata[Iion];
  GlobalData_t *v_ext = impdata[Vm];
  //Prepare all the private functions.

#pragma omp parallel for schedule(static)
  for (int __i=(start / 1) * 1; __i<end; __i+=1) {
    ToRORd_fkatp_epi_state *sv = sv_base+__i / 1;
                    
    //Initialize the external vars to their current values
    GlobalData_t Iion = Iion_ext[__i];
    GlobalData_t v = v_ext[__i];
    //Change the units of external variables as appropriate.
    
    
    //Compute lookup tables for things that have already been defined.
    
    
    //Compute storevars and external modvars
    GlobalData_t Afcaf = (0.3+(0.6/(1.+(exp(((v-(10.))/10.))))));
    GlobalData_t AiF = (1./(1.+(exp((((v+EKshift)-(213.6))/151.2)))));
    GlobalData_t CaMKb = ((CaMKo*(1.-(sv->CaMKt)))/(1.+(KmCaM/sv->cass)));
    GlobalData_t EK = (((R*T)/(zk*F))*(log((ko/sv->ki))));
    GlobalData_t EKs = (((R*T)/(zk*F))*(log(((ko+(PKNa*nao))/(sv->ki+(PKNa*sv->nai))))));
    GlobalData_t ENa = (((R*T)/(zna*F))*(log((nao/sv->nai))));
    GlobalData_t IClCa_junc = (((Fjunc*GClCa)/(1.+(KdClCa/sv->cass)))*(v-(ECl)));
    GlobalData_t IClCa_sl = ((((1.-(Fjunc))*GClCa)/(1.+(KdClCa/sv->cai)))*(v-(ECl)));
    GlobalData_t IClb = (GClb*(v-(ECl)));
    GlobalData_t Ii = ((0.5*(((sv->nai+sv->ki)+cli)+(4.*sv->cai)))/1000.);
    GlobalData_t IpCa = ((GpCa*sv->cai)/(KmCap+sv->cai));
    GlobalData_t Iss = ((0.5*(((sv->nass+sv->kss)+cli)+(4.*sv->cass)))/1000.);
    GlobalData_t KsCa = (1.+(0.6/(1.+(pow((3.8e-5/sv->cai),1.4)))));
    GlobalData_t P = (eP/(((1.+(H/Khp))+(sv->nai/Knap))+(sv->ki/Kxkur)));
    GlobalData_t allo_i = (1./(1.+((KmCaAct/sv->cai)*(KmCaAct/sv->cai))));
    GlobalData_t allo_ss = (1./(1.+((KmCaAct/sv->cass)*(KmCaAct/sv->cass))));
    GlobalData_t f = ((Aff*sv->ff)+(Afs*sv->fs));
    GlobalData_t fp = ((Aff*sv->ffp)+(Afs*sv->fs));
    GlobalData_t h4_i = (1.+((sv->nai/kna1)*(1.+(sv->nai/kna2))));
    GlobalData_t h4_ss = (1.+((sv->nass/kna1)*(1.+(sv->nass/kna2))));
    GlobalData_t vffrt = (((v*F)*F)/(R*T));
    GlobalData_t vfrt = ((v*F)/(R*T));
    GlobalData_t xkb = (1./(1.+(exp(((-(v-(10.8968)))/23.9871)))));
    GlobalData_t Afcas = (1.-(Afcaf));
    GlobalData_t AiS = (1.-(AiF));
    GlobalData_t CaMKa = (CaMKb+sv->CaMKt);
    GlobalData_t IClCa = (IClCa_junc+IClCa_sl);
    GlobalData_t IKb = ((GKb*xkb)*(v-(EK)));
    GlobalData_t IKr = (((GKr*(sqrt((ko/5.))))*sv->O)*(v-(EK)));
    GlobalData_t IKs = ((((GKs*KsCa)*sv->xs1)*sv->xs2)*(v-(EKs)));
    GlobalData_t INab = (((PNab*vffrt)*((sv->nai*(exp(vfrt)))-(nao)))/((exp(vfrt))-(1.)));
    GlobalData_t I_katp = ((((fkatp*gkatp)*akik)*bkik)*(v-(EK)));
    GlobalData_t Knai = (Knai0*(exp(((delta*vfrt)/3.))));
    GlobalData_t Knao = (Knao0*(exp((((1.-(delta))*vfrt)/3.))));
    GlobalData_t aK1 = (4.094/(1.+(exp((0.1217*((v-(EK))-(49.934)))))));
    GlobalData_t b3 = (((k3m*P)*H)/(1.+(MgATP/Kmgatp)));
    GlobalData_t bK1 = (((15.72*(exp((0.0674*((v-(EK))-(3.257))))))+(exp((0.0618*((v-(EK))-(594.31))))))/(1.+(exp((-0.1629*((v-(EK))+14.207))))));
    GlobalData_t gamma_cai = (exp((((-constA)*4.)*(((sqrt(Ii))/(1.+(sqrt(Ii))))-((0.3*Ii))))));
    GlobalData_t gamma_cass = (exp((((-constA)*4.)*(((sqrt(Iss))/(1.+(sqrt(Iss))))-((0.3*Iss))))));
    GlobalData_t gamma_ki = (exp(((-constA)*(((sqrt(Ii))/(1.+(sqrt(Ii))))-((0.3*Ii))))));
    GlobalData_t gamma_kss = (exp(((-constA)*(((sqrt(Iss))/(1.+(sqrt(Iss))))-((0.3*Iss))))));
    GlobalData_t gamma_nai = (exp(((-constA)*(((sqrt(Ii))/(1.+(sqrt(Ii))))-((0.3*Ii))))));
    GlobalData_t gamma_nass = (exp(((-constA)*(((sqrt(Iss))/(1.+(sqrt(Iss))))-((0.3*Iss))))));
    GlobalData_t h5_i = ((sv->nai*sv->nai)/((h4_i*kna1)*kna2));
    GlobalData_t h5_ss = ((sv->nass*sv->nass)/((h4_ss*kna1)*kna2));
    GlobalData_t h6_i = (1./h4_i);
    GlobalData_t h6_ss = (1./h4_ss);
    GlobalData_t hca = (exp((qca*vfrt)));
    GlobalData_t hna = (exp((qna*vfrt)));
    GlobalData_t ICab = ((((PCab*4.)*vffrt)*(((gamma_cai*sv->cai)*(exp((2.*vfrt))))-((gamma_cao*cao))))/((exp((2.*vfrt)))-(1.)));
    GlobalData_t K1ss = (aK1/(aK1+bK1));
    GlobalData_t PhiCaK_i = ((vffrt*(((gamma_ki*sv->ki)*(exp(vfrt)))-((gamma_ko*ko))))/((exp(vfrt))-(1.)));
    GlobalData_t PhiCaK_ss = ((vffrt*(((gamma_kss*sv->kss)*(exp(vfrt)))-((gamma_ko*ko))))/((exp(vfrt))-(1.)));
    GlobalData_t PhiCaL_i = (((4.*vffrt)*(((gamma_cai*sv->cai)*(exp((2.*vfrt))))-((gamma_cao*cao))))/((exp((2.*vfrt)))-(1.)));
    GlobalData_t PhiCaL_ss = (((4.*vffrt)*(((gamma_cass*sv->cass)*(exp((2.*vfrt))))-((gamma_cao*cao))))/((exp((2.*vfrt)))-(1.)));
    GlobalData_t PhiCaNa_i = ((vffrt*(((gamma_nai*sv->nai)*(exp(vfrt)))-((gamma_nao*nao))))/((exp(vfrt))-(1.)));
    GlobalData_t PhiCaNa_ss = ((vffrt*(((gamma_nass*sv->nass)*(exp(vfrt)))-((gamma_nao*nao))))/((exp(vfrt))-(1.)));
    GlobalData_t a1 = ((k1p*(((sv->nai/Knai)*(sv->nai/Knai))*(sv->nai/Knai)))/(((((1.+(sv->nai/Knai))*(1.+(sv->nai/Knai)))*(1.+(sv->nai/Knai)))+((1.+(sv->ki/Kki))*(1.+(sv->ki/Kki))))-(1.)));
    GlobalData_t a3 = ((k3p*((ko/Kko)*(ko/Kko)))/(((((1.+(nao/Knao))*(1.+(nao/Knao)))*(1.+(nao/Knao)))+((1.+(ko/Kko))*(1.+(ko/Kko))))-(1.)));
    GlobalData_t b2 = ((k2m*(((nao/Knao)*(nao/Knao))*(nao/Knao)))/(((((1.+(nao/Knao))*(1.+(nao/Knao)))*(1.+(nao/Knao)))+((1.+(ko/Kko))*(1.+(ko/Kko))))-(1.)));
    GlobalData_t b4 = ((k4m*((sv->ki/Kki)*(sv->ki/Kki)))/(((((1.+(sv->nai/Knai))*(1.+(sv->nai/Knai)))*(1.+(sv->nai/Knai)))+((1.+(sv->ki/Kki))*(1.+(sv->ki/Kki))))-(1.)));
    GlobalData_t fICaLp = (1./(1.+(KmCaMK/CaMKa)));
    GlobalData_t fINaLp = (1./(1.+(KmCaMK/CaMKa)));
    GlobalData_t fINap = (1./(1.+(KmCaMK/CaMKa)));
    GlobalData_t fItop = (1./(1.+(KmCaMK/CaMKa)));
    GlobalData_t fca = ((Afcaf*sv->fcaf)+(Afcas*sv->fcas));
    GlobalData_t fcap = ((Afcaf*sv->fcafp)+(Afcas*sv->fcas));
    GlobalData_t h1_i = (1.+((sv->nai/kna3)*(1.+hna)));
    GlobalData_t h1_ss = (1.+((sv->nass/kna3)*(1.+hna)));
    GlobalData_t h7_i = (1.+((nao/kna3)*(1.+(1./hna))));
    GlobalData_t h7_ss = (1.+((nao/kna3)*(1.+(1./hna))));
    GlobalData_t i_gate = ((AiF*sv->iF)+(AiS*sv->iS));
    GlobalData_t ip = ((AiF*sv->iFp)+(AiS*sv->iSp));
    GlobalData_t k6_i = ((h6_i*sv->cai)*kcaon);
    GlobalData_t k6_ss = ((h6_ss*sv->cass)*kcaon);
    GlobalData_t ICaK_i = ((1.-(ICaL_fractionSS))*((((((1.-(fICaLp))*PCaK)*PhiCaK_i)*sv->d)*((f*(1.-(sv->nca_i)))+((sv->jca*fca)*sv->nca_i)))+((((fICaLp*PCaKp)*PhiCaK_i)*sv->d)*((fp*(1.-(sv->nca_i)))+((sv->jca*fcap)*sv->nca_i)))));
    GlobalData_t ICaK_ss = (ICaL_fractionSS*((((((1.-(fICaLp))*PCaK)*PhiCaK_ss)*sv->d)*((f*(1.-(sv->nca_ss)))+((sv->jca*fca)*sv->nca_ss)))+((((fICaLp*PCaKp)*PhiCaK_ss)*sv->d)*((fp*(1.-(sv->nca_ss)))+((sv->jca*fcap)*sv->nca_ss)))));
    GlobalData_t ICaL_i = ((1.-(ICaL_fractionSS))*((((((1.-(fICaLp))*PCa)*PhiCaL_i)*sv->d)*((f*(1.-(sv->nca_i)))+((sv->jca*fca)*sv->nca_i)))+((((fICaLp*PCap)*PhiCaL_i)*sv->d)*((fp*(1.-(sv->nca_i)))+((sv->jca*fcap)*sv->nca_i)))));
    GlobalData_t ICaL_ss = (ICaL_fractionSS*((((((1.-(fICaLp))*PCa)*PhiCaL_ss)*sv->d)*((f*(1.-(sv->nca_ss)))+((sv->jca*fca)*sv->nca_ss)))+((((fICaLp*PCap)*PhiCaL_ss)*sv->d)*((fp*(1.-(sv->nca_ss)))+((sv->jca*fcap)*sv->nca_ss)))));
    GlobalData_t ICaNa_i = ((1.-(ICaL_fractionSS))*((((((1.-(fICaLp))*PCaNa)*PhiCaNa_i)*sv->d)*((f*(1.-(sv->nca_i)))+((sv->jca*fca)*sv->nca_i)))+((((fICaLp*PCaNap)*PhiCaNa_i)*sv->d)*((fp*(1.-(sv->nca_i)))+((sv->jca*fcap)*sv->nca_i)))));
    GlobalData_t ICaNa_ss = (ICaL_fractionSS*((((((1.-(fICaLp))*PCaNa)*PhiCaNa_ss)*sv->d)*((f*(1.-(sv->nca_ss)))+((sv->jca*fca)*sv->nca_ss)))+((((fICaLp*PCaNap)*PhiCaNa_ss)*sv->d)*((fp*(1.-(sv->nca_ss)))+((sv->jca*fcap)*sv->nca_ss)))));
    GlobalData_t IK1 = (((GK1*(sqrt((ko/5.))))*K1ss)*(v-(EK)));
    GlobalData_t INa = (((GNa*(v-(ENa)))*((sv->m*sv->m)*sv->m))*((((1.-(fINap))*sv->h)*sv->j)+((fINap*sv->hp)*sv->jp)));
    GlobalData_t INaL = (((GNaL*(v-(ENa)))*sv->mL)*(((1.-(fINaLp))*sv->hL)+(fINaLp*sv->hLp)));
    GlobalData_t Ito = ((Gto*(v-(EK)))*((((1.-(fItop))*sv->a)*i_gate)+((fItop*sv->ap)*ip)));
    GlobalData_t h2_i = ((sv->nai*hna)/(kna3*h1_i));
    GlobalData_t h2_ss = ((sv->nass*hna)/(kna3*h1_ss));
    GlobalData_t h3_i = (1./h1_i);
    GlobalData_t h3_ss = (1./h1_ss);
    GlobalData_t h8_i = (nao/((kna3*hna)*h7_i));
    GlobalData_t h8_ss = (nao/((kna3*hna)*h7_ss));
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
    GlobalData_t k3p_i = (h9_i*wca);
    GlobalData_t k3p_ss = (h9_ss*wca);
    GlobalData_t k3pp_i = (h8_i*wnaca);
    GlobalData_t k3pp_ss = (h8_ss*wnaca);
    GlobalData_t k4p_i = ((h3_i*wca)/hca);
    GlobalData_t k4p_ss = ((h3_ss*wca)/hca);
    GlobalData_t k4pp_i = (h2_i*wnaca);
    GlobalData_t k4pp_ss = (h2_ss*wnaca);
    GlobalData_t k7_i = ((h5_i*h2_i)*wna);
    GlobalData_t k7_ss = ((h5_ss*h2_ss)*wna);
    GlobalData_t k8_i = ((h8_i*h11_i)*wna);
    GlobalData_t k8_ss = ((h8_ss*h11_ss)*wna);
    GlobalData_t JnakK = (2.*((E4*b1)-((E3*a1))));
    GlobalData_t JnakNa = (3.*((E1*a3)-((E2*b3))));
    GlobalData_t k3_i = (k3p_i+k3pp_i);
    GlobalData_t k3_ss = (k3p_ss+k3pp_ss);
    GlobalData_t k4_i = (k4p_i+k4pp_i);
    GlobalData_t k4_ss = (k4p_ss+k4pp_ss);
    GlobalData_t INaK = (Pnak*((zna*JnakNa)+(zk*JnakK)));
    GlobalData_t x1_i = (((k2_i*k4_i)*(k7_i+k6_i))+((k5_i*k7_i)*(k2_i+k3_i)));
    GlobalData_t x1_ss = (((k2_ss*k4_ss)*(k7_ss+k6_ss))+((k5_ss*k7_ss)*(k2_ss+k3_ss)));
    GlobalData_t x2_i = (((k1_i*k7_i)*(k4_i+k5_i))+((k4_i*k6_i)*(k1_i+k8_i)));
    GlobalData_t x2_ss = (((k1_ss*k7_ss)*(k4_ss+k5_ss))+((k4_ss*k6_ss)*(k1_ss+k8_ss)));
    GlobalData_t x3_i = (((k1_i*k3_i)*(k7_i+k6_i))+((k8_i*k6_i)*(k2_i+k3_i)));
    GlobalData_t x3_ss = (((k1_ss*k3_ss)*(k7_ss+k6_ss))+((k8_ss*k6_ss)*(k2_ss+k3_ss)));
    GlobalData_t x4_i = (((k2_i*k8_i)*(k4_i+k5_i))+((k3_i*k5_i)*(k1_i+k8_i)));
    GlobalData_t x4_ss = (((k2_ss*k8_ss)*(k4_ss+k5_ss))+((k3_ss*k5_ss)*(k1_ss+k8_ss)));
    GlobalData_t E1_i = (x1_i/(((x1_i+x2_i)+x3_i)+x4_i));
    GlobalData_t E1_ss = (x1_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
    GlobalData_t E2_i = (x2_i/(((x1_i+x2_i)+x3_i)+x4_i));
    GlobalData_t E2_ss = (x2_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
    GlobalData_t E3_i = (x3_i/(((x1_i+x2_i)+x3_i)+x4_i));
    GlobalData_t E3_ss = (x3_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
    GlobalData_t E4_i = (x4_i/(((x1_i+x2_i)+x3_i)+x4_i));
    GlobalData_t E4_ss = (x4_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
    GlobalData_t JncxCa_i = ((E2_i*k2_i)-((E1_i*k1_i)));
    GlobalData_t JncxCa_ss = ((E2_ss*k2_ss)-((E1_ss*k1_ss)));
    GlobalData_t JncxNa_i = (((3.*((E4_i*k7_i)-((E1_i*k8_i))))+(E3_i*k4pp_i))-((E2_i*k3pp_i)));
    GlobalData_t JncxNa_ss = (((3.*((E4_ss*k7_ss)-((E1_ss*k8_ss))))+(E3_ss*k4pp_ss))-((E2_ss*k3pp_ss)));
    GlobalData_t INaCa_i = ((((1.-(INaCa_fractionSS))*Gncx)*allo_i)*((zna*JncxNa_i)+(zca*JncxCa_i)));
    GlobalData_t INaCa_ss = (((INaCa_fractionSS*Gncx)*allo_ss)*((zna*JncxNa_ss)+(zca*JncxCa_ss)));
    Iion = (-(-((((((((((((((((((INa+INaL)+Ito)+ICaL)+ICaNa)+ICaK)+IKr)+IKs)+IK1)+INaCa_i)+INaCa_ss)+INaK)+INab)+IKb)+IpCa)+ICab)+IClCa)+IClb)+I_katp)));
    
    
    //Complete Forward Euler Update
    
    
    //Complete Rush Larsen Update
    
    
    //Complete RK2 Update
    
    
    //Complete RK4 Update
    
    
    //Complete Sundnes Update
    
    
    //Complete Markov Backward Euler method
    
    
    //Complete Rosenbrock Update
    
    
    //Complete Cvode Update
    #ifdef USE_CVODE
    NV_Ith_S(CVODE,CVODE_C1) = sv->C1;
    NV_Ith_S(CVODE,CVODE_C2) = sv->C2;
    NV_Ith_S(CVODE,CVODE_C3) = sv->C3;
    NV_Ith_S(CVODE,CVODE_CaMKt) = sv->CaMKt;
    NV_Ith_S(CVODE,CVODE_I) = sv->I;
    NV_Ith_S(CVODE,CVODE_Jrel_np) = sv->Jrel_np;
    NV_Ith_S(CVODE,CVODE_Jrel_p) = sv->Jrel_p;
    NV_Ith_S(CVODE,CVODE_O) = sv->O;
    NV_Ith_S(CVODE,CVODE_a) = sv->a;
    NV_Ith_S(CVODE,CVODE_ap) = sv->ap;
    NV_Ith_S(CVODE,CVODE_cai) = sv->cai;
    NV_Ith_S(CVODE,CVODE_cajsr) = sv->cajsr;
    NV_Ith_S(CVODE,CVODE_cansr) = sv->cansr;
    NV_Ith_S(CVODE,CVODE_cass) = sv->cass;
    NV_Ith_S(CVODE,CVODE_d) = sv->d;
    NV_Ith_S(CVODE,CVODE_fcaf) = sv->fcaf;
    NV_Ith_S(CVODE,CVODE_fcafp) = sv->fcafp;
    NV_Ith_S(CVODE,CVODE_fcas) = sv->fcas;
    NV_Ith_S(CVODE,CVODE_ff) = sv->ff;
    NV_Ith_S(CVODE,CVODE_ffp) = sv->ffp;
    NV_Ith_S(CVODE,CVODE_fs) = sv->fs;
    NV_Ith_S(CVODE,CVODE_h) = sv->h;
    NV_Ith_S(CVODE,CVODE_hL) = sv->hL;
    NV_Ith_S(CVODE,CVODE_hLp) = sv->hLp;
    NV_Ith_S(CVODE,CVODE_hp) = sv->hp;
    NV_Ith_S(CVODE,CVODE_iF) = sv->iF;
    NV_Ith_S(CVODE,CVODE_iFp) = sv->iFp;
    NV_Ith_S(CVODE,CVODE_iS) = sv->iS;
    NV_Ith_S(CVODE,CVODE_iSp) = sv->iSp;
    NV_Ith_S(CVODE,CVODE_j) = sv->j;
    NV_Ith_S(CVODE,CVODE_jca) = sv->jca;
    NV_Ith_S(CVODE,CVODE_jp) = sv->jp;
    NV_Ith_S(CVODE,CVODE_ki) = sv->ki;
    NV_Ith_S(CVODE,CVODE_kss) = sv->kss;
    NV_Ith_S(CVODE,CVODE_m) = sv->m;
    NV_Ith_S(CVODE,CVODE_mL) = sv->mL;
    NV_Ith_S(CVODE,CVODE_nai) = sv->nai;
    NV_Ith_S(CVODE,CVODE_nass) = sv->nass;
    NV_Ith_S(CVODE,CVODE_nca_i) = sv->nca_i;
    NV_Ith_S(CVODE,CVODE_nca_ss) = sv->nca_ss;
    NV_Ith_S(CVODE,CVODE_xs1) = sv->xs1;
    NV_Ith_S(CVODE,CVODE_xs2) = sv->xs2;
    ion_private->nr.v = v;
    
    
    //Do the solve
    int CVODE_flag;
    CVODE_flag = CVodeReInit(cvode_mem, t, CVODE);
    assert(CVODE_flag == CV_SUCCESS);
    CVODE_flag = CVodeSetInitStep(cvode_mem, dt);
    assert(CVODE_flag == CV_SUCCESS);
    #if SUNDIALS_VERSION_MAJOR >= 7
    sunrealtype tret;
    #else
    realtype tret;
    #endif
    ion_private->node_number = __i + 0;
    CVODE_flag = CVode(cvode_mem, t+dt, CVODE, &tret, CV_NORMAL);
    assert(CVODE_flag == CV_SUCCESS);
    #endif
    
    //Unload the data back into LIMPET structures
    #ifdef USE_CVODE
    GlobalData_t C1_new = NV_Ith_S(CVODE,CVODE_C1);
    GlobalData_t C2_new = NV_Ith_S(CVODE,CVODE_C2);
    GlobalData_t C3_new = NV_Ith_S(CVODE,CVODE_C3);
    GlobalData_t CaMKt_new = NV_Ith_S(CVODE,CVODE_CaMKt);
    GlobalData_t I_new = NV_Ith_S(CVODE,CVODE_I);
    GlobalData_t Jrel_np_new = NV_Ith_S(CVODE,CVODE_Jrel_np);
    GlobalData_t Jrel_p_new = NV_Ith_S(CVODE,CVODE_Jrel_p);
    GlobalData_t O_new = NV_Ith_S(CVODE,CVODE_O);
    GlobalData_t a_new = NV_Ith_S(CVODE,CVODE_a);
    GlobalData_t ap_new = NV_Ith_S(CVODE,CVODE_ap);
    GlobalData_t cai_new = NV_Ith_S(CVODE,CVODE_cai);
    GlobalData_t cajsr_new = NV_Ith_S(CVODE,CVODE_cajsr);
    GlobalData_t cansr_new = NV_Ith_S(CVODE,CVODE_cansr);
    GlobalData_t cass_new = NV_Ith_S(CVODE,CVODE_cass);
    GlobalData_t d_new = NV_Ith_S(CVODE,CVODE_d);
    GlobalData_t fcaf_new = NV_Ith_S(CVODE,CVODE_fcaf);
    GlobalData_t fcafp_new = NV_Ith_S(CVODE,CVODE_fcafp);
    GlobalData_t fcas_new = NV_Ith_S(CVODE,CVODE_fcas);
    GlobalData_t ff_new = NV_Ith_S(CVODE,CVODE_ff);
    GlobalData_t ffp_new = NV_Ith_S(CVODE,CVODE_ffp);
    GlobalData_t fs_new = NV_Ith_S(CVODE,CVODE_fs);
    GlobalData_t h_new = NV_Ith_S(CVODE,CVODE_h);
    GlobalData_t hL_new = NV_Ith_S(CVODE,CVODE_hL);
    GlobalData_t hLp_new = NV_Ith_S(CVODE,CVODE_hLp);
    GlobalData_t hp_new = NV_Ith_S(CVODE,CVODE_hp);
    GlobalData_t iF_new = NV_Ith_S(CVODE,CVODE_iF);
    GlobalData_t iFp_new = NV_Ith_S(CVODE,CVODE_iFp);
    GlobalData_t iS_new = NV_Ith_S(CVODE,CVODE_iS);
    GlobalData_t iSp_new = NV_Ith_S(CVODE,CVODE_iSp);
    GlobalData_t j_new = NV_Ith_S(CVODE,CVODE_j);
    GlobalData_t jca_new = NV_Ith_S(CVODE,CVODE_jca);
    GlobalData_t jp_new = NV_Ith_S(CVODE,CVODE_jp);
    GlobalData_t ki_new = NV_Ith_S(CVODE,CVODE_ki);
    GlobalData_t kss_new = NV_Ith_S(CVODE,CVODE_kss);
    GlobalData_t m_new = NV_Ith_S(CVODE,CVODE_m);
    GlobalData_t mL_new = NV_Ith_S(CVODE,CVODE_mL);
    GlobalData_t nai_new = NV_Ith_S(CVODE,CVODE_nai);
    GlobalData_t nass_new = NV_Ith_S(CVODE,CVODE_nass);
    GlobalData_t nca_i_new = NV_Ith_S(CVODE,CVODE_nca_i);
    GlobalData_t nca_ss_new = NV_Ith_S(CVODE,CVODE_nca_ss);
    GlobalData_t xs1_new = NV_Ith_S(CVODE,CVODE_xs1);
    GlobalData_t xs2_new = NV_Ith_S(CVODE,CVODE_xs2);
    #else
    GlobalData_t C1_new = 0;
    GlobalData_t C2_new = 0;
    GlobalData_t C3_new = 0;
    GlobalData_t CaMKt_new = 0;
    GlobalData_t I_new = 0;
    GlobalData_t Jrel_np_new = 0;
    GlobalData_t Jrel_p_new = 0;
    GlobalData_t O_new = 0;
    GlobalData_t a_new = 0;
    GlobalData_t ap_new = 0;
    GlobalData_t cai_new = 0;
    GlobalData_t cajsr_new = 0;
    GlobalData_t cansr_new = 0;
    GlobalData_t cass_new = 0;
    GlobalData_t d_new = 0;
    GlobalData_t fcaf_new = 0;
    GlobalData_t fcafp_new = 0;
    GlobalData_t fcas_new = 0;
    GlobalData_t ff_new = 0;
    GlobalData_t ffp_new = 0;
    GlobalData_t fs_new = 0;
    GlobalData_t h_new = 0;
    GlobalData_t hL_new = 0;
    GlobalData_t hLp_new = 0;
    GlobalData_t hp_new = 0;
    GlobalData_t iF_new = 0;
    GlobalData_t iFp_new = 0;
    GlobalData_t iS_new = 0;
    GlobalData_t iSp_new = 0;
    GlobalData_t j_new = 0;
    GlobalData_t jca_new = 0;
    GlobalData_t jp_new = 0;
    GlobalData_t ki_new = 0;
    GlobalData_t kss_new = 0;
    GlobalData_t m_new = 0;
    GlobalData_t mL_new = 0;
    GlobalData_t nai_new = 0;
    GlobalData_t nass_new = 0;
    GlobalData_t nca_i_new = 0;
    GlobalData_t nca_ss_new = 0;
    GlobalData_t xs1_new = 0;
    GlobalData_t xs2_new = 0;
    #endif
    
    
    //Finish the update
    sv->C1 = C1_new;
    sv->C2 = C2_new;
    sv->C3 = C3_new;
    sv->CaMKt = CaMKt_new;
    sv->I = I_new;
    Iion = Iion;
    sv->Jrel_np = Jrel_np_new;
    sv->Jrel_p = Jrel_p_new;
    sv->O = O_new;
    sv->a = a_new;
    sv->ap = ap_new;
    sv->cai = cai_new;
    sv->cajsr = cajsr_new;
    sv->cansr = cansr_new;
    sv->cass = cass_new;
    sv->d = d_new;
    sv->fcaf = fcaf_new;
    sv->fcafp = fcafp_new;
    sv->fcas = fcas_new;
    sv->ff = ff_new;
    sv->ffp = ffp_new;
    sv->fs = fs_new;
    sv->h = h_new;
    sv->hL = hL_new;
    sv->hLp = hLp_new;
    sv->hp = hp_new;
    sv->iF = iF_new;
    sv->iFp = iFp_new;
    sv->iS = iS_new;
    sv->iSp = iSp_new;
    sv->j = j_new;
    sv->jca = jca_new;
    sv->jp = jp_new;
    sv->ki = ki_new;
    sv->kss = kss_new;
    sv->m = m_new;
    sv->mL = mL_new;
    sv->nai = nai_new;
    sv->nass = nass_new;
    sv->nca_i = nca_i_new;
    sv->nca_ss = nca_ss_new;
    sv->xs1 = xs1_new;
    sv->xs2 = xs2_new;
    //Change the units of external variables as appropriate.
    
    
    //Save all external vars
    Iion_ext[__i] = Iion;
    v_ext[__i] = v;

  }

  //free y
#ifdef USE_CVODE
  N_VDestroy(CVODE);
#endif

            }
}
#endif // TORORD_FKATP_EPI_CPU_GENERATED

#ifdef USE_CVODE
#if SUNDIALS_VERSION_MAJOR >= 7
int ToRORd_fkatp_epi_internal_rhs_function(sunrealtype t, N_Vector CVODE, N_Vector CVODE_dot, void* user_data) {
#else
int ToRORd_fkatp_epi_internal_rhs_function(realtype t, N_Vector CVODE, N_Vector CVODE_dot, void* user_data) {
#endif

  ToRORd_fkatp_epiIonType::private_type* ion_private = (ToRORd_fkatp_epiIonType::private_type*)user_data;
  ToRORd_fkatp_epi_Private_cpu::regional_constants_type* rc = &ion_private->rc;
  ToRORd_fkatp_epiIonType::IonIfDerived *IF = static_cast<ToRORd_fkatp_epiIonType::IonIfDerived*>(ion_private->IF);
  ToRORd_fkatp_epi_Nodal_Req_cpu* nr = &ion_private->nr;
  int __i = ((ToRORd_fkatp_epiIonType::private_type*)user_data)->node_number;
  cell_geom *region = &IF->cgeom();
  ToRORd_fkatp_epi_Params *p = IF->params();
  ToRORd_fkatp_epi_state *sv_base = IF->sv_tab().data();
  ToRORd_fkatp_epi_state *sv = sv_base+__i;
  int inner_id = 0;

  //Check to see if input is valid
  
  //Compute the equations
  GlobalData_t Afcaf = (0.3+(0.6/(1.+(exp(((nr->v-(10.))/10.))))));
  GlobalData_t AiF = (1./(1.+(exp((((nr->v+EKshift)-(213.6))/151.2)))));
  GlobalData_t Bcai = (1./((1.+((cmdnmax*kmcmdn)/((kmcmdn+NV_Ith_S(CVODE,CVODE_cai))*(kmcmdn+NV_Ith_S(CVODE,CVODE_cai)))))+((trpnmax*kmtrpn)/((kmtrpn+NV_Ith_S(CVODE,CVODE_cai))*(kmtrpn+NV_Ith_S(CVODE,CVODE_cai))))));
  GlobalData_t Bcajsr = (1./(1.+((csqnmax*kmcsqn)/((kmcsqn+NV_Ith_S(CVODE,CVODE_cajsr))*(kmcsqn+NV_Ith_S(CVODE,CVODE_cajsr))))));
  GlobalData_t Bcass = (1./((1.+((BSRmax*KmBSR)/((KmBSR+NV_Ith_S(CVODE,CVODE_cass))*(KmBSR+NV_Ith_S(CVODE,CVODE_cass)))))+((BSLmax*KmBSL)/((KmBSL+NV_Ith_S(CVODE,CVODE_cass))*(KmBSL+NV_Ith_S(CVODE,CVODE_cass))))));
  GlobalData_t CaMKb = ((CaMKo*(1.-(NV_Ith_S(CVODE,CVODE_CaMKt))))/(1.+(KmCaM/NV_Ith_S(CVODE,CVODE_cass))));
  GlobalData_t EK = (((R*T)/(zk*F))*(log((ko/NV_Ith_S(CVODE,CVODE_ki)))));
  GlobalData_t EKs = (((R*T)/(zk*F))*(log(((ko+(PKNa*nao))/(NV_Ith_S(CVODE,CVODE_ki)+(PKNa*NV_Ith_S(CVODE,CVODE_nai)))))));
  GlobalData_t ENa = (((R*T)/(zna*F))*(log((nao/NV_Ith_S(CVODE,CVODE_nai)))));
  GlobalData_t Ii = ((0.5*(((NV_Ith_S(CVODE,CVODE_nai)+NV_Ith_S(CVODE,CVODE_ki))+cli)+(4.*NV_Ith_S(CVODE,CVODE_cai))))/1000.);
  GlobalData_t IpCa = ((GpCa*NV_Ith_S(CVODE,CVODE_cai))/(KmCap+NV_Ith_S(CVODE,CVODE_cai)));
  GlobalData_t Iss = ((0.5*(((NV_Ith_S(CVODE,CVODE_nass)+NV_Ith_S(CVODE,CVODE_kss))+cli)+(4.*NV_Ith_S(CVODE,CVODE_cass))))/1000.);
  GlobalData_t Jdiff = ((NV_Ith_S(CVODE,CVODE_cass)-(NV_Ith_S(CVODE,CVODE_cai)))/tauCa);
  GlobalData_t JdiffK = ((NV_Ith_S(CVODE,CVODE_kss)-(NV_Ith_S(CVODE,CVODE_ki)))/tauK);
  GlobalData_t JdiffNa = ((NV_Ith_S(CVODE,CVODE_nass)-(NV_Ith_S(CVODE,CVODE_nai)))/tauNa);
  GlobalData_t Jleak = ((0.0048825*NV_Ith_S(CVODE,CVODE_cansr))/15.);
  GlobalData_t Jtr = ((NV_Ith_S(CVODE,CVODE_cansr)-(NV_Ith_S(CVODE,CVODE_cajsr)))/60.);
  GlobalData_t Jupnp = (((upScale*0.005425)*NV_Ith_S(CVODE,CVODE_cai))/(NV_Ith_S(CVODE,CVODE_cai)+0.00092));
  GlobalData_t Jupp = ((((upScale*2.75)*0.005425)*NV_Ith_S(CVODE,CVODE_cai))/((NV_Ith_S(CVODE,CVODE_cai)+0.00092)-(0.00017)));
  GlobalData_t KsCa = (1.+(0.6/(1.+(pow((3.8e-5/NV_Ith_S(CVODE,CVODE_cai)),1.4)))));
  GlobalData_t P = (eP/(((1.+(H/Khp))+(NV_Ith_S(CVODE,CVODE_nai)/Knap))+(NV_Ith_S(CVODE,CVODE_ki)/Kxkur)));
  GlobalData_t ah = ((nr->v>=-40.) ? 0. : (0.057*(exp(((-(nr->v+80.))/6.8)))));
  GlobalData_t aj = ((nr->v>=-40.) ? 0. : ((((-2.5428e4*(exp((0.2444*nr->v))))-((6.948e-6*(exp((-0.04391*nr->v))))))*(nr->v+37.78))/(1.+(exp((0.311*(nr->v+79.23)))))));
  GlobalData_t allo_i = (1./(1.+((KmCaAct/NV_Ith_S(CVODE,CVODE_cai))*(KmCaAct/NV_Ith_S(CVODE,CVODE_cai)))));
  GlobalData_t allo_ss = (1./(1.+((KmCaAct/NV_Ith_S(CVODE,CVODE_cass))*(KmCaAct/NV_Ith_S(CVODE,CVODE_cass)))));
  GlobalData_t ass = (1./(1.+(exp(((-((nr->v+EKshift)-(14.34)))/14.82)))));
  GlobalData_t assp = (1./(1.+(exp(((-((nr->v+EKshift)-(24.34)))/14.82)))));
  GlobalData_t bh = ((nr->v>=-40.) ? (0.77/(0.13*(1.+(exp(((-(nr->v+10.66))/11.1)))))) : ((2.7*(exp((0.079*nr->v))))+(3.1e5*(exp((0.3485*nr->v))))));
  GlobalData_t bj = ((nr->v>=-40.) ? ((0.6*(exp((0.057*nr->v))))/(1.+(exp((-0.1*(nr->v+32.)))))) : ((0.02424*(exp((-0.01052*nr->v))))/(1.+(exp((-0.1378*(nr->v+40.14)))))));
  GlobalData_t delta_epi = ((celltype==1.) ? (1.-((0.95/(1.+(exp((((nr->v+EKshift)+70.)/5.))))))) : 1.);
  GlobalData_t dss = ((nr->v>=31.4978) ? 1. : (1.0763*(exp((-1.0070*(exp((-0.0829*nr->v))))))));
  GlobalData_t dti_develop = (1.354+(1.e-4/((exp((((nr->v+EKshift)-(167.4))/15.89)))+(exp(((-((nr->v+EKshift)-(12.23)))/0.2154))))));
  GlobalData_t dti_recover = (1.-((0.5/(1.+(exp((((nr->v+EKshift)+70.0)/20.0)))))));
  GlobalData_t f = ((Aff*NV_Ith_S(CVODE,CVODE_ff))+(Afs*NV_Ith_S(CVODE,CVODE_fs)));
  GlobalData_t fp = ((Aff*NV_Ith_S(CVODE,CVODE_ffp))+(Afs*NV_Ith_S(CVODE,CVODE_fs)));
  GlobalData_t fss = (1./(1.+(exp(((nr->v+19.58)/3.696)))));
  GlobalData_t h4_i = (1.+((NV_Ith_S(CVODE,CVODE_nai)/kna1)*(1.+(NV_Ith_S(CVODE,CVODE_nai)/kna2))));
  GlobalData_t h4_ss = (1.+((NV_Ith_S(CVODE,CVODE_nass)/kna1)*(1.+(NV_Ith_S(CVODE,CVODE_nass)/kna2))));
  GlobalData_t hLss = (1./(1.+(exp(((nr->v+87.61)/7.488)))));
  GlobalData_t hLssp = (1./(1.+(exp(((nr->v+93.81)/7.488)))));
  GlobalData_t hss = (1./((1.+(exp(((nr->v+71.55)/7.43))))*(1.+(exp(((nr->v+71.55)/7.43))))));
  GlobalData_t hssp = (1./((1.+(exp(((nr->v+77.55)/7.43))))*(1.+(exp(((nr->v+77.55)/7.43))))));
  GlobalData_t iss = (1./(1.+(exp((((nr->v+EKshift)+43.94)/5.711)))));
  GlobalData_t jcass = (1.0/(1.0+(exp(((nr->v+18.08)/2.7916)))));
  GlobalData_t km2n = NV_Ith_S(CVODE,CVODE_jca);
  GlobalData_t mLss = (1./(1.+(exp(((-(nr->v+42.85))/5.264)))));
  GlobalData_t mss = (1./((1.+(exp(((-(nr->v+56.86))/9.03))))*(1.+(exp(((-(nr->v+56.86))/9.03))))));
  GlobalData_t ta = (1.0515/((1./(1.2089*(1.+(exp(((-((nr->v+EKshift)-(18.4099)))/29.3814))))))+(3.5/(1.+(exp((((nr->v+EKshift)+100.)/29.3814)))))));
  GlobalData_t tau_rel_b = (bt/(1.+(0.0123/NV_Ith_S(CVODE,CVODE_cajsr))));
  GlobalData_t tau_relp_b = (btp/(1.+(0.0123/NV_Ith_S(CVODE,CVODE_cajsr))));
  GlobalData_t td = ((offset+0.6)+(1./((exp((-0.05*((nr->v+vShift)+6.))))+(exp((0.09*((nr->v+vShift)+14.)))))));
  GlobalData_t tfcaf = (7.+(1./((0.04*(exp(((-(nr->v-(4.)))/7.))))+(0.04*(exp(((nr->v-(4.))/7.)))))));
  GlobalData_t tfcas = (100.+(1./((0.00012*(exp(((-nr->v)/3.))))+(0.00012*(exp((nr->v/7.)))))));
  GlobalData_t tff = (7.+(1./((0.0045*(exp(((-(nr->v+20.))/10.))))+(0.0045*(exp(((nr->v+20.)/10.)))))));
  GlobalData_t tfs = (1000.+(1./((0.000035*(exp(((-(nr->v+5.))/4.))))+(0.000035*(exp(((nr->v+5.)/6.)))))));
  GlobalData_t tiF_b = (4.562+(1./((0.3933*(exp(((-((nr->v+EKshift)+100.))/100.))))+(0.08004*(exp((((nr->v+EKshift)+50.)/16.59)))))));
  GlobalData_t tiS_b = (23.62+(1./((0.001416*(exp(((-((nr->v+EKshift)+96.52))/59.05))))+(1.78e-8*(exp((((nr->v+EKshift)+114.1)/8.079)))))));
  GlobalData_t tm = ((0.1292*(exp((-(((nr->v+45.79)/15.54)*((nr->v+45.79)/15.54))))))+(0.06487*(exp((-(((nr->v-(4.823))/51.12)*((nr->v-(4.823))/51.12)))))));
  GlobalData_t tmL = ((0.1292*(exp((-(((nr->v+45.79)/15.54)*((nr->v+45.79)/15.54))))))+(0.06487*(exp((-(((nr->v-(4.823))/51.12)*((nr->v-(4.823))/51.12)))))));
  GlobalData_t txs1 = (817.3+(1./((2.326e-4*(exp(((nr->v+48.28)/17.8))))+(0.001292*(exp(((-(nr->v+210.))/230.)))))));
  GlobalData_t txs2 = (1./((0.01*(exp(((nr->v-(50.))/20.))))+(0.0193*(exp(((-(nr->v+66.54))/31.))))));
  GlobalData_t vffrt = (((nr->v*F)*F)/(R*T));
  GlobalData_t vfrt = ((nr->v*F)/(R*T));
  GlobalData_t xkb = (1./(1.+(exp(((-(nr->v-(10.8968)))/23.9871)))));
  GlobalData_t xs1ss = (1./(1.+(exp(((-(nr->v+11.6))/8.932)))));
  GlobalData_t Afcas = (1.-(Afcaf));
  GlobalData_t AiS = (1.-(AiF));
  GlobalData_t CaMKa = (CaMKb+NV_Ith_S(CVODE,CVODE_CaMKt));
  GlobalData_t IKb = ((GKb*xkb)*(nr->v-(EK)));
  GlobalData_t IKr = (((GKr*(sqrt((ko/5.))))*NV_Ith_S(CVODE,CVODE_O))*(nr->v-(EK)));
  GlobalData_t IKs = ((((GKs*KsCa)*NV_Ith_S(CVODE,CVODE_xs1))*NV_Ith_S(CVODE,CVODE_xs2))*(nr->v-(EKs)));
  GlobalData_t INab = (((PNab*vffrt)*((NV_Ith_S(CVODE,CVODE_nai)*(exp(vfrt)))-(nao)))/((exp(vfrt))-(1.)));
  GlobalData_t I_katp = ((((fkatp*gkatp)*akik)*bkik)*(nr->v-(EK)));
  GlobalData_t Knai = (Knai0*(exp(((delta*vfrt)/3.))));
  GlobalData_t Knao = (Knao0*(exp((((1.-(delta))*vfrt)/3.))));
  GlobalData_t aK1 = (4.094/(1.+(exp((0.1217*((nr->v-(EK))-(49.934)))))));
  GlobalData_t alpha = (0.1161*(exp((0.2990*vfrt))));
  GlobalData_t alpha_2 = (0.0578*(exp((0.9710*vfrt))));
  GlobalData_t alpha_C2ToI = (0.52e-4*(exp((1.525*vfrt))));
  GlobalData_t alpha_i = (0.2533*(exp((0.5953*vfrt))));
  GlobalData_t anca_i = (1./((k2n/km2n)+((((1.+(Kmn/NV_Ith_S(CVODE,CVODE_cai)))*(1.+(Kmn/NV_Ith_S(CVODE,CVODE_cai))))*(1.+(Kmn/NV_Ith_S(CVODE,CVODE_cai))))*(1.+(Kmn/NV_Ith_S(CVODE,CVODE_cai))))));
  GlobalData_t anca_ss = (1./((k2n/km2n)+((((1.+(Kmn/NV_Ith_S(CVODE,CVODE_cass)))*(1.+(Kmn/NV_Ith_S(CVODE,CVODE_cass))))*(1.+(Kmn/NV_Ith_S(CVODE,CVODE_cass))))*(1.+(Kmn/NV_Ith_S(CVODE,CVODE_cass))))));
  GlobalData_t b3 = (((k3m*P)*H)/(1.+(MgATP/Kmgatp)));
  GlobalData_t bK1 = (((15.72*(exp((0.0674*((nr->v-(EK))-(3.257))))))+(exp((0.0618*((nr->v-(EK))-(594.31))))))/(1.+(exp((-0.1629*((nr->v-(EK))+14.207))))));
  GlobalData_t beta = (0.2442*(exp((-1.604*vfrt))));
  GlobalData_t beta_2 = (0.349e-3*(exp((-1.062*vfrt))));
  GlobalData_t beta_i = (0.06525*(exp((-0.8209*vfrt))));
  GlobalData_t diff_CaMKt = (((aCaMK*CaMKb)*(CaMKb+NV_Ith_S(CVODE,CVODE_CaMKt)))-((bCaMK*NV_Ith_S(CVODE,CVODE_CaMKt))));
  GlobalData_t diff_a = ((ass-(NV_Ith_S(CVODE,CVODE_a)))/ta);
  GlobalData_t diff_ap = ((assp-(NV_Ith_S(CVODE,CVODE_ap)))/ta);
  GlobalData_t diff_d = ((dss-(NV_Ith_S(CVODE,CVODE_d)))/td);
  GlobalData_t diff_ff = ((fss-(NV_Ith_S(CVODE,CVODE_ff)))/tff);
  GlobalData_t diff_fs = ((fss-(NV_Ith_S(CVODE,CVODE_fs)))/tfs);
  GlobalData_t diff_hL = ((hLss-(NV_Ith_S(CVODE,CVODE_hL)))/thL);
  GlobalData_t diff_hLp = ((hLssp-(NV_Ith_S(CVODE,CVODE_hLp)))/thLp);
  GlobalData_t diff_jca = ((jcass-(NV_Ith_S(CVODE,CVODE_jca)))/tjca);
  GlobalData_t diff_m = ((mss-(NV_Ith_S(CVODE,CVODE_m)))/tm);
  GlobalData_t diff_mL = ((mLss-(NV_Ith_S(CVODE,CVODE_mL)))/tmL);
  GlobalData_t diff_xs1 = ((xs1ss-(NV_Ith_S(CVODE,CVODE_xs1)))/txs1);
  GlobalData_t fcass = fss;
  GlobalData_t gamma_cai = (exp((((-constA)*4.)*(((sqrt(Ii))/(1.+(sqrt(Ii))))-((0.3*Ii))))));
  GlobalData_t gamma_cass = (exp((((-constA)*4.)*(((sqrt(Iss))/(1.+(sqrt(Iss))))-((0.3*Iss))))));
  GlobalData_t gamma_ki = (exp(((-constA)*(((sqrt(Ii))/(1.+(sqrt(Ii))))-((0.3*Ii))))));
  GlobalData_t gamma_kss = (exp(((-constA)*(((sqrt(Iss))/(1.+(sqrt(Iss))))-((0.3*Iss))))));
  GlobalData_t gamma_nai = (exp(((-constA)*(((sqrt(Ii))/(1.+(sqrt(Ii))))-((0.3*Ii))))));
  GlobalData_t gamma_nass = (exp(((-constA)*(((sqrt(Iss))/(1.+(sqrt(Iss))))-((0.3*Iss))))));
  GlobalData_t h5_i = ((NV_Ith_S(CVODE,CVODE_nai)*NV_Ith_S(CVODE,CVODE_nai))/((h4_i*kna1)*kna2));
  GlobalData_t h5_ss = ((NV_Ith_S(CVODE,CVODE_nass)*NV_Ith_S(CVODE,CVODE_nass))/((h4_ss*kna1)*kna2));
  GlobalData_t h6_i = (1./h4_i);
  GlobalData_t h6_ss = (1./h4_ss);
  GlobalData_t hca = (exp((qca*vfrt)));
  GlobalData_t hna = (exp((qna*vfrt)));
  GlobalData_t jss = hss;
  GlobalData_t tau_rel = ((tau_rel_b<0.001) ? 0.001 : tau_rel_b);
  GlobalData_t tau_relp = ((tau_relp_b<0.001) ? 0.001 : tau_relp_b);
  GlobalData_t tfcafp = (2.5*tfcaf);
  GlobalData_t tffp = (2.5*tff);
  GlobalData_t th = (1./(ah+bh));
  GlobalData_t tiF = (tiF_b*delta_epi);
  GlobalData_t tiS = (tiS_b*delta_epi);
  GlobalData_t tj = (1./(aj+bj));
  GlobalData_t xs2ss = xs1ss;
  GlobalData_t ICab = ((((PCab*4.)*vffrt)*(((gamma_cai*NV_Ith_S(CVODE,CVODE_cai))*(exp((2.*vfrt))))-((gamma_cao*cao))))/((exp((2.*vfrt)))-(1.)));
  GlobalData_t K1ss = (aK1/(aK1+bK1));
  GlobalData_t PhiCaK_i = ((vffrt*(((gamma_ki*NV_Ith_S(CVODE,CVODE_ki))*(exp(vfrt)))-((gamma_ko*ko))))/((exp(vfrt))-(1.)));
  GlobalData_t PhiCaK_ss = ((vffrt*(((gamma_kss*NV_Ith_S(CVODE,CVODE_kss))*(exp(vfrt)))-((gamma_ko*ko))))/((exp(vfrt))-(1.)));
  GlobalData_t PhiCaL_i = (((4.*vffrt)*(((gamma_cai*NV_Ith_S(CVODE,CVODE_cai))*(exp((2.*vfrt))))-((gamma_cao*cao))))/((exp((2.*vfrt)))-(1.)));
  GlobalData_t PhiCaL_ss = (((4.*vffrt)*(((gamma_cass*NV_Ith_S(CVODE,CVODE_cass))*(exp((2.*vfrt))))-((gamma_cao*cao))))/((exp((2.*vfrt)))-(1.)));
  GlobalData_t PhiCaNa_i = ((vffrt*(((gamma_nai*NV_Ith_S(CVODE,CVODE_nai))*(exp(vfrt)))-((gamma_nao*nao))))/((exp(vfrt))-(1.)));
  GlobalData_t PhiCaNa_ss = ((vffrt*(((gamma_nass*NV_Ith_S(CVODE,CVODE_nass))*(exp(vfrt)))-((gamma_nao*nao))))/((exp(vfrt))-(1.)));
  GlobalData_t a1 = ((k1p*(((NV_Ith_S(CVODE,CVODE_nai)/Knai)*(NV_Ith_S(CVODE,CVODE_nai)/Knai))*(NV_Ith_S(CVODE,CVODE_nai)/Knai)))/(((((1.+(NV_Ith_S(CVODE,CVODE_nai)/Knai))*(1.+(NV_Ith_S(CVODE,CVODE_nai)/Knai)))*(1.+(NV_Ith_S(CVODE,CVODE_nai)/Knai)))+((1.+(NV_Ith_S(CVODE,CVODE_ki)/Kki))*(1.+(NV_Ith_S(CVODE,CVODE_ki)/Kki))))-(1.)));
  GlobalData_t a3 = ((k3p*((ko/Kko)*(ko/Kko)))/(((((1.+(nao/Knao))*(1.+(nao/Knao)))*(1.+(nao/Knao)))+((1.+(ko/Kko))*(1.+(ko/Kko))))-(1.)));
  GlobalData_t b2 = ((k2m*(((nao/Knao)*(nao/Knao))*(nao/Knao)))/(((((1.+(nao/Knao))*(1.+(nao/Knao)))*(1.+(nao/Knao)))+((1.+(ko/Kko))*(1.+(ko/Kko))))-(1.)));
  GlobalData_t b4 = ((k4m*((NV_Ith_S(CVODE,CVODE_ki)/Kki)*(NV_Ith_S(CVODE,CVODE_ki)/Kki)))/(((((1.+(NV_Ith_S(CVODE,CVODE_nai)/Knai))*(1.+(NV_Ith_S(CVODE,CVODE_nai)/Knai)))*(1.+(NV_Ith_S(CVODE,CVODE_nai)/Knai)))+((1.+(NV_Ith_S(CVODE,CVODE_ki)/Kki))*(1.+(NV_Ith_S(CVODE,CVODE_ki)/Kki))))-(1.)));
  GlobalData_t beta_ItoC2 = (((beta_2*beta_i)*alpha_C2ToI)/(alpha_2*alpha_i));
  GlobalData_t diff_C2 = (((alpha*NV_Ith_S(CVODE,CVODE_C3))+(beta_1*NV_Ith_S(CVODE,CVODE_C1)))-(((beta+alpha_1)*NV_Ith_S(CVODE,CVODE_C2))));
  GlobalData_t diff_C3 = ((beta*NV_Ith_S(CVODE,CVODE_C2))-((alpha*NV_Ith_S(CVODE,CVODE_C3))));
  GlobalData_t diff_O = (((alpha_2*NV_Ith_S(CVODE,CVODE_C1))+(beta_i*NV_Ith_S(CVODE,CVODE_I)))-(((beta_2+alpha_i)*NV_Ith_S(CVODE,CVODE_O))));
  GlobalData_t diff_fcaf = ((fcass-(NV_Ith_S(CVODE,CVODE_fcaf)))/tfcaf);
  GlobalData_t diff_fcafp = ((fcass-(NV_Ith_S(CVODE,CVODE_fcafp)))/tfcafp);
  GlobalData_t diff_fcas = ((fcass-(NV_Ith_S(CVODE,CVODE_fcas)))/tfcas);
  GlobalData_t diff_ffp = ((fss-(NV_Ith_S(CVODE,CVODE_ffp)))/tffp);
  GlobalData_t diff_h = ((hss-(NV_Ith_S(CVODE,CVODE_h)))/th);
  GlobalData_t diff_hp = ((hssp-(NV_Ith_S(CVODE,CVODE_hp)))/th);
  GlobalData_t diff_iF = ((iss-(NV_Ith_S(CVODE,CVODE_iF)))/tiF);
  GlobalData_t diff_iS = ((iss-(NV_Ith_S(CVODE,CVODE_iS)))/tiS);
  GlobalData_t diff_j = ((jss-(NV_Ith_S(CVODE,CVODE_j)))/tj);
  GlobalData_t diff_nca_i = ((anca_i*k2n)-((NV_Ith_S(CVODE,CVODE_nca_i)*km2n)));
  GlobalData_t diff_nca_ss = ((anca_ss*k2n)-((NV_Ith_S(CVODE,CVODE_nca_ss)*km2n)));
  GlobalData_t diff_xs2 = ((xs2ss-(NV_Ith_S(CVODE,CVODE_xs2)))/txs2);
  GlobalData_t fICaLp = (1./(1.+(KmCaMK/CaMKa)));
  GlobalData_t fINaLp = (1./(1.+(KmCaMK/CaMKa)));
  GlobalData_t fINap = (1./(1.+(KmCaMK/CaMKa)));
  GlobalData_t fItop = (1./(1.+(KmCaMK/CaMKa)));
  GlobalData_t fJrelp = (1./(1.+(KmCaMK/CaMKa)));
  GlobalData_t fJupp = (1./(1.+(KmCaMK/CaMKa)));
  GlobalData_t fca = ((Afcaf*NV_Ith_S(CVODE,CVODE_fcaf))+(Afcas*NV_Ith_S(CVODE,CVODE_fcas)));
  GlobalData_t fcap = ((Afcaf*NV_Ith_S(CVODE,CVODE_fcafp))+(Afcas*NV_Ith_S(CVODE,CVODE_fcas)));
  GlobalData_t h1_i = (1.+((NV_Ith_S(CVODE,CVODE_nai)/kna3)*(1.+hna)));
  GlobalData_t h1_ss = (1.+((NV_Ith_S(CVODE,CVODE_nass)/kna3)*(1.+hna)));
  GlobalData_t h7_i = (1.+((nao/kna3)*(1.+(1./hna))));
  GlobalData_t h7_ss = (1.+((nao/kna3)*(1.+(1./hna))));
  GlobalData_t i_gate = ((AiF*NV_Ith_S(CVODE,CVODE_iF))+(AiS*NV_Ith_S(CVODE,CVODE_iS)));
  GlobalData_t ip = ((AiF*NV_Ith_S(CVODE,CVODE_iFp))+(AiS*NV_Ith_S(CVODE,CVODE_iSp)));
  GlobalData_t k6_i = ((h6_i*NV_Ith_S(CVODE,CVODE_cai))*kcaon);
  GlobalData_t k6_ss = ((h6_ss*NV_Ith_S(CVODE,CVODE_cass))*kcaon);
  GlobalData_t tiFp = ((dti_develop*dti_recover)*tiF);
  GlobalData_t tiSp = ((dti_develop*dti_recover)*tiS);
  GlobalData_t tjp = (1.46*tj);
  GlobalData_t ICaK_i = ((1.-(ICaL_fractionSS))*((((((1.-(fICaLp))*PCaK)*PhiCaK_i)*NV_Ith_S(CVODE,CVODE_d))*((f*(1.-(NV_Ith_S(CVODE,CVODE_nca_i))))+((NV_Ith_S(CVODE,CVODE_jca)*fca)*NV_Ith_S(CVODE,CVODE_nca_i))))+((((fICaLp*PCaKp)*PhiCaK_i)*NV_Ith_S(CVODE,CVODE_d))*((fp*(1.-(NV_Ith_S(CVODE,CVODE_nca_i))))+((NV_Ith_S(CVODE,CVODE_jca)*fcap)*NV_Ith_S(CVODE,CVODE_nca_i))))));
  GlobalData_t ICaK_ss = (ICaL_fractionSS*((((((1.-(fICaLp))*PCaK)*PhiCaK_ss)*NV_Ith_S(CVODE,CVODE_d))*((f*(1.-(NV_Ith_S(CVODE,CVODE_nca_ss))))+((NV_Ith_S(CVODE,CVODE_jca)*fca)*NV_Ith_S(CVODE,CVODE_nca_ss))))+((((fICaLp*PCaKp)*PhiCaK_ss)*NV_Ith_S(CVODE,CVODE_d))*((fp*(1.-(NV_Ith_S(CVODE,CVODE_nca_ss))))+((NV_Ith_S(CVODE,CVODE_jca)*fcap)*NV_Ith_S(CVODE,CVODE_nca_ss))))));
  GlobalData_t ICaL_i = ((1.-(ICaL_fractionSS))*((((((1.-(fICaLp))*PCa)*PhiCaL_i)*NV_Ith_S(CVODE,CVODE_d))*((f*(1.-(NV_Ith_S(CVODE,CVODE_nca_i))))+((NV_Ith_S(CVODE,CVODE_jca)*fca)*NV_Ith_S(CVODE,CVODE_nca_i))))+((((fICaLp*PCap)*PhiCaL_i)*NV_Ith_S(CVODE,CVODE_d))*((fp*(1.-(NV_Ith_S(CVODE,CVODE_nca_i))))+((NV_Ith_S(CVODE,CVODE_jca)*fcap)*NV_Ith_S(CVODE,CVODE_nca_i))))));
  GlobalData_t ICaL_ss = (ICaL_fractionSS*((((((1.-(fICaLp))*PCa)*PhiCaL_ss)*NV_Ith_S(CVODE,CVODE_d))*((f*(1.-(NV_Ith_S(CVODE,CVODE_nca_ss))))+((NV_Ith_S(CVODE,CVODE_jca)*fca)*NV_Ith_S(CVODE,CVODE_nca_ss))))+((((fICaLp*PCap)*PhiCaL_ss)*NV_Ith_S(CVODE,CVODE_d))*((fp*(1.-(NV_Ith_S(CVODE,CVODE_nca_ss))))+((NV_Ith_S(CVODE,CVODE_jca)*fcap)*NV_Ith_S(CVODE,CVODE_nca_ss))))));
  GlobalData_t ICaNa_i = ((1.-(ICaL_fractionSS))*((((((1.-(fICaLp))*PCaNa)*PhiCaNa_i)*NV_Ith_S(CVODE,CVODE_d))*((f*(1.-(NV_Ith_S(CVODE,CVODE_nca_i))))+((NV_Ith_S(CVODE,CVODE_jca)*fca)*NV_Ith_S(CVODE,CVODE_nca_i))))+((((fICaLp*PCaNap)*PhiCaNa_i)*NV_Ith_S(CVODE,CVODE_d))*((fp*(1.-(NV_Ith_S(CVODE,CVODE_nca_i))))+((NV_Ith_S(CVODE,CVODE_jca)*fcap)*NV_Ith_S(CVODE,CVODE_nca_i))))));
  GlobalData_t ICaNa_ss = (ICaL_fractionSS*((((((1.-(fICaLp))*PCaNa)*PhiCaNa_ss)*NV_Ith_S(CVODE,CVODE_d))*((f*(1.-(NV_Ith_S(CVODE,CVODE_nca_ss))))+((NV_Ith_S(CVODE,CVODE_jca)*fca)*NV_Ith_S(CVODE,CVODE_nca_ss))))+((((fICaLp*PCaNap)*PhiCaNa_ss)*NV_Ith_S(CVODE,CVODE_d))*((fp*(1.-(NV_Ith_S(CVODE,CVODE_nca_ss))))+((NV_Ith_S(CVODE,CVODE_jca)*fcap)*NV_Ith_S(CVODE,CVODE_nca_ss))))));
  GlobalData_t IK1 = (((GK1*(sqrt((ko/5.))))*K1ss)*(nr->v-(EK)));
  GlobalData_t INa = (((GNa*(nr->v-(ENa)))*((NV_Ith_S(CVODE,CVODE_m)*NV_Ith_S(CVODE,CVODE_m))*NV_Ith_S(CVODE,CVODE_m)))*((((1.-(fINap))*NV_Ith_S(CVODE,CVODE_h))*NV_Ith_S(CVODE,CVODE_j))+((fINap*NV_Ith_S(CVODE,CVODE_hp))*NV_Ith_S(CVODE,CVODE_jp))));
  GlobalData_t INaL = (((GNaL*(nr->v-(ENa)))*NV_Ith_S(CVODE,CVODE_mL))*(((1.-(fINaLp))*NV_Ith_S(CVODE,CVODE_hL))+(fINaLp*NV_Ith_S(CVODE,CVODE_hLp))));
  GlobalData_t Ito = ((Gto*(nr->v-(EK)))*((((1.-(fItop))*NV_Ith_S(CVODE,CVODE_a))*i_gate)+((fItop*NV_Ith_S(CVODE,CVODE_ap))*ip)));
  GlobalData_t Jrel = (Jrel_b*(((1.-(fJrelp))*NV_Ith_S(CVODE,CVODE_Jrel_np))+(fJrelp*NV_Ith_S(CVODE,CVODE_Jrel_p))));
  GlobalData_t Jup = (Jup_b*((((1.-(fJupp))*Jupnp)+(fJupp*Jupp))-(Jleak)));
  GlobalData_t diff_C1 = ((((alpha_1*NV_Ith_S(CVODE,CVODE_C2))+(beta_2*NV_Ith_S(CVODE,CVODE_O)))+(beta_ItoC2*NV_Ith_S(CVODE,CVODE_I)))-((((beta_1+alpha_2)+alpha_C2ToI)*NV_Ith_S(CVODE,CVODE_C1))));
  GlobalData_t diff_I = (((alpha_C2ToI*NV_Ith_S(CVODE,CVODE_C1))+(alpha_i*NV_Ith_S(CVODE,CVODE_O)))-(((beta_ItoC2+beta_i)*NV_Ith_S(CVODE,CVODE_I))));
  GlobalData_t diff_iFp = ((iss-(NV_Ith_S(CVODE,CVODE_iFp)))/tiFp);
  GlobalData_t diff_iSp = ((iss-(NV_Ith_S(CVODE,CVODE_iSp)))/tiSp);
  GlobalData_t diff_jp = ((jss-(NV_Ith_S(CVODE,CVODE_jp)))/tjp);
  GlobalData_t h2_i = ((NV_Ith_S(CVODE,CVODE_nai)*hna)/(kna3*h1_i));
  GlobalData_t h2_ss = ((NV_Ith_S(CVODE,CVODE_nass)*hna)/(kna3*h1_ss));
  GlobalData_t h3_i = (1./h1_i);
  GlobalData_t h3_ss = (1./h1_ss);
  GlobalData_t h8_i = (nao/((kna3*hna)*h7_i));
  GlobalData_t h8_ss = (nao/((kna3*hna)*h7_ss));
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
  GlobalData_t Jrel_inf_b = (((-a_rel)*ICaL_ss)/(1.+((((((((cajsr_half/NV_Ith_S(CVODE,CVODE_cajsr))*(cajsr_half/NV_Ith_S(CVODE,CVODE_cajsr)))*(cajsr_half/NV_Ith_S(CVODE,CVODE_cajsr)))*(cajsr_half/NV_Ith_S(CVODE,CVODE_cajsr)))*(cajsr_half/NV_Ith_S(CVODE,CVODE_cajsr)))*(cajsr_half/NV_Ith_S(CVODE,CVODE_cajsr)))*(cajsr_half/NV_Ith_S(CVODE,CVODE_cajsr)))*(cajsr_half/NV_Ith_S(CVODE,CVODE_cajsr)))));
  GlobalData_t Jrel_infp_b = (((-a_relp)*ICaL_ss)/(1.+((((((((cajsr_half/NV_Ith_S(CVODE,CVODE_cajsr))*(cajsr_half/NV_Ith_S(CVODE,CVODE_cajsr)))*(cajsr_half/NV_Ith_S(CVODE,CVODE_cajsr)))*(cajsr_half/NV_Ith_S(CVODE,CVODE_cajsr)))*(cajsr_half/NV_Ith_S(CVODE,CVODE_cajsr)))*(cajsr_half/NV_Ith_S(CVODE,CVODE_cajsr)))*(cajsr_half/NV_Ith_S(CVODE,CVODE_cajsr)))*(cajsr_half/NV_Ith_S(CVODE,CVODE_cajsr)))));
  GlobalData_t diff_cajsr = (Bcajsr*(Jtr-(Jrel)));
  GlobalData_t diff_cansr = (Jup-(((Jtr*vjsr)/vnsr)));
  GlobalData_t diff_kss = ((((-ICaK_ss)*Acap)/(F*vss))-(JdiffK));
  GlobalData_t k3p_i = (h9_i*wca);
  GlobalData_t k3p_ss = (h9_ss*wca);
  GlobalData_t k3pp_i = (h8_i*wnaca);
  GlobalData_t k3pp_ss = (h8_ss*wnaca);
  GlobalData_t k4p_i = ((h3_i*wca)/hca);
  GlobalData_t k4p_ss = ((h3_ss*wca)/hca);
  GlobalData_t k4pp_i = (h2_i*wnaca);
  GlobalData_t k4pp_ss = (h2_ss*wnaca);
  GlobalData_t k7_i = ((h5_i*h2_i)*wna);
  GlobalData_t k7_ss = ((h5_ss*h2_ss)*wna);
  GlobalData_t k8_i = ((h8_i*h11_i)*wna);
  GlobalData_t k8_ss = ((h8_ss*h11_ss)*wna);
  GlobalData_t JnakK = (2.*((E4*b1)-((E3*a1))));
  GlobalData_t JnakNa = (3.*((E1*a3)-((E2*b3))));
  GlobalData_t Jrel_inf = ((celltype==2.) ? (Jrel_inf_b*1.7) : Jrel_inf_b);
  GlobalData_t Jrel_infp = ((celltype==2.) ? (Jrel_infp_b*1.7) : Jrel_infp_b);
  GlobalData_t k3_i = (k3p_i+k3pp_i);
  GlobalData_t k3_ss = (k3p_ss+k3pp_ss);
  GlobalData_t k4_i = (k4p_i+k4pp_i);
  GlobalData_t k4_ss = (k4p_ss+k4pp_ss);
  GlobalData_t INaK = (Pnak*((zna*JnakNa)+(zk*JnakK)));
  GlobalData_t diff_Jrel_np = ((Jrel_inf-(NV_Ith_S(CVODE,CVODE_Jrel_np)))/tau_rel);
  GlobalData_t diff_Jrel_p = ((Jrel_infp-(NV_Ith_S(CVODE,CVODE_Jrel_p)))/tau_relp);
  GlobalData_t x1_i = (((k2_i*k4_i)*(k7_i+k6_i))+((k5_i*k7_i)*(k2_i+k3_i)));
  GlobalData_t x1_ss = (((k2_ss*k4_ss)*(k7_ss+k6_ss))+((k5_ss*k7_ss)*(k2_ss+k3_ss)));
  GlobalData_t x2_i = (((k1_i*k7_i)*(k4_i+k5_i))+((k4_i*k6_i)*(k1_i+k8_i)));
  GlobalData_t x2_ss = (((k1_ss*k7_ss)*(k4_ss+k5_ss))+((k4_ss*k6_ss)*(k1_ss+k8_ss)));
  GlobalData_t x3_i = (((k1_i*k3_i)*(k7_i+k6_i))+((k8_i*k6_i)*(k2_i+k3_i)));
  GlobalData_t x3_ss = (((k1_ss*k3_ss)*(k7_ss+k6_ss))+((k8_ss*k6_ss)*(k2_ss+k3_ss)));
  GlobalData_t x4_i = (((k2_i*k8_i)*(k4_i+k5_i))+((k3_i*k5_i)*(k1_i+k8_i)));
  GlobalData_t x4_ss = (((k2_ss*k8_ss)*(k4_ss+k5_ss))+((k3_ss*k5_ss)*(k1_ss+k8_ss)));
  GlobalData_t E1_i = (x1_i/(((x1_i+x2_i)+x3_i)+x4_i));
  GlobalData_t E1_ss = (x1_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
  GlobalData_t E2_i = (x2_i/(((x1_i+x2_i)+x3_i)+x4_i));
  GlobalData_t E2_ss = (x2_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
  GlobalData_t E3_i = (x3_i/(((x1_i+x2_i)+x3_i)+x4_i));
  GlobalData_t E3_ss = (x3_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
  GlobalData_t E4_i = (x4_i/(((x1_i+x2_i)+x3_i)+x4_i));
  GlobalData_t E4_ss = (x4_ss/(((x1_ss+x2_ss)+x3_ss)+x4_ss));
  GlobalData_t diff_ki = ((((-(((((((Ito+IKr)+IKs)+IK1)+IKb)+I_katp)-((2.*INaK)))+ICaK_i))*Acap)/(F*vmyo))+((JdiffK*vss)/vmyo));
  GlobalData_t JncxCa_i = ((E2_i*k2_i)-((E1_i*k1_i)));
  GlobalData_t JncxCa_ss = ((E2_ss*k2_ss)-((E1_ss*k1_ss)));
  GlobalData_t JncxNa_i = (((3.*((E4_i*k7_i)-((E1_i*k8_i))))+(E3_i*k4pp_i))-((E2_i*k3pp_i)));
  GlobalData_t JncxNa_ss = (((3.*((E4_ss*k7_ss)-((E1_ss*k8_ss))))+(E3_ss*k4pp_ss))-((E2_ss*k3pp_ss)));
  GlobalData_t INaCa_i = ((((1.-(INaCa_fractionSS))*Gncx)*allo_i)*((zna*JncxNa_i)+(zca*JncxCa_i)));
  GlobalData_t INaCa_ss = (((INaCa_fractionSS*Gncx)*allo_ss)*((zna*JncxNa_ss)+(zca*JncxCa_ss)));
  GlobalData_t diff_cai = (Bcai*(((((-(((ICaL_i+IpCa)+ICab)-((2.*INaCa_i))))*Acap)/((2.*F)*vmyo))-(((Jup*vnsr)/vmyo)))+((Jdiff*vss)/vmyo)));
  GlobalData_t diff_cass = (Bcass*(((((-(ICaL_ss-((2.*INaCa_ss))))*Acap)/((2.*F)*vss))+((Jrel*vjsr)/vss))-(Jdiff)));
  GlobalData_t diff_nai = ((((-(((((INa+INaL)+(3.*INaCa_i))+ICaNa_i)+(3.*INaK))+INab))*Acap)/(F*vmyo))+((JdiffNa*vss)/vmyo));
  GlobalData_t diff_nass = ((((-(ICaNa_ss+(3.*INaCa_ss)))*Acap)/(F*vss))-(JdiffNa));
  
  
  //Complete the update
  NV_Ith_S(CVODE_dot,CVODE_C1) = diff_C1;
  NV_Ith_S(CVODE_dot,CVODE_C2) = diff_C2;
  NV_Ith_S(CVODE_dot,CVODE_C3) = diff_C3;
  NV_Ith_S(CVODE_dot,CVODE_CaMKt) = diff_CaMKt;
  NV_Ith_S(CVODE_dot,CVODE_I) = diff_I;
  NV_Ith_S(CVODE_dot,CVODE_Jrel_np) = diff_Jrel_np;
  NV_Ith_S(CVODE_dot,CVODE_Jrel_p) = diff_Jrel_p;
  NV_Ith_S(CVODE_dot,CVODE_O) = diff_O;
  NV_Ith_S(CVODE_dot,CVODE_a) = diff_a;
  NV_Ith_S(CVODE_dot,CVODE_ap) = diff_ap;
  NV_Ith_S(CVODE_dot,CVODE_cai) = diff_cai;
  NV_Ith_S(CVODE_dot,CVODE_cajsr) = diff_cajsr;
  NV_Ith_S(CVODE_dot,CVODE_cansr) = diff_cansr;
  NV_Ith_S(CVODE_dot,CVODE_cass) = diff_cass;
  NV_Ith_S(CVODE_dot,CVODE_d) = diff_d;
  NV_Ith_S(CVODE_dot,CVODE_fcaf) = diff_fcaf;
  NV_Ith_S(CVODE_dot,CVODE_fcafp) = diff_fcafp;
  NV_Ith_S(CVODE_dot,CVODE_fcas) = diff_fcas;
  NV_Ith_S(CVODE_dot,CVODE_ff) = diff_ff;
  NV_Ith_S(CVODE_dot,CVODE_ffp) = diff_ffp;
  NV_Ith_S(CVODE_dot,CVODE_fs) = diff_fs;
  NV_Ith_S(CVODE_dot,CVODE_h) = diff_h;
  NV_Ith_S(CVODE_dot,CVODE_hL) = diff_hL;
  NV_Ith_S(CVODE_dot,CVODE_hLp) = diff_hLp;
  NV_Ith_S(CVODE_dot,CVODE_hp) = diff_hp;
  NV_Ith_S(CVODE_dot,CVODE_iF) = diff_iF;
  NV_Ith_S(CVODE_dot,CVODE_iFp) = diff_iFp;
  NV_Ith_S(CVODE_dot,CVODE_iS) = diff_iS;
  NV_Ith_S(CVODE_dot,CVODE_iSp) = diff_iSp;
  NV_Ith_S(CVODE_dot,CVODE_j) = diff_j;
  NV_Ith_S(CVODE_dot,CVODE_jca) = diff_jca;
  NV_Ith_S(CVODE_dot,CVODE_jp) = diff_jp;
  NV_Ith_S(CVODE_dot,CVODE_ki) = diff_ki;
  NV_Ith_S(CVODE_dot,CVODE_kss) = diff_kss;
  NV_Ith_S(CVODE_dot,CVODE_m) = diff_m;
  NV_Ith_S(CVODE_dot,CVODE_mL) = diff_mL;
  NV_Ith_S(CVODE_dot,CVODE_nai) = diff_nai;
  NV_Ith_S(CVODE_dot,CVODE_nass) = diff_nass;
  NV_Ith_S(CVODE_dot,CVODE_nca_i) = diff_nca_i;
  NV_Ith_S(CVODE_dot,CVODE_nca_ss) = diff_nca_ss;
  NV_Ith_S(CVODE_dot,CVODE_xs1) = diff_xs1;
  NV_Ith_S(CVODE_dot,CVODE_xs2) = diff_xs2;

  return 0;
}
#endif

bool ToRORd_fkatp_epiIonType::has_trace() const {
    return false;
}

void ToRORd_fkatp_epiIonType::trace(IonIfBase& imp_base, int node, FILE* file, GlobalData_t** data) const {}
IonIfBase* ToRORd_fkatp_epiIonType::make_ion_if(Target target, int num_node, const std::vector<std::reference_wrapper<IonType>>& plugins) const {
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

void ToRORd_fkatp_epiIonType::destroy_ion_if(IonIfBase *imp) const {
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
        