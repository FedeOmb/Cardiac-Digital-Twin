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
        
//// HEADER GUARD ///////////////////////////
// If automatically generated, keep above
// comment as first line in file.
#ifndef __TORORD_FKATP_ENDO_MID_H__
#define __TORORD_FKATP_ENDO_MID_H__
//// HEADER GUARD ///////////////////////////
// DO NOT EDIT THIS SOURCE CODE FILE
// ANY CHANGES TO THIS FILE WILL BE OVERWRITTEN!!!!

#include "ION_IF.h"

#if !(defined(TORORD_FKATP_ENDO_MID_CPU_GENERATED)    || defined(TORORD_FKATP_ENDO_MID_MLIR_CPU_GENERATED)    || defined(TORORD_FKATP_ENDO_MID_MLIR_ROCM_GENERATED)    || defined(TORORD_FKATP_ENDO_MID_MLIR_CUDA_GENERATED))
#ifdef MLIR_CPU_GENERATED
#define TORORD_FKATP_ENDO_MID_MLIR_CPU_GENERATED
#endif

#ifdef MLIR_ROCM_GENERATED
#define TORORD_FKATP_ENDO_MID_MLIR_ROCM_GENERATED
#endif

#ifdef MLIR_CUDA_GENERATED
#define TORORD_FKATP_ENDO_MID_MLIR_CUDA_GENERATED
#endif
#endif

#ifdef CPU_GENERATED
#define TORORD_FKATP_ENDO_MID_CPU_GENERATED
#endif

namespace limpet {

#define ToRORd_fkatp_endo_mid_REQDAT Vm_DATA_FLAG
#define ToRORd_fkatp_endo_mid_MODDAT Iion_DATA_FLAG

struct ToRORd_fkatp_endo_mid_Params {

};

struct ToRORd_fkatp_endo_mid_state {
    GlobalData_t C1;
    GlobalData_t C2;
    GlobalData_t C3;
    GlobalData_t CaMKt;
    GlobalData_t I;
    GlobalData_t Jrel_np;
    GlobalData_t Jrel_p;
    GlobalData_t O;
    GlobalData_t a;
    GlobalData_t ap;
    GlobalData_t cai;
    GlobalData_t cajsr;
    GlobalData_t cansr;
    GlobalData_t cass;
    GlobalData_t d;
    GlobalData_t fcaf;
    GlobalData_t fcafp;
    GlobalData_t fcas;
    GlobalData_t ff;
    GlobalData_t ffp;
    GlobalData_t fs;
    GlobalData_t h;
    GlobalData_t hL;
    GlobalData_t hLp;
    GlobalData_t hp;
    GlobalData_t iF;
    GlobalData_t iFp;
    GlobalData_t iS;
    GlobalData_t iSp;
    GlobalData_t j;
    GlobalData_t jca;
    GlobalData_t jp;
    GlobalData_t ki;
    GlobalData_t kss;
    GlobalData_t m;
    GlobalData_t mL;
    GlobalData_t nai;
    GlobalData_t nass;
    GlobalData_t nca_i;
    GlobalData_t nca_ss;
    GlobalData_t xs1;
    GlobalData_t xs2;

};

    struct ToRORd_fkatp_endo_mid_Regional_Constants_cpu {
                    
    char dummy;

    };

    struct ToRORd_fkatp_endo_mid_Nodal_Req_cpu {
    
    GlobalData_t v;

    };

    struct ToRORd_fkatp_endo_mid_Private_cpu {
      using nodal_req_type = ToRORd_fkatp_endo_mid_Nodal_Req_cpu;
      using regional_constants_type = ToRORd_fkatp_endo_mid_Regional_Constants_cpu;
      IonIfBase *IF;
      int   node_number;
      void* cvode_mem;
      void* sunctx_ptr;
      bool  trace_init;
      regional_constants_type rc;
      nodal_req_type nr;
    };
    

//Printing out the Rosenbrock declarations
extern "C" {

void ToRORd_fkatp_endo_mid_rosenbrock_f_cpu(float*, float*, void*);

void ToRORd_fkatp_endo_mid_rosenbrock_jacobian_cpu(float**, float*, void*, int);

void rbStepX ( float *X,
                void (*calcDX) (float*,  float*, void*),
                void (*calcJ)  (float**, float*, void*, int ),
                void *params, float h, int N );
}

class ToRORd_fkatp_endo_midIonType : public IonType {
public:
    using IonIfDerived = IonIf<ToRORd_fkatp_endo_midIonType>;
    using params_type = ToRORd_fkatp_endo_mid_Params;
    using state_type = ToRORd_fkatp_endo_mid_state;

    using private_type = ToRORd_fkatp_endo_mid_Private_cpu;
    #ifdef TORORD_FKATP_ENDO_MID_MLIR_CPU_GENERATED
    using private_type_vector = ToRORd_fkatp_endo_mid_Private_mlir_cpu;
    #endif

    ToRORd_fkatp_endo_midIonType(bool plugin);

    size_t params_size() const override;

    size_t dlo_vector_size() const override;

    uint32_t reqdat() const override;

    uint32_t moddat() const override;

    void initialize_params(IonIfBase& imp_base) const override;

    void construct_tables(IonIfBase& imp_base) const override;

    void destroy(IonIfBase& imp_base) const override;

    void initialize_sv(IonIfBase& imp_base, GlobalData_t** data) const override;

    Target select_target(Target target) const override;

    void compute(Target target, int start, int end, IonIfBase& imp_base, GlobalData_t** data) const override;

    bool has_trace() const override;

    void trace(IonIfBase& imp_base, int node, FILE* file, GlobalData_t** data) const override;

    void tune(IonIfBase& imp_base, const char* im_par) const override;

    int read_svs(IonIfBase& imp_base, FILE* file) const override;

    int write_svs(IonIfBase& imp_base, FILE* file, int node) const override;

    SVgetfcn get_sv_offset(const char* svname, int* off, int* sz) const override;

    int get_sv_list(char*** list) const override;

    int get_sv_type(const char* svname, int* type, char** type_name) const override;

    void print_params() const override;

    void print_metadata() const override;

    IonIfBase* make_ion_if(Target target, int num_node, const std::vector<std::reference_wrapper<IonType>>& plugins) const override;

    void destroy_ion_if(IonIfBase *imp) const override;
};

// This needs to be extern C in order to be linked correctly with the MLIR code
extern "C" {

//void compute_ToRORd_fkatp_endo_mid(int, int, IonIfBase&, GlobalData_t**);
#ifdef TORORD_FKATP_ENDO_MID_CPU_GENERATED
void compute_ToRORd_fkatp_endo_mid_cpu(int, int, IonIfBase&, GlobalData_t**);
#endif
#ifdef TORORD_FKATP_ENDO_MID_MLIR_CPU_GENERATED
void compute_ToRORd_fkatp_endo_mid_mlir_cpu(int, int, IonIfBase&, GlobalData_t**);
#endif
#ifdef TORORD_FKATP_ENDO_MID_MLIR_ROCM_GENERATED
void compute_ToRORd_fkatp_endo_mid_mlir_gpu_rocm(int, int, IonIfBase&, GlobalData_t**);
#endif
#ifdef TORORD_FKATP_ENDO_MID_MLIR_CUDA_GENERATED
void compute_ToRORd_fkatp_endo_mid_mlir_gpu_cuda(int, int, IonIfBase&, GlobalData_t**);
#endif

}
}  // namespace limpet

//// HEADER GUARD ///////////////////////////
#endif
//// HEADER GUARD ///////////////////////////
