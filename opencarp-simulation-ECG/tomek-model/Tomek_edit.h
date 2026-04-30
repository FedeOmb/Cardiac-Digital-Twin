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
        
//// HEADER GUARD ///////////////////////////
// If automatically generated, keep above
// comment as first line in file.
#ifndef __TOMEK_EDIT_H__
#define __TOMEK_EDIT_H__
//// HEADER GUARD ///////////////////////////
// DO NOT EDIT THIS SOURCE CODE FILE
// ANY CHANGES TO THIS FILE WILL BE OVERWRITTEN!!!!

#include "ION_IF.h"

#if !(defined(TOMEK_EDIT_CPU_GENERATED)    || defined(TOMEK_EDIT_MLIR_CPU_GENERATED)    || defined(TOMEK_EDIT_MLIR_ROCM_GENERATED)    || defined(TOMEK_EDIT_MLIR_CUDA_GENERATED))
#ifdef MLIR_CPU_GENERATED
#define TOMEK_EDIT_MLIR_CPU_GENERATED
#endif

#ifdef MLIR_ROCM_GENERATED
#define TOMEK_EDIT_MLIR_ROCM_GENERATED
#endif

#ifdef MLIR_CUDA_GENERATED
#define TOMEK_EDIT_MLIR_CUDA_GENERATED
#endif
#endif

#ifdef CPU_GENERATED
#define TOMEK_EDIT_CPU_GENERATED
#endif

namespace limpet {

#define Tomek_edit_REQDAT Vm_DATA_FLAG
#define Tomek_edit_MODDAT Iion_DATA_FLAG

struct Tomek_edit_Params {
    GlobalData_t CaMKo;
    GlobalData_t Cajsr_half;
    GlobalData_t GK1_b;
    GlobalData_t GKr_b;
    GlobalData_t GNaL_b;
    GlobalData_t Gncx_b;
    GlobalData_t Gto_b;
    GlobalData_t Jrel_b;
    GlobalData_t Jup_b;
    GlobalData_t PNaK_b;
    GlobalData_t thL;
    
    // UNMODIFIABLE BELOW HERE
    GlobalData_t celltype;
    char* flags;

};
static const char* Tomek_edit_flags = "ENDO|EPI|MCELL";


struct Tomek_edit_state {
    GlobalData_t C1;
    GlobalData_t C2;
    GlobalData_t C3;
    GlobalData_t CaMKt;
    GlobalData_t Cai;
    GlobalData_t Cajsr;
    GlobalData_t Cansr;
    GlobalData_t Cass;
    GlobalData_t I;
    GlobalData_t Jrel_np;
    GlobalData_t Jrel_p;
    GlobalData_t Ki;
    GlobalData_t Kss;
    GlobalData_t Nai;
    GlobalData_t Nass;
    GlobalData_t O;
    GlobalData_t a;
    GlobalData_t ap;
    GlobalData_t d;
    GlobalData_t fCaf;
    GlobalData_t fCafp;
    GlobalData_t fCas;
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
    GlobalData_t jCa;
    GlobalData_t jp;
    GlobalData_t m;
    GlobalData_t mL;
    GlobalData_t nCa_i;
    GlobalData_t nCa_ss;
    GlobalData_t xs1;
    GlobalData_t xs2;

};

class Tomek_editIonType : public IonType {
public:
    using IonIfDerived = IonIf<Tomek_editIonType>;
    using params_type = Tomek_edit_Params;
    using state_type = Tomek_edit_state;

    Tomek_editIonType(bool plugin);

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

//void compute_Tomek_edit(int, int, IonIfBase&, GlobalData_t**);
#ifdef TOMEK_EDIT_CPU_GENERATED
void compute_Tomek_edit_cpu(int, int, IonIfBase&, GlobalData_t**);
#endif
#ifdef TOMEK_EDIT_MLIR_CPU_GENERATED
void compute_Tomek_edit_mlir_cpu(int, int, IonIfBase&, GlobalData_t**);
#endif
#ifdef TOMEK_EDIT_MLIR_ROCM_GENERATED
void compute_Tomek_edit_mlir_gpu_rocm(int, int, IonIfBase&, GlobalData_t**);
#endif
#ifdef TOMEK_EDIT_MLIR_CUDA_GENERATED
void compute_Tomek_edit_mlir_gpu_cuda(int, int, IonIfBase&, GlobalData_t**);
#endif

}
}  // namespace limpet

//// HEADER GUARD ///////////////////////////
#endif
//// HEADER GUARD ///////////////////////////
