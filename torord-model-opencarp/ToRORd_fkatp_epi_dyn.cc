
#ifdef _AIX
#include <stdlib.h>
#endif
#include <string.h>
#include <dlfcn.h>

#include "ION_IF.h"
#include "MULTI_ION_IF.h"

namespace limpet {
using ::opencarp::dupstr;
extern opencarp::FILE_SPEC _nc_logf;
}  // namespace limpet


////////////////////////////////////////////////

#include "ToRORd_fkatp_epi.h"



namespace limpet {

void ToRORd_fkatp_epiIonType::tune(IonIfBase& iif_base, const char* im_par) const {
  IonIfDerived& iif = static_cast<IonIfDerived&>(iif_base);
  ToRORd_fkatp_epi_Params *p;							// pointer to parameter structure
  char parameter[1024], mod[1024];

  p = iif.params();

  // make sure flags is a valid member and flags specified are valid
  char *npar, *par, *parptr;
  par=tokstr_r(npar = dupstr(im_par), ",", &parptr);
  while( par ) {
    if( !strncmp(par, "flags=", 6 ) ) {


      log_msg( _nc_logf, 5, 0, "Unrecognized parameter: flags\n" );
      exit(1);


    }
    par = tokstr_r( NULL, ",", &parptr );
  }
  free(npar);

  // now process the regular parameters
  par=tokstr_r(npar = dupstr(im_par), ",", &parptr);
  while( par ) {
    process_param_mod( par, parameter, mod );
    if (0) ;


    else if( !strcmp( "flags", parameter ) )
      ;
    else {
      log_msg( _nc_logf, 5, 0,"Unrecognized parameter: %s.\n",parameter);
      log_msg( _nc_logf, 5, 0,"Run bench --imp=YourModel --imp-info to get a list of all parameters.\n",parameter);
      exit(1);
    }
    par = tokstr_r( NULL, ",", &parptr );
  }
  free( npar );
}




/** output all parameters which may be tuned for all IMPs
 */
void ToRORd_fkatp_epiIonType::print_params() const {
  IonIfDerived IF{*this, Target::AUTO, 0, {}};
  IF.initialize_params();
  printf("Name: ToRORd_fkatp_epi\n" );
  printf("\tParameters:\n" );


}



int ToRORd_fkatp_epiIonType::write_svs(IonIfBase& IF_base, FILE *out, int node) const {
  IonIfDerived& IF = static_cast<IonIfDerived&>(IF_base);
  fprintf( out, "%s\n", IF.get_type().get_name().c_str() );

  int inner_id = node % this->dlo_vector_size();

  ToRORd_fkatp_epi_state *sv = IF.sv_tab().data()+(node / (this->dlo_vector_size()));

  fprintf( out, "%-20g# C1\n", sv->C1 );

  fprintf( out, "%-20g# C2\n", sv->C2 );

  fprintf( out, "%-20g# C3\n", sv->C3 );

  fprintf( out, "%-20g# CaMKt\n", sv->CaMKt );

  fprintf( out, "%-20g# I\n", sv->I );

  fprintf( out, "%-20g# Jrel_np\n", sv->Jrel_np );

  fprintf( out, "%-20g# Jrel_p\n", sv->Jrel_p );

  fprintf( out, "%-20g# O\n", sv->O );

  fprintf( out, "%-20g# a\n", sv->a );

  fprintf( out, "%-20g# ap\n", sv->ap );

  fprintf( out, "%-20g# cai\n", sv->cai );

  fprintf( out, "%-20g# cajsr\n", sv->cajsr );

  fprintf( out, "%-20g# cansr\n", sv->cansr );

  fprintf( out, "%-20g# cass\n", sv->cass );

  fprintf( out, "%-20g# d\n", sv->d );

  fprintf( out, "%-20g# fcaf\n", sv->fcaf );

  fprintf( out, "%-20g# fcafp\n", sv->fcafp );

  fprintf( out, "%-20g# fcas\n", sv->fcas );

  fprintf( out, "%-20g# ff\n", sv->ff );

  fprintf( out, "%-20g# ffp\n", sv->ffp );

  fprintf( out, "%-20g# fs\n", sv->fs );

  fprintf( out, "%-20g# h\n", sv->h );

  fprintf( out, "%-20g# hL\n", sv->hL );

  fprintf( out, "%-20g# hLp\n", sv->hLp );

  fprintf( out, "%-20g# hp\n", sv->hp );

  fprintf( out, "%-20g# iF\n", sv->iF );

  fprintf( out, "%-20g# iFp\n", sv->iFp );

  fprintf( out, "%-20g# iS\n", sv->iS );

  fprintf( out, "%-20g# iSp\n", sv->iSp );

  fprintf( out, "%-20g# j\n", sv->j );

  fprintf( out, "%-20g# jca\n", sv->jca );

  fprintf( out, "%-20g# jp\n", sv->jp );

  fprintf( out, "%-20g# ki\n", sv->ki );

  fprintf( out, "%-20g# kss\n", sv->kss );

  fprintf( out, "%-20g# m\n", sv->m );

  fprintf( out, "%-20g# mL\n", sv->mL );

  fprintf( out, "%-20g# nai\n", sv->nai );

  fprintf( out, "%-20g# nass\n", sv->nass );

  fprintf( out, "%-20g# nca_i\n", sv->nca_i );

  fprintf( out, "%-20g# nca_ss\n", sv->nca_ss );

  fprintf( out, "%-20g# xs1\n", sv->xs1 );

  fprintf( out, "%-20g# xs2\n", sv->xs2 );


  fprintf( out, "\n");
  return 0;
}



int ToRORd_fkatp_epiIonType::read_svs(IonIfBase& IF_base, FILE *in) const {
  IonIfDerived& IF = static_cast<IonIfDerived&>(IF_base);
  const int  BUFSIZE=256;
  char       impname[256], buf[BUFSIZE];
  const char *gdt_sc = sizeof(GlobalData_t)==sizeof(float)?"%f":"%lf";
  int flg = 0;
  
  // skip possible empty lines
  do {
    if( fgets(buf,BUFSIZE,in)==NULL ) {
      log_msg( _nc_logf, 4, 0, "no state information for IMP: %s\n", IF.get_type().get_name().c_str() );
      return 2;
    }
  } while( *buf=='\n' );
  sscanf( buf, "%s", impname );

  if( strcmp( impname, IF.get_type().get_name().c_str() ) ) {
    log_msg( _nc_logf, 5, 0, "IMPs do not match region (%s vs %s). Skipping this statefile.\n",impname, IF.get_type().get_name().c_str());
    return 2;
  }


  if(!IF.get_num_node())  {
    flg = 1;
  }
  ToRORd_fkatp_epi_state *sv = IF.sv_tab().data();

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->C1 );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->C2 );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->C3 );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->CaMKt );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->I );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->Jrel_np );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->Jrel_p );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->O );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->a );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->ap );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->cai );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->cajsr );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->cansr );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->cass );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->d );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->fcaf );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->fcafp );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->fcas );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->ff );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->ffp );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->fs );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->h );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->hL );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->hLp );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->hp );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->iF );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->iFp );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->iS );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->iSp );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->j );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->jca );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->jp );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->ki );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->kss );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->m );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->mL );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->nai );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->nass );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->nca_i );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->nca_ss );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->xs1 );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->xs2 );


  return flg;
}



SVgetfcn ToRORd_fkatp_epiIonType::get_sv_offset(const char *svname, int *off, int *sz) const {
  SVgetfcn retall = (SVgetfcn)(1);


        ToRORd_fkatp_epi_state *sv;

        if( !strcmp(svname,"ALL_SV") )  {

          *off  = 0;

          *sz   = sizeof(ToRORd_fkatp_epi_state);

          return retall;

        }

        if( !strcmp(svname,"C1") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,C1);

          *sz   = sizeof  (sv->C1) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"C2") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,C2);

          *sz   = sizeof  (sv->C2) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"C3") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,C3);

          *sz   = sizeof  (sv->C3) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"CaMKt") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,CaMKt);

          *sz   = sizeof  (sv->CaMKt) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"I") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,I);

          *sz   = sizeof  (sv->I) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"Jrel_np") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,Jrel_np);

          *sz   = sizeof  (sv->Jrel_np) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"Jrel_p") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,Jrel_p);

          *sz   = sizeof  (sv->Jrel_p) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"O") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,O);

          *sz   = sizeof  (sv->O) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"a") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,a);

          *sz   = sizeof  (sv->a) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"ap") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,ap);

          *sz   = sizeof  (sv->ap) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"cai") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,cai);

          *sz   = sizeof  (sv->cai) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"cajsr") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,cajsr);

          *sz   = sizeof  (sv->cajsr) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"cansr") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,cansr);

          *sz   = sizeof  (sv->cansr) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"cass") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,cass);

          *sz   = sizeof  (sv->cass) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"d") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,d);

          *sz   = sizeof  (sv->d) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"fcaf") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,fcaf);

          *sz   = sizeof  (sv->fcaf) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"fcafp") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,fcafp);

          *sz   = sizeof  (sv->fcafp) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"fcas") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,fcas);

          *sz   = sizeof  (sv->fcas) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"ff") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,ff);

          *sz   = sizeof  (sv->ff) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"ffp") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,ffp);

          *sz   = sizeof  (sv->ffp) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"fs") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,fs);

          *sz   = sizeof  (sv->fs) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"h") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,h);

          *sz   = sizeof  (sv->h) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"hL") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,hL);

          *sz   = sizeof  (sv->hL) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"hLp") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,hLp);

          *sz   = sizeof  (sv->hLp) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"hp") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,hp);

          *sz   = sizeof  (sv->hp) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"iF") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,iF);

          *sz   = sizeof  (sv->iF) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"iFp") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,iFp);

          *sz   = sizeof  (sv->iFp) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"iS") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,iS);

          *sz   = sizeof  (sv->iS) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"iSp") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,iSp);

          *sz   = sizeof  (sv->iSp) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"j") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,j);

          *sz   = sizeof  (sv->j) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"jca") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,jca);

          *sz   = sizeof  (sv->jca) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"jp") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,jp);

          *sz   = sizeof  (sv->jp) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"ki") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,ki);

          *sz   = sizeof  (sv->ki) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"kss") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,kss);

          *sz   = sizeof  (sv->kss) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"m") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,m);

          *sz   = sizeof  (sv->m) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"mL") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,mL);

          *sz   = sizeof  (sv->mL) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"nai") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,nai);

          *sz   = sizeof  (sv->nai) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"nass") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,nass);

          *sz   = sizeof  (sv->nass) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"nca_i") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,nca_i);

          *sz   = sizeof  (sv->nca_i) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"nca_ss") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,nca_ss);

          *sz   = sizeof  (sv->nca_ss) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"xs1") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,xs1);

          *sz   = sizeof  (sv->xs1) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"xs2") )  {

          *off  = offsetof(ToRORd_fkatp_epi_state,xs2);

          *sz   = sizeof  (sv->xs2) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }


  return NULL;
}



int ToRORd_fkatp_epiIonType::get_sv_list(char*** list) const {

  *list = (char**)malloc( sizeof(char*)*42 );

  (*list)[0] = dupstr("C1");

  (*list)[1] = dupstr("C2");

  (*list)[2] = dupstr("C3");

  (*list)[3] = dupstr("CaMKt");

  (*list)[4] = dupstr("I");

  (*list)[5] = dupstr("Jrel_np");

  (*list)[6] = dupstr("Jrel_p");

  (*list)[7] = dupstr("O");

  (*list)[8] = dupstr("a");

  (*list)[9] = dupstr("ap");

  (*list)[10] = dupstr("cai");

  (*list)[11] = dupstr("cajsr");

  (*list)[12] = dupstr("cansr");

  (*list)[13] = dupstr("cass");

  (*list)[14] = dupstr("d");

  (*list)[15] = dupstr("fcaf");

  (*list)[16] = dupstr("fcafp");

  (*list)[17] = dupstr("fcas");

  (*list)[18] = dupstr("ff");

  (*list)[19] = dupstr("ffp");

  (*list)[20] = dupstr("fs");

  (*list)[21] = dupstr("h");

  (*list)[22] = dupstr("hL");

  (*list)[23] = dupstr("hLp");

  (*list)[24] = dupstr("hp");

  (*list)[25] = dupstr("iF");

  (*list)[26] = dupstr("iFp");

  (*list)[27] = dupstr("iS");

  (*list)[28] = dupstr("iSp");

  (*list)[29] = dupstr("j");

  (*list)[30] = dupstr("jca");

  (*list)[31] = dupstr("jp");

  (*list)[32] = dupstr("ki");

  (*list)[33] = dupstr("kss");

  (*list)[34] = dupstr("m");

  (*list)[35] = dupstr("mL");

  (*list)[36] = dupstr("nai");

  (*list)[37] = dupstr("nass");

  (*list)[38] = dupstr("nca_i");

  (*list)[39] = dupstr("nca_ss");

  (*list)[40] = dupstr("xs1");

  (*list)[41] = dupstr("xs2");

  return 42;


}



#define BOGUSTYPE -1
int ToRORd_fkatp_epiIonType::get_sv_type(const char *svname, int *type, char **Typename) const
{
  *type = BOGUSTYPE;
  if (0) ;

  else if( !strcmp(svname,"C1") )
 *type = 7;

  else if( !strcmp(svname,"C2") )
 *type = 7;

  else if( !strcmp(svname,"C3") )
 *type = 7;

  else if( !strcmp(svname,"CaMKt") )
 *type = 7;

  else if( !strcmp(svname,"I") )
 *type = 7;

  else if( !strcmp(svname,"Jrel_np") )
 *type = 7;

  else if( !strcmp(svname,"Jrel_p") )
 *type = 7;

  else if( !strcmp(svname,"O") )
 *type = 7;

  else if( !strcmp(svname,"a") )
 *type = 7;

  else if( !strcmp(svname,"ap") )
 *type = 7;

  else if( !strcmp(svname,"cai") )
 *type = 7;

  else if( !strcmp(svname,"cajsr") )
 *type = 7;

  else if( !strcmp(svname,"cansr") )
 *type = 7;

  else if( !strcmp(svname,"cass") )
 *type = 7;

  else if( !strcmp(svname,"d") )
 *type = 7;

  else if( !strcmp(svname,"fcaf") )
 *type = 7;

  else if( !strcmp(svname,"fcafp") )
 *type = 7;

  else if( !strcmp(svname,"fcas") )
 *type = 7;

  else if( !strcmp(svname,"ff") )
 *type = 7;

  else if( !strcmp(svname,"ffp") )
 *type = 7;

  else if( !strcmp(svname,"fs") )
 *type = 7;

  else if( !strcmp(svname,"h") )
 *type = 7;

  else if( !strcmp(svname,"hL") )
 *type = 7;

  else if( !strcmp(svname,"hLp") )
 *type = 7;

  else if( !strcmp(svname,"hp") )
 *type = 7;

  else if( !strcmp(svname,"iF") )
 *type = 7;

  else if( !strcmp(svname,"iFp") )
 *type = 7;

  else if( !strcmp(svname,"iS") )
 *type = 7;

  else if( !strcmp(svname,"iSp") )
 *type = 7;

  else if( !strcmp(svname,"j") )
 *type = 7;

  else if( !strcmp(svname,"jca") )
 *type = 7;

  else if( !strcmp(svname,"jp") )
 *type = 7;

  else if( !strcmp(svname,"ki") )
 *type = 7;

  else if( !strcmp(svname,"kss") )
 *type = 7;

  else if( !strcmp(svname,"m") )
 *type = 7;

  else if( !strcmp(svname,"mL") )
 *type = 7;

  else if( !strcmp(svname,"nai") )
 *type = 7;

  else if( !strcmp(svname,"nass") )
 *type = 7;

  else if( !strcmp(svname,"nca_i") )
 *type = 7;

  else if( !strcmp(svname,"nca_ss") )
 *type = 7;

  else if( !strcmp(svname,"xs1") )
 *type = 7;

  else if( !strcmp(svname,"xs2") )
 *type = 7;


  else return 0;
  *Typename = get_typename(*type);
  return 1;
}




void ToRORd_fkatp_epiIonType::print_metadata() const {
  printf("Metadata:\n");
  printf("\t\n");
  printf("\t\n");
  printf("\t\n");
  printf("\t\n");
  printf("\t\n");
  printf("\t\n");
  printf("\n");
}

}  // namespace limpet

        

extern "C" {
    limpet::ToRORd_fkatp_epiIonType* __new_IonType(bool plugin) {
              return new limpet::ToRORd_fkatp_epiIonType(plugin);
    }
}
    
