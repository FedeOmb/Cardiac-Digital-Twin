
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

#include "Tomek_editv2.h"



namespace limpet {

void Tomek_editv2IonType::tune(IonIfBase& iif_base, const char* im_par) const {
  IonIfDerived& iif = static_cast<IonIfDerived&>(iif_base);
  Tomek_editv2_Params *p;							// pointer to parameter structure
  char parameter[1024], mod[1024];

  p = iif.params();

  // make sure flags is a valid member and flags specified are valid
  char *npar, *par, *parptr;
  par=tokstr_r(npar = dupstr(im_par), ",", &parptr);
  while( par ) {
    if( !strncmp(par, "flags=", 6 ) ) {


      bool valid_flags = verify_flags(Tomek_editv2_flags, par+6);
      if( !valid_flags ) {
        log_msg(_nc_logf, 5, 0, "Illegal flag specified: %s\n", par );
        exit( 1 );
      }
      ((Tomek_editv2_Params *)p)->flags = dupstr(par+6);
      iif.initialize_params();


    }
    par = tokstr_r( NULL, ",", &parptr );
  }
  free(npar);

  // now process the regular parameters
  par=tokstr_r(npar = dupstr(im_par), ",", &parptr);
  while( par ) {
    process_param_mod( par, parameter, mod );
    if (0) ;

    else if( !strcmp( "CaMKo", parameter ) )

      CHANGE_PARAM( Tomek_editv2, p, CaMKo, mod );

    else if( !strcmp( "Cajsr_half", parameter ) )

      CHANGE_PARAM( Tomek_editv2, p, Cajsr_half, mod );

    else if( !strcmp( "GK1_b", parameter ) )

      CHANGE_PARAM( Tomek_editv2, p, GK1_b, mod );

    else if( !strcmp( "GKr_b", parameter ) )

      CHANGE_PARAM( Tomek_editv2, p, GKr_b, mod );

    else if( !strcmp( "GKs_b", parameter ) )

      CHANGE_PARAM( Tomek_editv2, p, GKs_b, mod );

    else if( !strcmp( "GNaL_b", parameter ) )

      CHANGE_PARAM( Tomek_editv2, p, GNaL_b, mod );

    else if( !strcmp( "Gncx_b", parameter ) )

      CHANGE_PARAM( Tomek_editv2, p, Gncx_b, mod );

    else if( !strcmp( "Gto_b", parameter ) )

      CHANGE_PARAM( Tomek_editv2, p, Gto_b, mod );

    else if( !strcmp( "Jrel_b", parameter ) )

      CHANGE_PARAM( Tomek_editv2, p, Jrel_b, mod );

    else if( !strcmp( "Jup_b", parameter ) )

      CHANGE_PARAM( Tomek_editv2, p, Jup_b, mod );

    else if( !strcmp( "PNaK_b", parameter ) )

      CHANGE_PARAM( Tomek_editv2, p, PNaK_b, mod );

    else if( !strcmp( "thL", parameter ) )

      CHANGE_PARAM( Tomek_editv2, p, thL, mod );

    else if( !strcmp( "tjCa", parameter ) )

      CHANGE_PARAM( Tomek_editv2, p, tjCa, mod );


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
void Tomek_editv2IonType::print_params() const {
  IonIfDerived IF{*this, Target::AUTO, 0, {}};
  IF.initialize_params();
  printf("Name: Tomek_editv2\n" );
  printf("\tParameters:\n" );

  printf( "\t%32s\t%g\n","CaMKo", IF.params()->CaMKo );

  printf( "\t%32s\t%g\n","Cajsr_half", IF.params()->Cajsr_half );

  printf( "\t%32s\t%g\n","GK1_b", IF.params()->GK1_b );

  printf( "\t%32s\t%g\n","GKr_b", IF.params()->GKr_b );

  printf( "\t%32s\t%g\n","GKs_b", IF.params()->GKs_b );

  printf( "\t%32s\t%g\n","GNaL_b", IF.params()->GNaL_b );

  printf( "\t%32s\t%g\n","Gncx_b", IF.params()->Gncx_b );

  printf( "\t%32s\t%g\n","Gto_b", IF.params()->Gto_b );

  printf( "\t%32s\t%g\n","Jrel_b", IF.params()->Jrel_b );

  printf( "\t%32s\t%g\n","Jup_b", IF.params()->Jup_b );

  printf( "\t%32s\t%g\n","PNaK_b", IF.params()->PNaK_b );

  printf( "\t%32s\t%g\n","thL", IF.params()->thL );

  printf( "\t%32s\t%g\n","tjCa", IF.params()->tjCa );

  printf( "\t%32s\t%s\n","flags", "ENDO|EPI|MCELL" );


}



int Tomek_editv2IonType::write_svs(IonIfBase& IF_base, FILE *out, int node) const {
  IonIfDerived& IF = static_cast<IonIfDerived&>(IF_base);
  fprintf( out, "%s\n", IF.get_type().get_name().c_str() );

  int inner_id = node % this->dlo_vector_size();

  Tomek_editv2_state *sv = IF.sv_tab().data()+(node / (this->dlo_vector_size()));

  fprintf( out, "%-20g# C1\n", sv->C1 );

  fprintf( out, "%-20g# C2\n", sv->C2 );

  fprintf( out, "%-20g# C3\n", sv->C3 );

  fprintf( out, "%-20g# CaMKt\n", sv->CaMKt );

  fprintf( out, "%-20g# Cai\n", sv->Cai );

  fprintf( out, "%-20g# Cajsr\n", sv->Cajsr );

  fprintf( out, "%-20g# Cansr\n", sv->Cansr );

  fprintf( out, "%-20g# Cass\n", sv->Cass );

  fprintf( out, "%-20g# GKs_b\n", sv->GKs_b );

  fprintf( out, "%-20g# I\n", sv->I );

  fprintf( out, "%-20g# Jrel_np\n", sv->Jrel_np );

  fprintf( out, "%-20g# Jrel_p\n", sv->Jrel_p );

  fprintf( out, "%-20g# Ki\n", sv->Ki );

  fprintf( out, "%-20g# Kss\n", sv->Kss );

  fprintf( out, "%-20g# Nai\n", sv->Nai );

  fprintf( out, "%-20g# Nass\n", sv->Nass );

  fprintf( out, "%-20g# O\n", sv->O );

  fprintf( out, "%-20g# a\n", sv->a );

  fprintf( out, "%-20g# ap\n", sv->ap );

  fprintf( out, "%-20g# d\n", sv->d );

  fprintf( out, "%-20g# fCaf\n", sv->fCaf );

  fprintf( out, "%-20g# fCafp\n", sv->fCafp );

  fprintf( out, "%-20g# fCas\n", sv->fCas );

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

  fprintf( out, "%-20g# jCa\n", sv->jCa );

  fprintf( out, "%-20g# jp\n", sv->jp );

  fprintf( out, "%-20g# m\n", sv->m );

  fprintf( out, "%-20g# mL\n", sv->mL );

  fprintf( out, "%-20g# nCa_i\n", sv->nCa_i );

  fprintf( out, "%-20g# nCa_ss\n", sv->nCa_ss );

  fprintf( out, "%-20g# xs1\n", sv->xs1 );

  fprintf( out, "%-20g# xs2\n", sv->xs2 );


  fprintf( out, "\n");
  return 0;
}



int Tomek_editv2IonType::read_svs(IonIfBase& IF_base, FILE *in) const {
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
  Tomek_editv2_state *sv = IF.sv_tab().data();

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->C1 );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->C2 );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->C3 );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->CaMKt );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->Cai );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->Cajsr );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->Cansr );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->Cass );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->GKs_b );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->I );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->Jrel_np );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->Jrel_p );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->Ki );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->Kss );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->Nai );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->Nass );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->O );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->a );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->ap );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->d );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->fCaf );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->fCafp );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->fCas );

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

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->jCa );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->jp );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->m );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->mL );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->nCa_i );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->nCa_ss );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->xs1 );

  sscanf( fgets(buf,BUFSIZE,in), gdt_sc, &sv->xs2 );


  return flg;
}



SVgetfcn Tomek_editv2IonType::get_sv_offset(const char *svname, int *off, int *sz) const {
  SVgetfcn retall = (SVgetfcn)(1);


        Tomek_editv2_state *sv;

        if( !strcmp(svname,"ALL_SV") )  {

          *off  = 0;

          *sz   = sizeof(Tomek_editv2_state);

          return retall;

        }

        if( !strcmp(svname,"C1") )  {

          *off  = offsetof(Tomek_editv2_state,C1);

          *sz   = sizeof  (sv->C1) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"C2") )  {

          *off  = offsetof(Tomek_editv2_state,C2);

          *sz   = sizeof  (sv->C2) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"C3") )  {

          *off  = offsetof(Tomek_editv2_state,C3);

          *sz   = sizeof  (sv->C3) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"CaMKt") )  {

          *off  = offsetof(Tomek_editv2_state,CaMKt);

          *sz   = sizeof  (sv->CaMKt) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"Cai") )  {

          *off  = offsetof(Tomek_editv2_state,Cai);

          *sz   = sizeof  (sv->Cai) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"Cajsr") )  {

          *off  = offsetof(Tomek_editv2_state,Cajsr);

          *sz   = sizeof  (sv->Cajsr) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"Cansr") )  {

          *off  = offsetof(Tomek_editv2_state,Cansr);

          *sz   = sizeof  (sv->Cansr) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"Cass") )  {

          *off  = offsetof(Tomek_editv2_state,Cass);

          *sz   = sizeof  (sv->Cass) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"GKs_b") )  {

          *off  = offsetof(Tomek_editv2_state,GKs_b);

          *sz   = sizeof  (sv->GKs_b) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"I") )  {

          *off  = offsetof(Tomek_editv2_state,I);

          *sz   = sizeof  (sv->I) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"Jrel_np") )  {

          *off  = offsetof(Tomek_editv2_state,Jrel_np);

          *sz   = sizeof  (sv->Jrel_np) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"Jrel_p") )  {

          *off  = offsetof(Tomek_editv2_state,Jrel_p);

          *sz   = sizeof  (sv->Jrel_p) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"Ki") )  {

          *off  = offsetof(Tomek_editv2_state,Ki);

          *sz   = sizeof  (sv->Ki) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"Kss") )  {

          *off  = offsetof(Tomek_editv2_state,Kss);

          *sz   = sizeof  (sv->Kss) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"Nai") )  {

          *off  = offsetof(Tomek_editv2_state,Nai);

          *sz   = sizeof  (sv->Nai) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"Nass") )  {

          *off  = offsetof(Tomek_editv2_state,Nass);

          *sz   = sizeof  (sv->Nass) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"O") )  {

          *off  = offsetof(Tomek_editv2_state,O);

          *sz   = sizeof  (sv->O) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"a") )  {

          *off  = offsetof(Tomek_editv2_state,a);

          *sz   = sizeof  (sv->a) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"ap") )  {

          *off  = offsetof(Tomek_editv2_state,ap);

          *sz   = sizeof  (sv->ap) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"d") )  {

          *off  = offsetof(Tomek_editv2_state,d);

          *sz   = sizeof  (sv->d) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"fCaf") )  {

          *off  = offsetof(Tomek_editv2_state,fCaf);

          *sz   = sizeof  (sv->fCaf) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"fCafp") )  {

          *off  = offsetof(Tomek_editv2_state,fCafp);

          *sz   = sizeof  (sv->fCafp) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"fCas") )  {

          *off  = offsetof(Tomek_editv2_state,fCas);

          *sz   = sizeof  (sv->fCas) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"ff") )  {

          *off  = offsetof(Tomek_editv2_state,ff);

          *sz   = sizeof  (sv->ff) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"ffp") )  {

          *off  = offsetof(Tomek_editv2_state,ffp);

          *sz   = sizeof  (sv->ffp) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"fs") )  {

          *off  = offsetof(Tomek_editv2_state,fs);

          *sz   = sizeof  (sv->fs) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"h") )  {

          *off  = offsetof(Tomek_editv2_state,h);

          *sz   = sizeof  (sv->h) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"hL") )  {

          *off  = offsetof(Tomek_editv2_state,hL);

          *sz   = sizeof  (sv->hL) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"hLp") )  {

          *off  = offsetof(Tomek_editv2_state,hLp);

          *sz   = sizeof  (sv->hLp) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"hp") )  {

          *off  = offsetof(Tomek_editv2_state,hp);

          *sz   = sizeof  (sv->hp) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"iF") )  {

          *off  = offsetof(Tomek_editv2_state,iF);

          *sz   = sizeof  (sv->iF) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"iFp") )  {

          *off  = offsetof(Tomek_editv2_state,iFp);

          *sz   = sizeof  (sv->iFp) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"iS") )  {

          *off  = offsetof(Tomek_editv2_state,iS);

          *sz   = sizeof  (sv->iS) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"iSp") )  {

          *off  = offsetof(Tomek_editv2_state,iSp);

          *sz   = sizeof  (sv->iSp) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"j") )  {

          *off  = offsetof(Tomek_editv2_state,j);

          *sz   = sizeof  (sv->j) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"jCa") )  {

          *off  = offsetof(Tomek_editv2_state,jCa);

          *sz   = sizeof  (sv->jCa) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"jp") )  {

          *off  = offsetof(Tomek_editv2_state,jp);

          *sz   = sizeof  (sv->jp) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"m") )  {

          *off  = offsetof(Tomek_editv2_state,m);

          *sz   = sizeof  (sv->m) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"mL") )  {

          *off  = offsetof(Tomek_editv2_state,mL);

          *sz   = sizeof  (sv->mL) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"nCa_i") )  {

          *off  = offsetof(Tomek_editv2_state,nCa_i);

          *sz   = sizeof  (sv->nCa_i) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"nCa_ss") )  {

          *off  = offsetof(Tomek_editv2_state,nCa_ss);

          *sz   = sizeof  (sv->nCa_ss) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"xs1") )  {

          *off  = offsetof(Tomek_editv2_state,xs1);

          *sz   = sizeof  (sv->xs1) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }

        if( !strcmp(svname,"xs2") )  {

          *off  = offsetof(Tomek_editv2_state,xs2);

          *sz   = sizeof  (sv->xs2) / this->dlo_vector_size();

          return getGlobalData_tSV;

        }


  return NULL;
}



int Tomek_editv2IonType::get_sv_list(char*** list) const {

  *list = (char**)malloc( sizeof(char*)*43 );

  (*list)[0] = dupstr("C1");

  (*list)[1] = dupstr("C2");

  (*list)[2] = dupstr("C3");

  (*list)[3] = dupstr("CaMKt");

  (*list)[4] = dupstr("Cai");

  (*list)[5] = dupstr("Cajsr");

  (*list)[6] = dupstr("Cansr");

  (*list)[7] = dupstr("Cass");

  (*list)[8] = dupstr("GKs_b");

  (*list)[9] = dupstr("I");

  (*list)[10] = dupstr("Jrel_np");

  (*list)[11] = dupstr("Jrel_p");

  (*list)[12] = dupstr("Ki");

  (*list)[13] = dupstr("Kss");

  (*list)[14] = dupstr("Nai");

  (*list)[15] = dupstr("Nass");

  (*list)[16] = dupstr("O");

  (*list)[17] = dupstr("a");

  (*list)[18] = dupstr("ap");

  (*list)[19] = dupstr("d");

  (*list)[20] = dupstr("fCaf");

  (*list)[21] = dupstr("fCafp");

  (*list)[22] = dupstr("fCas");

  (*list)[23] = dupstr("ff");

  (*list)[24] = dupstr("ffp");

  (*list)[25] = dupstr("fs");

  (*list)[26] = dupstr("h");

  (*list)[27] = dupstr("hL");

  (*list)[28] = dupstr("hLp");

  (*list)[29] = dupstr("hp");

  (*list)[30] = dupstr("iF");

  (*list)[31] = dupstr("iFp");

  (*list)[32] = dupstr("iS");

  (*list)[33] = dupstr("iSp");

  (*list)[34] = dupstr("j");

  (*list)[35] = dupstr("jCa");

  (*list)[36] = dupstr("jp");

  (*list)[37] = dupstr("m");

  (*list)[38] = dupstr("mL");

  (*list)[39] = dupstr("nCa_i");

  (*list)[40] = dupstr("nCa_ss");

  (*list)[41] = dupstr("xs1");

  (*list)[42] = dupstr("xs2");

  return 43;


}



#define BOGUSTYPE -1
int Tomek_editv2IonType::get_sv_type(const char *svname, int *type, char **Typename) const
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

  else if( !strcmp(svname,"Cai") )
 *type = 7;

  else if( !strcmp(svname,"Cajsr") )
 *type = 7;

  else if( !strcmp(svname,"Cansr") )
 *type = 7;

  else if( !strcmp(svname,"Cass") )
 *type = 7;

  else if( !strcmp(svname,"GKs_b") )
 *type = 7;

  else if( !strcmp(svname,"I") )
 *type = 7;

  else if( !strcmp(svname,"Jrel_np") )
 *type = 7;

  else if( !strcmp(svname,"Jrel_p") )
 *type = 7;

  else if( !strcmp(svname,"Ki") )
 *type = 7;

  else if( !strcmp(svname,"Kss") )
 *type = 7;

  else if( !strcmp(svname,"Nai") )
 *type = 7;

  else if( !strcmp(svname,"Nass") )
 *type = 7;

  else if( !strcmp(svname,"O") )
 *type = 7;

  else if( !strcmp(svname,"a") )
 *type = 7;

  else if( !strcmp(svname,"ap") )
 *type = 7;

  else if( !strcmp(svname,"d") )
 *type = 7;

  else if( !strcmp(svname,"fCaf") )
 *type = 7;

  else if( !strcmp(svname,"fCafp") )
 *type = 7;

  else if( !strcmp(svname,"fCas") )
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

  else if( !strcmp(svname,"jCa") )
 *type = 7;

  else if( !strcmp(svname,"jp") )
 *type = 7;

  else if( !strcmp(svname,"m") )
 *type = 7;

  else if( !strcmp(svname,"mL") )
 *type = 7;

  else if( !strcmp(svname,"nCa_i") )
 *type = 7;

  else if( !strcmp(svname,"nCa_ss") )
 *type = 7;

  else if( !strcmp(svname,"xs1") )
 *type = 7;

  else if( !strcmp(svname,"xs2") )
 *type = 7;


  else return 0;
  *Typename = get_typename(*type);
  return 1;
}




void Tomek_editv2IonType::print_metadata() const {
  printf("Metadata:\n");
  printf("\tAuthors: Jakub Tomek, Alfonso Bueno-Orovio, Elisa Passini, Xin Zhou, Ana Minchole, Oliver Britton, Chiara Bartolucci, Stefano Severi, Alvin Shrier, Laszlo Virag, Andras Varro, Blanca Rodriguez\n");
  printf("\tYear: 2019\n");
  printf("\tTitle: Development, calibration, and validation of a novel human ventricular myocyte model in health, disease, and drug block\n");
  printf("\tJournal: eLife\n");
  printf("\tDOI: 10.7554/eLife.48890\n");
  printf("\tComment: \n");
  printf("\n");
}

}  // namespace limpet

        

extern "C" {
    limpet::Tomek_editv2IonType* __new_IonType(bool plugin) {
              return new limpet::Tomek_editv2IonType(plugin);
    }
}
    
