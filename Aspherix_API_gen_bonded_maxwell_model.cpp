// include the header file which defines the class
#include "bond_max.h"

// includes from the standard library
#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#include <fstream>
#include <iostream>

using namespace Aspherix_API;

CohesionBondMax::CohesionBondMax(Aspherix *asx, std::string id) :
    // call the parent constructor
    ContactModelExternal(asx, id),

    // Bond parameters
    time_step_create_bond_(-1),
    max_dist_index_(0),

    Y(0),
    v(0),
    cf(0),
    cr(0),
    k1(0),
    k2(0),
    k3(0),
    k4(0),
    kk(0),
    c1(0),
    c2(0),
    c3(0),
    c4(0),
    ns_ratio(1),
    lambda(1),
    sigma_max(0),
    tau_max(0)
{
    connectToAspherix();
}

// this member function is responsible for adding history values for the model
// history values are values that are stored on a per-interaction basis
// history values can be either vectors or values (scalars)
// the 1 and 0 indicate whether the sign of the value/vector should change in case the order of the two particles in the interaction is switched, i.e.
// if the second argument equals to 1 for value v, then v_ij = - v_ji (e.g. the normal vector)
// if the second arguemnt equals to 0 for value v, then v_ij = v_ji   (e.g. the overlap between two particles)
void CohesionBondMax::addHistoryValues()
{
   // Hertz tangential history
   addHistoryVector("shear_hz", 1);

   // Bond history values
    addHistoryValue("bond_exists", 0);
    addHistoryValue("bond_broken", 0);

    addHistoryVector("r_0", 1); // Initial length
    addHistoryVector("r_old", 1); // Previous timestep length

  // Generalized Maxwell
    addHistoryVector("f_n1", 1); // Force on normal element 1
    addHistoryVector("f_n2", 1); // Force on normal element 2
    addHistoryVector("f_n3", 1); // Force on normal element 3
    addHistoryVector("f_n4", 1); // Force on normal element 4
    addHistoryVector("f_s1", 1); // Force on shear element 1
    addHistoryVector("f_s2", 1); // Force on shear element 2
    addHistoryVector("f_s3", 1); // Force on shear element 3
    addHistoryVector("f_s4", 1); // Force on shear element 4
}

// this member function is responsible for connecting the contact model to material or global properties
void CohesionBondMax::registerGlobalProperties()
{
    // properties for the Hertz and Tangential history models
    registerScalarProperty("youngs_modulus_part");
    registerScalarProperty("poissons_ratio_part");
    registerScalarProperty("coeff_friction_part");
    registerScalarProperty("coeff_restitution_part");

    // Generalized Maxwell parameters
    registerScalarProperty("k1");
    registerScalarProperty("k2");
    registerScalarProperty("k3");
    registerScalarProperty("k4");
    registerScalarProperty("kk");
    registerScalarProperty("c1");
    registerScalarProperty("c2");
    registerScalarProperty("c3");
    registerScalarProperty("c4");
    registerScalarProperty("ns_ratio");

    // Bond model properties
    // scalar value
    registerScalarProperty("time_step_create_bond");
    registerScalarProperty("radius_multiplier");
    registerScalarProperty("limit_sigma");
    registerScalarProperty("limit_tau");
    // material interaction values
    max_dist_index_ = registerPairProperty("maximum_distance");
}

// this member function determines the maximum distance between two particles that still requires a contact model interaction
double CohesionBondMax::maximumParticleDistance(const int itype, const int jtype)
{
    return getPairProperty(max_dist_index_, itype, jtype);
}

// member function that is executed directly before the loop over all particle interactions
// can be used to initialize variables from the registry and others
void CohesionBondMax::beginPass(const GlobalProperties &properties)
{
    time_step_create_bond_ = static_cast<long long>(getScalarProperty("time_step_create_bond"));
    Y = static_cast<double>(getScalarProperty("youngs_modulus_part"));
    v = static_cast<double>(getScalarProperty("poissons_ratio_part"));
    cf = static_cast<double>(getScalarProperty("coeff_friction_part"));
    cr = static_cast<double>(getScalarProperty("coeff_restitution_part"));
    k1 = static_cast<double>(getScalarProperty("k1"));
    k2 = static_cast<double>(getScalarProperty("k2"));
    k3 = static_cast<double>(getScalarProperty("k3"));
    k4 = static_cast<double>(getScalarProperty("k4"));
    kk = static_cast<double>(getScalarProperty("kk"));
    c1 = static_cast<double>(getScalarProperty("c1"));
    c2 = static_cast<double>(getScalarProperty("c2"));
    c3 = static_cast<double>(getScalarProperty("c3"));
    c4 = static_cast<double>(getScalarProperty("c4"));
    ns_ratio = static_cast<double>(getScalarProperty("ns_ratio"));
    lambda = static_cast<double>(getScalarProperty("radius_multiplier"));
    sigma_max = static_cast<double>(getScalarProperty("limit_sigma"));
    tau_max = static_cast<double>(getScalarProperty("limit_tau"));
}

// member function to deal with two particles that are overlapping
void CohesionBondMax::surfacesIntersect(Particle &pi, Particle &pj, ParticleInteraction &pij, const GlobalProperties &properties)
{
    fullModel(pi, pj, pij, properties, true);
}

// member functions to deal with two particles that are in each others vicinity (at most maximumParticleDistance apart)
void CohesionBondMax::surfacesClose(Particle &pi, Particle &pj, ParticleInteraction &pij, const GlobalProperties &properties)
{
    fullModel(pi, pj, pij, properties, false);
}

// this function is specific to this case and determines which contact models we call
void CohesionBondMax::fullModel(Particle &pi, Particle &pj, ParticleInteraction &pij, const GlobalProperties &properties, const bool do_hertz)
{
//    const double bond_exists = pij.getHistoryValue("bond_exists");
    Vector total_force;

    // in case particles overlap and there is no bond the normal model Hertz and tangential model Tangential history is called
    if (do_hertz)
    {
        modelHertzHistory(pi, pj, pij, properties, total_force);
    }
    // in any case the bond model is called after
    modelBond(pi, pj, pij, properties, total_force);

    // set the force for both particles (with opposite signs)
    pij.setForce(-total_force);
}

void CohesionBondMax::modelHertzHistory(Particle &pi, Particle &pj, ParticleInteraction &pij, const GlobalProperties &properties, Vector &force)
{
    // Computing the Hertzian contact force
    const double bond_exists = pij.getHistoryValue("bond_exists");

    // Condition to trigger Hertzian calculation
    if (bond_exists > 0.5 || properties.getTimeStep() <= time_step_create_bond_)
        return;

    const bool is_wall = pij.isWall();

    const double ri = pi.getRadius();
    const double rj = is_wall? 0. : pj.getRadius();

    const double reff = is_wall ? ri : ri*rj/(ri+rj);
   
    const double meff = pij.getMeff();
    const double radsuminv = 1./(ri+rj);

    const double dt = properties.getDt();
    const double deltan = pij.getDelta();
    const double sqrtval = sqrt(reff*deltan);
    const double distance = pij.getDistance();

    const int itype = pi.getType();
    const int jtype = pj.getType();

    const double Yeff = Y/(2.*(1.-v*v));
    const double Geff = Y/(4.*(2.-v)*(1.+v));
    const double betaeff = log(cr)/sqrt(log(cr)*log(cr)+M_PI*M_PI);

    const double Sn = 2.*Yeff*sqrtval;
    const double St = 8.*Geff*sqrtval;

    const double kn = 4./3.*Yeff*sqrtval;
    const double kt = St;
    const double sqrtFiveOverSix = 0.91287092917527685576161630466800355658790782499663875;
    const double gamman = -2.*sqrtFiveOverSix*betaeff*sqrt(Sn*meff);
    const double gammat = -2.*sqrtFiveOverSix*betaeff*sqrt(St*meff);

    const double fn_damping = -gamman*pij.getNormalVelocity();
    const double fn_contact = kn*deltan;

    const double fn = fn_damping + fn_contact;
    const Vector en = pij.getNormal();
    force += -en*fn;

   // Tangential history
   Vector vr = pij.getRelativeVelocity();
   Vector vn = vr.projectOn(en);
   Vector vt = vr - vn;

   Vector weighted_wr = is_wall ? pi.getAngularVelocity()*0.5 : (pi.getAngularVelocity()*ri + pj.getAngularVelocity()*rj)*radsuminv;

   //relative tangential velocity with rotation correction
   Vector vtr = vt + en.cross(weighted_wr)*distance;

   //old shear vector
   Vector shear = pij.getHistoryVector("shear_hz");

   //shear update
   shear += vtr*dt;
   const double rsht = shear*en;
   shear -= en*rsht;

   const double shearmag = shear.length();
   const double xmu = cf;
   Vector Ft_ela = -shear*kt;
   Vector Ft = Ft_ela;

   const double Ft_shear = kt*shearmag;
   const double Ft_friction = xmu*fabs(fn);

   if (Ft_shear > Ft_friction)
   {
       if (shearmag != 0.)
       {
           const double ratio = Ft_friction/Ft_shear;
           Ft *= ratio;
           shear = -Ft/kt;
       }
       else
           Ft = 0;
   }
   else
       Ft -= vt*gammat;

   force += Ft;

   pij.setHistoryVector("shear_hz", shear);

}

void CohesionBondMax::modelBond(Particle &pi, Particle &pj, ParticleInteraction &pij, const GlobalProperties &properties, Vector &force)
{
    // Computing the Bond forces
    const bool is_wall = pij.isWall();

    // Only particle-particle interaction is allowed for the bonds
    if (is_wall)
        return;

    double bond_broken = pij.getHistoryValue("bond_broken");

    double bond_exists = pij.getHistoryValue("bond_exists");
    
    const double distance = pij.getDistance();
    const double dt = properties.getDt();

    const int itype = pi.getType();
    const int jtype = pj.getType();

    const double ri = pi.getRadius();
    const double rj = pj.getRadius();

    Vector r_0 = pij.getHistoryVector("r_0");
    Vector r_old = pij.getHistoryVector("r_old");

    const double d_0 = r_0.length();
    const double d_old = r_old.length();

    Vector n = pij.getNormal();

    // check if bond does not yet exist
    if (bond_exists < 0.5)
    {
        if (bond_broken > 0.5)
            return;

        else
        {
            const bool create_now = (time_step_create_bond_ < 0 || properties.getTimeStep() == time_step_create_bond_);
            const double distance_create = lambda*(ri+rj);
            const bool is_close = distance_create > distance;

            //if bond did not exist, did not break, timestep is the creation time and the increased radii overlap than create a bond
            if (create_now && is_close)
            {
                createBond(pij, pi, pj);
                return;
            }
            else
                return;
        }
    }

    //loading history
    Vector f_n1 = pij.getHistoryVector("f_n1");
    Vector f_n2 = pij.getHistoryVector("f_n2");
    Vector f_n3 = pij.getHistoryVector("f_n3");
    Vector f_n4 = pij.getHistoryVector("f_n4");
    Vector f_s1 = pij.getHistoryVector("f_s1");
    Vector f_s2 = pij.getHistoryVector("f_s2");
    Vector f_s3 = pij.getHistoryVector("f_s3");
    Vector f_s4 = pij.getHistoryVector("f_s4");

    //Computing bond properties
    const double rb = std::min(ri, rj);
    const double Ab = M_PI*rb*rb;
    const double J = 0.5*Ab*rb*rb;
    const double I = 0.5*J;


    //Calculating micro parameters
    const double k1_n = Ab/d_0*k1;
    const double k2_n = Ab/d_0*k2;
    const double k3_n = Ab/d_0*k3;
    const double k4_n = Ab/d_0*k4;
    const double kk_n = Ab/d_0*kk;
    const double c1_n = Ab/d_0*c1;
    const double c2_n = Ab/d_0*c2;
    const double c3_n = Ab/d_0*c3;
    const double c4_n = Ab/d_0*c4;

    const double k1_s = k1_n/ns_ratio;
    const double k2_s = k2_n/ns_ratio;
    const double k3_s = k3_n/ns_ratio;
    const double k4_s = k4_n/ns_ratio;
    const double kk_s = kk_n/ns_ratio;
    const double c1_s = c1_n/ns_ratio;
    const double c2_s = c2_n/ns_ratio;
    const double c3_s = c3_n/ns_ratio;
    const double c4_s = c4_n/ns_ratio;

    // Semi-projecting old timestep forces onto new relative normal direction
    const double fold_n1 = f_n1*n;
    const double fold_n2 = f_n2*n;
    const double fold_n3 = f_n3*n;
    const double fold_n4 = f_n4*n;

    // Computing normal displacement
    Vector n_old = r_old/d_old;
    Vector n_0 = r_0/d_0;

    Vector r = n*distance;
    Vector delta_u = r - r_old;
    Vector delta_u0 = r - r_0;

    const double dn = n*delta_u;

    //Computing new scalar normal bond forces from discretization of consitutive differential equation
    const double fnew_n1 = (dn/dt + fold_n1*(1./(k1_n*dt)-0.5/c1_n))/(1./(k1_n*dt)+0.5/c1_n);
    const double fnew_n2 = (dn/dt + fold_n2*(1./(k2_n*dt)-0.5/c2_n))/(1./(k2_n*dt)+0.5/c2_n);
    const double fnew_n3 = (dn/dt + fold_n3*(1./(k3_n*dt)-0.5/c3_n))/(1./(k3_n*dt)+0.5/c3_n);
    const double fnew_n4 = (dn/dt + fold_n4*(1./(k4_n*dt)-0.5/c4_n))/(1./(k4_n*dt)+0.5/c4_n);

    // Computing shear displacement
    Vector shear = n.cross(n.cross(n_0));

    Vector s;

    const double s_abs = shear.length();

    if (s_abs > 1.e-20*d_0)
    {
        s = shear/s_abs;
    }
    else
    {
        s = {0.,0.,0.};
    }

    const double ds = s*delta_u;
    const double ds0 = s*delta_u0;

    // Semi-projecting old timestep forces onto new shear direction
    const double fold_s1 = f_s1*s;
    const double fold_s2 = f_s2*s;
    const double fold_s3 = f_s3*s;
    const double fold_s4 = f_s4*s;

    //Computing new scalar shear bond forces from discretization of consitutive differential equation
    const double fnew_s1 = (ds/dt + fold_s1*(1./(k1_s*dt)-0.5/c1_s))/(1./(k1_s*dt)+0.5/c1_s);
    const double fnew_s2 = (ds/dt + fold_s2*(1./(k2_s*dt)-0.5/c2_s))/(1./(k2_s*dt)+0.5/c2_s);
    const double fnew_s3 = (ds/dt + fold_s3*(1./(k3_s*dt)-0.5/c3_s))/(1./(k3_s*dt)+0.5/c3_s);
    const double fnew_s4 = (ds/dt + fold_s4*(1./(k4_s*dt)-0.5/c4_s))/(1./(k4_s*dt)+0.5/c4_s);

    // Full projection of normal forces onto new normal and shear directions
    f_n1 = n*fnew_n1;
    f_n2 = n*fnew_n2;
    f_n3 = n*fnew_n3;
    f_n4 = n*fnew_n4;
    f_s1 = s*fnew_s1;
    f_s2 = s*fnew_s2;
    f_s3 = s*fnew_s3;
    f_s4 = s*fnew_s4;

    // Computing single spring element forces
    Vector f_nk = n*kk_n*(distance-d_0);
    Vector f_sk = s*kk_s*ds0;

    // Updating bond forces vectors
    Vector fn = f_n1 + f_n2 + f_n3 + f_n4 + f_nk;
    Vector fs = f_s1 + f_s2 + f_s3 + f_s4 + f_sk;

    //check if bond breaks because maximum stress is reached
    const double current_sigma = fn.length()/Ab;
    const bool normal_stress_exceeded = sigma_max < current_sigma;
    const double current_tau = fs.length()/Ab;
    const bool shear_stress_exceeded = tau_max < current_tau;

    if (normal_stress_exceeded || shear_stress_exceeded)
    {
        breakBond(pij);
        return;
    }

    //apply total force
    force += fn + fs;

    //store new history values/Vectors
    pij.setHistoryVector("r_old", r);

    pij.setHistoryVector("f_n1", f_n1);
    pij.setHistoryVector("f_n2", f_n2);
    pij.setHistoryVector("f_n3", f_n3);
    pij.setHistoryVector("f_n4", f_n4);
    pij.setHistoryVector("f_s1", f_s1);
    pij.setHistoryVector("f_s2", f_s2);
    pij.setHistoryVector("f_s3", f_s3);
    pij.setHistoryVector("f_s4", f_s4);
}

void CohesionBondMax::breakBond(ParticleInteraction &pij)
{
    pij.setHistoryValue("bond_exists", 0.);
    pij.setHistoryValue("bond_broken", 1.);
}

void CohesionBondMax::createBond(ParticleInteraction &pij, Particle &pi, Particle &pj)
{
    pij.setHistoryValue("bond_exists", 1.);
    pij.setHistoryValue("bond_broken", 0.);

    Vector r_0 = pi.getPosition() - pj.getPosition();
    pij.setHistoryVector("r_0", r_0);
    pij.setHistoryVector("r_old", r_0);
}
