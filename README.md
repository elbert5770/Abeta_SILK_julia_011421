# Abeta_SILK_julia_011421

Archive of code used in Elbert et al. Comms Bio.

Important parameters Qleak, VSP and fVCSF are set at the bottom of the code following these functions:

function Optim.simplexer(A::MatlabSimplexer, initial_x::AbstractArray{T, N}) where {T, N}
function subject_optimization(subjectnum,initial_param_estimate,VSP,Qleak,fVCSF)
function glymphmodel(z,LB,UB,T38,MFL38,T40,MFL40,T42,MFL42,f,TotalCSF,VCSF,Vbr,timespan,Ab38conc,Ab40conc,Ab42conc,Tconc,CSFave,Km,Vm,result,maxMFL38,maxMFL40,maxMFL42,maxconc,VSP,Qleak,fVCSF,leak_region,GT1,LT1)
function SILK_dydt(dy,y,p,t)
function run_ode_solver(z,LB,UB,T38,MFL38,T40,MFL40,T42,MFL42,f,TotalCSF,VCSF,Vbr,timespan,Ab38conc,Ab40conc,Ab42conc,Tconc,CSFave,Km,Vm,result,maxMFL38,maxMFL40,maxMFL42,maxconc,VSP,Qleak,fVCSF,leak_region,GT1,LT1)
function setparams(a,VCSF,TotalCSF,Vbr,VSP,Qleak,fVCSF,leak_region)
function set_kf(kBPD38,kBPD40,kBPD42,QCSF,Qglym,QSN,Qosc2,Vbr,c_vent,cCSF38,cCSF40,cCSF42,kf,Km,Vm)
function dy!(dy,y,c_brain38,c_brain40,c_brain42,ccr38,ccr40,ccr42,Qglym,kBPD38,kBPD40,kBPD42,Vbr,kf,Km,Vm,kgamma40)
function dyj!(J,y,c_brain38,c_brain40,c_brain42,ccr38,ccr40,ccr42,Qglym,kBPD38,kBPD40,kBPD42,Vbr,kf,Km,Vm,kgamma40)
function set_flows(t,f,VSP3,QSN,VLP,Qleak,timespan,QCSF,leak_region,GT1,LT1,Qleak2)

There are a number of places where code could be sped up.  In particular, a more modern approach would be to define a loss function that received only the optimized parameters.  This would allow the use of automatic differentiation instead of finite differences.  The ODE solver would be called from within the loss function and the ODEProblem would be remade instead of reallocated at each call.

Notice that the hourly withdrawal of CSF is simulated, with a resulting change in volume of the lumbar compartment modeled.  This severely restricts the timestep size.  Additionally, numerical errors result in some amount of overshoot or undershoot at the end of the withdrawal period.  These are corrected by allowing an exponential decay back to the initial value.  This will result in slightly more variation in the derived parameters based on timestep size and solver.  These effects are small compared to the effects of Qleak, VSP and fVCSF on the concentration time course.

Initial estimates of parameter values are contained in 'inital_param_estimate.mat'.  These are fully optimized values that were determined starting with these values for all subjects: QCSF = 25; kBPD38 = 0.25; kBPD40 = 0.25; kBBPD42 = 0.25; Qosc = 10; Qosc2 = Qosc; Qglym = Qosc; VSP = 80; SF38 = 0.95; SF40 = 0.9; SF42 = 0.85; kex42 = 0.0001; fVCSF = 1.0; fSN = 0.2.  These valuse were then fed into a model with QSN = 0.1*QCSF and fSN = 0.1 and the fully optimized output were used as the initial estimate values.
