# MIT License
#
# Copyright (c) 2021 Donald L. Elbert
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

using Plots
using MAT
using DataFrames
using Optim
using DifferentialEquations
using CSV
using NLsolve
using DelimitedFiles

gr()


struct MatlabSimplexer{T} <: Optim.Simplexer
    a::T
    b::T
end
MatlabSimplexer(;a = 0.00025, b = 0.05) = MatlabSimplexer(a, b)

function Optim.simplexer(A::MatlabSimplexer, initial_x::AbstractArray{T, N}) where {T, N}
    n = length(initial_x)

    initial_simplex = Array{T, N}[initial_x for i = 1:n+1]


    for j = 1:n
        Arb = zeros(Float64,n)
        Arb[j] = A.b
        if all(abs.(Arb.*initial_simplex[j+1]) .< A.a)
            initial_value = A.a
        else
            initial_value = 0.0

        end
        initial_simplex[j+1] += Arb .* initial_simplex[j+1] .+ initial_value


    end

    return initial_simplex
end

function subject_optimization(subjectnum,initial_param_estimate,VSP,Qleak,fVCSF)


    if subjectnum == 21 || subjectnum == 67
        outputfile = string(Base.source_path(),".csv")

        outputstr = vcat(subjects, 0)
        fid1 = open(outputfile, "a")
        writedlm(fid1, permutedims(outputstr),",")
        close(fid1)

        return
    end

    filename = string(@__DIR__ ,"\\LOAD_YNC_FACS_",subjectnum,"_20211210.mat")



    vars = matread(filename)

    if vars["ELISAonly"] == 0
        Tconc = vec(vars["Tconcexport"])
        CSFave = vec(vars["CSFexport"])
    else
        Tconc = vec(vars["TconcexportELISA"])
        CSFave = vec(vars["CSFexportELISA"])
    end
    println(Tconc)
    println(CSFave)
    Ab38conc = vars["Ab38conc"]
    Ab40conc = vars["Ab40conc"]
    Ab42conc = vars["Ab42conc"]
    Abtotconc = vars["Abtotconc"]
    T38 = vec(vars["T38"])
    MFL38 = vec(vars["MFL38"])
    T40 = vec(vars["T40"])
    MFL40 = vec(vars["MFL40"])
    T42 =  vec(vars["T42"])
    MFL42 = vec(vars["MFL42"])
    maxMFL38 = maximum(MFL38)-minimum(MFL38)
    maxMFL40 = maximum(MFL40)-minimum(MFL40)
    maxMFL42 = maximum(MFL42)-minimum(MFL42)
    maxconc = (maximum(CSFave)-minimum(CSFave))*3


    f = vars["f"]

    Time48h = vars["Time48h"]

    println(Time48h)

    Brain_thickness = vars["Brain_thickness"]
    V_ISF = vars["V_ISF"]
    Vent_CSF = vars["Vent_CSF"]
    timespan = 48

    VCSF = (-1227.5*Brain_thickness+1229.5)/2.33
    Vbr = V_ISF
    TotalCSF = Vent_CSF
    QCSF = initial_param_estimate[9]

    kBPD38 = initial_param_estimate[2]
    kBPD40 = initial_param_estimate[3]
    kBPD42 = initial_param_estimate[4]
    Qosc = initial_param_estimate[5]
    Qglym = QCSF
    SF38 = initial_param_estimate[6]
    SF40 = initial_param_estimate[7]
    SF42 = initial_param_estimate[8]
    kex42 = initial_param_estimate[10]



    Km = [0.186
    1.64
    28.8
    0.915
    0.0672]*3.2012*4329.8

    Vm = [1.10
    0.153
    14.6
    1
    0.0223]*82.3529*4329.8


    leak_region = 0
    GT1 = 22
    LT1 = 32
    Leak2 = 5



    a = [kBPD38, kBPD40, kBPD42, Qosc, SF38, SF40, SF42, QCSF, kex42]
    LB = [0.001,0.001,0.001,2,0.5,0.5,0.5,10,0]
    UB = [2,2,2,100,1.5,1.5,1.5,80,1]


    result = 0

    z = (2*((a-LB)./(UB-LB)) ) .- 1


    for i = 1:length(z)
        z[i] = asin(z[i])+pi
    end
    optimizations = 2
    for  rounds = 1:optimizations
        res = Optim.optimize(z -> glymphmodel(z,LB,UB,T38,MFL38,T40,MFL40,T42,MFL42,f,TotalCSF,VCSF,Vbr,timespan,Ab38conc,Ab40conc,Ab42conc,Tconc,CSFave,Km,Vm,result,maxMFL38,maxMFL40,maxMFL42,maxconc,VSP,Qleak,fVCSF,leak_region,GT1,LT1),
        z,
        NelderMead(initial_simplex = MatlabSimplexer(),parameters = Optim.FixedParameters(α = 1, β = 2, γ = 0.5, δ = 0.5)),
        Optim.Options(iterations = 100,g_tol = 1e-6))


        println(res)

        z = Optim.minimizer(res)

        for i = 1:length(z)
            a[i] = LB[i] + (UB[i] - LB[i])*(sin(z[i]-pi) + 1) /2

        end
        println(a)
    end


    result = 1


    ssr1,ssr2,ssr3,ssrconc,ssrconctot,conc38,conc40,conc42,df38,df40,df42,dfconc,volumes,conc_common,instructions = run_ode_solver(z,LB,UB,T38,MFL38,T40,MFL40,T42,MFL42,f,TotalCSF,VCSF,Vbr,timespan,Ab38conc,Ab40conc,Ab42conc,Tconc,CSFave,Km,Vm,result,maxMFL38,maxMFL40,maxMFL42,maxconc,VSP,Qleak,fVCSF,leak_region,GT1,LT1);


    @show ssr1+ssr2+ssr3

    outputfile = string(@__FILE__,".csv")

    outputstr = vcat(subjectnum, 1, a, instructions, ssr1, ssr2, ssr3, ssr1+ssr2+ssr3, ssrconc, conc38, conc40, conc42, conc_common, volumes)

    fid1 = open(outputfile, "a")
    writedlm(fid1, permutedims(outputstr),",")
    close(fid1)

    filename = string(@__FILE__,"_",subjectnum,"_Ab38results.csv")
    CSV.write(filename, df38,append=true,header=true)
    filename = string(@__FILE__,"_",subjectnum,"_Ab40results.csv")
    CSV.write(filename, df40,append=true,header=true)
    filename = string(@__FILE__,"_",subjectnum,"_Ab42results.csv")
    CSV.write(filename, df42,append=true,header=true)
    filename = string(@__FILE__,"_",subjectnum,"_concresults.csv")
    CSV.write(filename, dfconc,append=true,header=true)


end

function glymphmodel(z,LB,UB,T38,MFL38,T40,MFL40,T42,MFL42,f,TotalCSF,VCSF,Vbr,timespan,Ab38conc,Ab40conc,Ab42conc,Tconc,CSFave,Km,Vm,result,maxMFL38,maxMFL40,maxMFL42,maxconc,VSP,Qleak,fVCSF,leak_region,GT1,LT1)

    ssr1,ssr2,ssr3,ssrconc,ssrconctot,conc38,conc40,conc42,df,dfconc,volumes,conc_common,instructions = run_ode_solver(z,LB,UB,T38,MFL38,T40,MFL40,T42,MFL42,f,TotalCSF,VCSF,Vbr,timespan,Ab38conc,Ab40conc,Ab42conc,Tconc,CSFave,Km,Vm,result,maxMFL38,maxMFL40,maxMFL42,maxconc,VSP,Qleak,fVCSF,leak_region,GT1,LT1);

    return ssr1+ssr2+ssr3
end

function SILK_dydt(dy,y,p,t)
    QCSF,VCSF_CM,VCSF_cr,Qglym,c_vent,Vbr,kBPD38,kBPD40,kBPD42,kgamma38,kgamma40,kgamma42,kf,Qosc,f,VSP1,VSP2,VSP3,kAPP,Qosc2,VLP,QSN,timespan,Qleak,Km,Vm,kex42,kret,leak_region,GT1,LT1,Qleak2 = p
    Leu,Qcranial,QSN,QLP,Qrefill,VSP3 = set_flows(t,f,VSP3,QSN,VLP,Qleak,timespan,QCSF,leak_region,GT1,LT1,Qleak2);

    if Qrefill != 0
        VSP3 = y[45]
    end


    dy[1] = (Vm[2]*y[15]/Km[2])/(1+(y[15]+y[16])/Km[2]) - ((kgamma38+kgamma40+kgamma42)*y[1]/Km[4])/(1+(y[17]+y[18])/Km[3]+(y[1]+y[2])/Km[4]) - (Vm[5]*y[1]/Km[5])/(1+(y[15]+y[16])/Km[1]+(y[1]+y[2])/Km[5]);#C99
    dy[2] = (Vm[2]*y[16]/Km[2])/(1+(y[15]+y[16])/Km[2]) - ((kgamma38+kgamma40+kgamma42)*y[2]/Km[4])/(1+(y[17]+y[18])/Km[3]+(y[1]+y[2])/Km[4]) - (Vm[5]*y[2]/Km[5])/(1+(y[15]+y[16])/Km[1]+(y[1]+y[2])/Km[5])
    dy[3] = (kgamma38*y[1]/Km[4])/(1+(y[17]+y[18])/Km[3]+(y[1]+y[2])/Km[4]) + Qglym/Vbr*(y[5]-y[3]) - kBPD38*y[3];# brain 38
    dy[4] = (kgamma38*y[2]/Km[4])/(1+(y[17]+y[18])/Km[3]+(y[1]+y[2])/Km[4]) + Qglym/Vbr*(y[6]-y[4]) - kBPD38*y[4];# brain 38l
    dy[5] = Qcranial/VCSF_cr*(y[7]-y[5]) + Qglym/VCSF_cr*(y[3]-y[5]) + Qosc2/VCSF_cr*(y[7]-y[5]);# cranial 38
    dy[6] = Qcranial/VCSF_cr*(y[8]-y[6]) + Qglym/VCSF_cr*(y[4]-y[6]) + Qosc2/VCSF_cr*(y[8]-y[6]);# cranial 38l
    dy[7] = QCSF/VCSF_CM*(c_vent-y[7]) + Qosc/VCSF_CM*(y[9]-y[7]) + Qosc2/VCSF_CM*(y[5]-y[7]) ;# cisterna magna + ventricles 38
    dy[8] = QCSF/VCSF_CM*(c_vent-y[8]) + Qosc/VCSF_CM*(y[10]-y[8]) + Qosc2/VCSF_CM*(y[6]-y[8]) # cisterna magna + ventricles 38l
    dy[9] = Qosc/VSP1*(y[7]-2*y[9]+y[11])+QLP/VSP1*(y[7]-y[9])+QSN/VSP1*(y[7]-y[9]) #sp1 38
    dy[10] = Qosc/VSP1*(y[8]-2*y[10]+y[12])+QLP/VSP1*(y[8]-y[10])+QSN/VSP1*(y[8]-y[10]) #sp1 38l
    dy[11] = Qosc/VSP2*(y[9]-2*y[11]+y[46])+QLP/VSP2*(y[9]-y[11])+QSN/VSP2*2/3*(y[9]-y[11]) #sp2 38
    dy[12] = Qosc/VSP2*(y[10]-2*y[12]+y[47])+QLP/VSP2*(y[10]-y[12])+QSN/VSP2*2/3*(y[10]-y[12]) #sp2 38l
    dy[13] = Qosc*(y[11]-y[46])+QLP*(y[11]-y[46])+QSN/3*(y[11]-y[46])+Qrefill*y[46] #sp3 38
    dy[14] = Qosc*(y[12]-y[47])+QLP*(y[12]-y[47])+QSN/3*(y[12]-y[47])+Qrefill*y[47] #sp3 38l
    dy[15] = (1-Leu)*kf - (Vm[1]*y[15]/Km[1])/(1+(y[15]+y[16])/Km[1]+(y[1]+y[2])/Km[5]) - (Vm[2]*y[15]/Km[2])/(1+(y[15]+y[16])/Km[2]); #APP
    dy[16] = Leu*kf - (Vm[1]*y[16]/Km[1])/(1+(y[15]+y[16])/Km[1]+(y[1]+y[2])/Km[5]) - (Vm[2]*y[16]/Km[2])/(1+(y[15]+y[16])/Km[2]) #APPl
    dy[17] = (Vm[1]*y[15]/Km[1])/(1+(y[15]+y[16])/Km[1]+(y[1]+y[2])/Km[5]) +(Vm[5]*y[1]/Km[5])/(1+(y[15]+y[16])/Km[1]+(y[1]+y[2])/Km[5]) - (Vm[3]*y[17]/Km[3])/(1+(y[17]+y[18])/Km[3]+(y[1]+y[2])/Km[4]);#C83
    dy[18] = (Vm[1]*y[16]/Km[1])/(1+(y[15]+y[16])/Km[1]+(y[1]+y[2])/Km[5]) +(Vm[5]*y[2]/Km[5])/(1+(y[15]+y[16])/Km[1]+(y[1]+y[2])/Km[5]) - (Vm[3]*y[18]/Km[3])/(1+(y[17]+y[18])/Km[3]+(y[1]+y[2])/Km[4]) #C83l
    dy[19] = (kgamma40*y[1]/Km[4])/(1+(y[17]+y[18])/Km[3]+(y[1]+y[2])/Km[4]) + Qglym/Vbr*(y[21]-y[19]) - kBPD40*y[19];# brain 40
    dy[20] = (kgamma40*y[2]/Km[4])/(1+(y[17]+y[18])/Km[3]+(y[1]+y[2])/Km[4]) + Qglym/Vbr*(y[22]-y[20]) - kBPD40*y[20];# brain 40l
    dy[21] = Qcranial/VCSF_cr*(y[23]-y[21]) + Qglym/VCSF_cr*(y[19]-y[21]) + Qosc2/VCSF_cr*(y[23]-y[21]);# cranial 40
    dy[22] = Qcranial/VCSF_cr*(y[24]-y[22]) + Qglym/VCSF_cr*(y[20]-y[22]) + Qosc2/VCSF_cr*(y[24]-y[22]);# cranial 40l
    dy[23] = QCSF/VCSF_CM*(c_vent-y[23]) + Qosc/VCSF_CM*(y[25]-y[23]) + Qosc2/VCSF_CM*(y[21]-y[23]) ;# cisterna magna + ventricles 40
    dy[24] = QCSF/VCSF_CM*(c_vent-y[24]) + Qosc/VCSF_CM*(y[26]-y[24]) + Qosc2/VCSF_CM*(y[22]-y[24]) # cisterna magna + ventricles 40l
    dy[25] = Qosc/VSP1*(y[23]-2*y[25]+y[27])+QLP/VSP1*(y[23]-y[25])+QSN/VSP1*(y[23]-y[25]) #sp1 40
    dy[26] = Qosc/VSP1*(y[24]-2*y[26]+y[28])+QLP/VSP1*(y[24]-y[26])+QSN/VSP1*(y[24]-y[26]) #sp1 40l
    dy[27] = Qosc/VSP2*(y[25]-2*y[27]+y[48])+QLP/VSP2*(y[25]-y[27])+QSN/VSP2*2/3*(y[25]-y[27]) #sp2 40
    dy[28] = Qosc/VSP2*(y[26]-2*y[28]+y[49])+QLP/VSP2*(y[26]-y[28])+QSN/VSP2*2/3*(y[26]-y[28]) #sp2 40l
    dy[29] = Qosc*(y[27]-y[48])+QLP*(y[27]-y[48])+QSN/3*(y[27]-y[48])+Qrefill*y[48] #sp3 40
    dy[30] = Qosc*(y[28]-y[49])+QLP*(y[28]-y[49])+QSN/3*(y[28]-y[49])+Qrefill*y[49] #sp3 40l
    dy[31] = (kgamma42*y[1]/Km[4])/(1+(y[17]+y[18])/Km[3]+(y[1]+y[2])/Km[4]) + Qglym/Vbr*(y[33]-y[31]) - kBPD42*y[31] - kex42*y[31] + kret*y[43];# brain 42
    dy[32] = (kgamma42*y[2]/Km[4])/(1+(y[17]+y[18])/Km[3]+(y[1]+y[2])/Km[4]) + Qglym/Vbr*(y[34]-y[32]) - kBPD42*y[32] - kex42*y[32] + kret*y[44];# brain 42l
    dy[33] = Qcranial/VCSF_cr*(y[35]-y[33]) + Qglym/VCSF_cr*(y[31]-y[33]) + Qosc2/VCSF_cr*(y[35]-y[33]);# cranial 42
    dy[34] = Qcranial/VCSF_cr*(y[36]-y[34]) + Qglym/VCSF_cr*(y[32]-y[34]) + Qosc2/VCSF_cr*(y[36]-y[34]);# cranial 42l
    dy[35] = QCSF/VCSF_CM*(c_vent-y[35]) + Qosc/VCSF_CM*(y[37]-y[35]) + Qosc2/VCSF_CM*(y[33]-y[35]) ;# cisterna magna + ventricles 42
    dy[36] = QCSF/VCSF_CM*(c_vent-y[36]) + Qosc/VCSF_CM*(y[38]-y[36]) + Qosc2/VCSF_CM*(y[34]-y[36]) # cisterna magna + ventricles 42l
    dy[37] = Qosc/VSP1*(y[35]-2*y[37]+y[39])+QLP/VSP1*(y[35]-y[37])+QSN/VSP1*(y[35]-y[37]) #sp1 42
    dy[38] = Qosc/VSP1*(y[36]-2*y[38]+y[40])+QLP/VSP1*(y[36]-y[38])+QSN/VSP1*(y[36]-y[38]) #sp1 42l
    dy[39] = Qosc/VSP2*(y[37]-2*y[39]+y[50])+QLP/VSP2*(y[37]-y[39])+QSN/VSP2*2/3*(y[37]-y[39]) #sp2 42
    dy[40] = Qosc/VSP2*(y[38]-2*y[40]+y[51])+QLP/VSP2*(y[38]-y[40])+QSN/VSP2*2/3*(y[38]-y[40]) #sp2 42l
    dy[41] = Qosc*(y[39]-y[50])+QLP*(y[39]-y[50])+QSN/3*(y[39]-y[50])+Qrefill*y[50] #sp3 42
    dy[42] = Qosc*(y[40]-y[51])+QLP*(y[40]-y[51])+QSN/3*(y[40]-y[51])+Qrefill*y[51] #sp3 42l
    dy[43] = kex42*y[31] - kret*y[43] #ex42
    dy[44] = kex42*y[32] - kret*y[44] #ex42
    dy[45] = Qrefill
    if Qrefill == 0
        dy[45] = 10*(VSP3-y[45])
    end
    dy[46] = 1/VSP3*dy[13] - y[13]/VSP3^2*dy[45]
    dy[47] = 1/VSP3*dy[14] - y[14]/VSP3^2*dy[45]
    dy[48] = 1/VSP3*dy[29] - y[29]/VSP3^2*dy[45]
    dy[49] = 1/VSP3*dy[30] - y[30]/VSP3^2*dy[45]
    dy[50] = 1/VSP3*dy[41] - y[41]/VSP3^2*dy[45]
    dy[51] = 1/VSP3*dy[42] - y[42]/VSP3^2*dy[45]




end



function run_ode_solver(z,LB,UB,T38,MFL38,T40,MFL40,T42,MFL42,f,TotalCSF,VCSF,Vbr,timespan,Ab38conc,Ab40conc,Ab42conc,Tconc,CSFave,Km,Vm,result,maxMFL38,maxMFL40,maxMFL42,maxconc,VSP,Qleak,fVCSF,leak_region,GT1,LT1)

    a = zeros(size(z))
    for i = 1:length(z)

        a[i] = LB[i] + (UB[i] - LB[i])*(sin(z[i]-pi)+1)/2

    end


    VCSF_cr,VCSF_CM, kBPD38, kBPD40, kBPD42, Qosc, Qosc2, SF38, SF40, SF42, kex42, kret, c_vent, kf, QCSF, Qglym, VLP, QSN, VSP1, VSP2, VSP3, kAPP, Qleak2 = setparams(a,VCSF,TotalCSF,Vbr,VSP,Qleak,fVCSF,leak_region);


    instructions = [Qleak,0,0,0,0,fVCSF,VSP]


    ccr38,ccr40,ccr42,c_brain38,c_brain40,c_brain42,kf,cC99,cAPP,cC83,kgamma38,kgamma40,kgamma42 = set_kf(kBPD38,kBPD40,kBPD42,QCSF,Qglym,QSN,Qosc2,Vbr,c_vent,Ab38conc,Ab40conc,Ab42conc,kf,Km,Vm);


    cCM38 = Ab38conc
    cCSF38 = Ab38conc
    cCM40 = Ab40conc
    cCSF40 = Ab40conc
    cCM42 = Ab42conc
    cCSF42 = Ab42conc
    cex = kex42/kret*c_brain42;
    conc_common = [cC99,cC83,cAPP,kf]
    conc38 = [Ab38conc,ccr38,c_brain38,kgamma38];

    u0 = [cC99, 0,  c_brain38, 0, ccr38, 0, cCM38, 0, cCSF38, 0, cCSF38, 0, cCSF38*VSP3, 0, cAPP, 0, cC83, 0, c_brain40, 0, ccr40, 0, cCM40, 0, cCSF40, 0, cCSF40, 0, cCSF40*VSP3, 0, c_brain42, 0, ccr42, 0, cCM42, 0, cCSF42, 0, cCSF42, 0, cCSF42*VSP3, 0, cex, 0, VSP3, cCSF38, 0, cCSF40, 0, cCSF42, 0]
    p = [QCSF,VCSF_CM,VCSF_cr,Qglym,c_vent,Vbr,kBPD38,kBPD40,kBPD42,kgamma38,kgamma40,kgamma42,kf,Qosc,f,VSP1,VSP2,VSP3,kAPP,Qosc2,VLP,QSN,timespan,Qleak,Km,Vm,kex42,kret,leak_region,GT1,LT1,Qleak2]
    if result == 1
        dtmax = 0.001
    else
        dtmax = 0.01
    end
    tspan = (0.0,timespan)
    prob = ODEProblem(SILK_dydt,u0,tspan,p)
    sol = solve(prob,Rosenbrock23(),reltol=1e-3,dtmax=dtmax)
    y38 = sol(T38)


    conc40 = [Ab40conc,ccr40,c_brain40,kgamma40];
    y40 = sol(T40)
    y40conc = sol(Tconc)

    xplot2 = sol.t
    yplot2 = sol[45,:]

    ssrconc = 0.0
    for i = 1:length(Tconc)

        ssrconc += (((y40conc[48,i]+y40conc[49,i])/cCSF40-CSFave[i]/CSFave[1])/maxconc).^2

    end
    conc_pred = (y40conc[48,:]+y40conc[49,:])/cCSF40
    conc_actual = CSFave/CSFave[1]





    conc42 = [Ab42conc,ccr42,c_brain42,kgamma42,cex];
    volumes = [TotalCSF,VCSF,Vbr,VCSF_cr,VCSF_CM,c_vent,VLP,QSN,VSP1,VSP2,VSP3,VSP1+VSP2+VSP3,Qglym]

    y42 = sol(T42)



    ssr1 = sum(((SF38.*y38[47,:]./(y38[46,:]+y38[47,:])-MFL38)/maxMFL38).^2)
    ssr2 = sum(((SF40.*y40[49,:]./(y40[48,:]+y40[49,:])-MFL40)/maxMFL40).^2)
    ssr3 = sum(((SF42.*y42[51,:]./(y42[50,:]+y42[51,:])-MFL42)/maxMFL42).^2)


    if result == 1
        ssr1 = 0.0
        ssr2 = 0.0
        ssr3 = 0.0

        for i = 1:length(T38)
            if T38[i] > 0 && T38[i] <= 36
                ssr1 += ((SF38.*y38[47,i]./(y38[46,i]+y38[47,i])-MFL38[i])).^2
            end
        end
        for i = 1:length(T40)
            if T40[i] > 0 && T40[i] <= 36
                ssr2 += ((SF40.*y40[49,i]./(y40[48,i]+y40[49,i])-MFL40[i])).^2
            end
        end
        for i = 1:length(T42)
            if T42[i] > 0 && T42[i] <= 36
                ssr3 += ((SF42.*y42[51,i]./(y42[50,i]+y42[51,i])-MFL42[i])).^2
            end
        end

        ssrconctot = sum(((conc_pred-conc_actual)).^2)

        @show a

        df38 = DataFrame(time38=T38,result38=SF38.*y38[47,:]./(y38[46,:]+y38[47,:]),data38=MFL38)
        df40 = DataFrame(time40=T40,result40=SF40.*y40[49,:]./(y40[48,:]+y40[49,:]),data40=MFL40)
        df42 = DataFrame(time42=T42,result42=SF42.*y42[51,:]./(y42[50,:]+y42[51,:]),data42=MFL42)
        dfconc = DataFrame(timeconc=Tconc,resultconc=(y40conc[48,:]+y40conc[49,:])./cCSF40,conc=CSFave./CSFave[1])
        p4 = plot(T40,(y40[48,:]+y40[49,:])/cCSF40,legend=:bottomright,legendfontsize=5,linecolor =  :green,label="SP3",xlabel = "Time (h)",ylabel = "normalized Aβ \nconcentration")
        scatter!(Tconc,conc_actual,legend = false,markercolor=:green)

        p2 = plot(xplot2,yplot2,legend = false,xlabel = "Time (h)",ylabel = "lumbar volume (mL)")

        xplot1 = Vector[T38,Vector[0:timespan],T38,T38,T38,T38,T38]
        yplot1 = Vector[SF38*y38[47,:]./(y38[46,:]+y38[47,:]),vec(f),SF38*y38[16,:]./(y38[15,:]+y38[16,:]),SF38*y38[2,:]./(y38[1,:]+y38[2,:]),SF38*y38[4,:]./(y38[3,:]+y38[4,:]),SF38*y38[6,:]./(y38[5,:]+y38[6,:]),SF38*y38[8,:]./(y38[7,:]+y38[8,:])]
        p1 = plot(xplot1,yplot1,legendfontsize=5,label = ["SP3" "Leu" "APP" "C99" "ISF" "cranial" "CM"],xlabel = "Time (h)",ylabel = "SILK Aβ \n labeled fraction")


        p3 = plot(T38,SF38*y38[47,:]./(y38[46,:]+y38[47,:]),legend = false,linecolor =  :blue,xlabel = "Time (h)",ylabel = "SILK Aβ \n labeled fraction")
        plot!(p3,T40,SF40*y40[49,:]./(y40[48,:]+y40[49,:]),legend = false,linecolor =  :green)
        plot!(p3,T42,SF42*y42[51,:]./(y42[50,:]+y42[51,:]),legend = false,linecolor =  :red)
        scatter!(p3,T38,MFL38,legend = false,markercolor=:blue)
        scatter!(p3,T40,MFL40,legend = false,markercolor=:green)
        scatter!(p3,T42,MFL42,legend = false,markercolor=:red)
        display(plot(p1,p2,p3,p4,layout = (2,2)))
    else
        ssrconctot = 0
        df38 = 0
        df40 = 0
        df42 = 0
        dfconc = 0
    end
    ssr1,ssr2,ssr3,ssrconc,ssrconctot,conc38,conc40,conc42,df38,df40,df42,dfconc,volumes,conc_common,instructions
end

function setparams(a,VCSF,TotalCSF,Vbr,VSP,Qleak,fVCSF,leak_region)

    VCSF = fVCSF*VCSF
    VCSF_cr = VCSF*0.838
    VCSF_CM = VCSF*0.162+TotalCSF
    kBPD38 = a[1]
    kBPD40 = a[2]
    kBPD42 = a[3]

    Qosc = a[4]

    SF38 = a[5]
    SF40 = a[6]
    SF42 = a[7]
    kex42 = a[9]
    kret = 0.1
    kAPP = 1.119778
    c_vent = 0
    kf = 100.0*Vbr
    QCSF = a[8]
    Qglym = QCSF
    Qosc2 = Qosc
    VLP = 6.0
    QSN = QCSF*0.1
    #VSP = 80.0
    VSP1 = VSP*fVCSF*26.197/97.33
    VSP2 = VSP*fVCSF*28.707/97.33
    VSP3 = VSP*fVCSF*42.43/97.33

    Qleak2 = 0

    VCSF_cr,VCSF_CM, kBPD38, kBPD40, kBPD42, Qosc, Qosc2, SF38, SF40, SF42, kex42, kret, c_vent, kf, QCSF, Qglym, VLP, QSN, VSP1, VSP2, VSP3, kAPP, Qleak2
end

function set_kf(kBPD38,kBPD40,kBPD42,QCSF,Qglym,QSN,Qosc2,Vbr,c_vent,cCSF38,cCSF40,cCSF42,kf,Km,Vm)

    kgamma40 = 609798.0
    ccr38 = (cCSF38*(QCSF+Qosc2)-QCSF*c_vent)/Qosc2
    ccr40 = (cCSF40*(QCSF+Qosc2)-QCSF*c_vent)/Qosc2
    ccr42 = (cCSF42*(QCSF+Qosc2)-QCSF*c_vent)/Qosc2

    c_brain38 = (cCSF38*(QCSF*Qglym - QCSF*QSN + QCSF*Qosc2 + Qglym*Qosc2 + QCSF^2)-(QCSF^2*c_vent - QCSF*QSN*c_vent + QCSF*Qglym*c_vent + QCSF*Qosc2*c_vent))/(Qglym*Qosc2)
    c_brain40 = (cCSF40*(QCSF*Qglym - QCSF*QSN + QCSF*Qosc2 + Qglym*Qosc2 + QCSF^2)-(QCSF^2*c_vent - QCSF*QSN*c_vent + QCSF*Qglym*c_vent + QCSF*Qosc2*c_vent))/(Qglym*Qosc2)
    c_brain42 = (cCSF42*(QCSF*Qglym - QCSF*QSN + QCSF*Qosc2 + Qglym*Qosc2 + QCSF^2)-(QCSF^2*c_vent - QCSF*QSN*c_vent + QCSF*Qglym*c_vent + QCSF*Qosc2*c_vent))/(Qglym*Qosc2)
    r = nlsolve((dy,y) ->dy!(dy,y,c_brain38,c_brain40,c_brain42,ccr38,ccr40,ccr42,Qglym,kBPD38,kBPD40,kBPD42,Vbr,kf,Km,Vm,kgamma40), (J,y) ->dyj!(J,y,c_brain38,c_brain40,c_brain42,ccr38,ccr40,ccr42,Qglym,kBPD38,kBPD40,kBPD42,Vbr,kf,Km,Vm,kgamma40), [ c_brain40; 10000; c_brain40; c_brain40; 1.71; 1.71])
    cC99 = r.zero[1]
    kf = r.zero[2]
    cAPP = r.zero[3]
    cC83 = r.zero[4]
    kgamma38 = r.zero[5]
    kgamma42 = r.zero[6]
    ccr38,ccr40,ccr42,c_brain38,c_brain40,c_brain42,kf,cC99,cAPP,cC83,kgamma38,kgamma40,kgamma42
end



function dy!(dy,y,c_brain38,c_brain40,c_brain42,ccr38,ccr40,ccr42,Qglym,kBPD38,kBPD40,kBPD42,Vbr,kf,Km,Vm,kgamma40)

    dy[1] = (Vm[2]*y[3]/Km[2])/(1+(y[3])/Km[2]) - ((kgamma40+y[5]+y[6])*y[1]/Km[4])/(1+(y[4])/Km[3]+(y[1])/Km[4]) - (Vm[5]*y[1]/Km[5])/(1+(y[3])/Km[1]+(y[1])/Km[5]);#C99
    dy[2] = (kgamma40*y[1]/Km[4])/(1+(y[4])/Km[3]+(y[1])/Km[4]) + Qglym/Vbr*ccr40 - (Qglym/Vbr+kBPD40)*c_brain40 ;# brain
    dy[3] = y[2] - (Vm[1]*y[3]/Km[1])/(1+(y[3])/Km[1]+(y[1])/Km[5]) - (Vm[2]*y[3]/Km[2])/(1+(y[3])/Km[2]);#APP
    dy[4] = (Vm[1]*y[3]/Km[1])/(1+(y[3])/Km[1]+(y[1])/Km[5]) +(Vm[5]*y[1]/Km[5])/(1+(y[3])/Km[1]+(y[1])/Km[5]) - (Vm[3]*y[4]/Km[3])/(1+(y[4])/Km[3]+(y[1])/Km[4]);#C83
    dy[5] = (y[5]*y[1]/Km[4])/(1+(y[4])/Km[3]+(y[1])/Km[4]) + Qglym/Vbr*ccr38 - (Qglym/Vbr+kBPD38)*c_brain38
    dy[6] = (y[6]*y[1]/Km[4])/(1+(y[4])/Km[3]+(y[1])/Km[4]) + Qglym/Vbr*ccr42 - (Qglym/Vbr+kBPD42)*c_brain42

end

function dyj!(J,y,c_brain38,c_brain40,c_brain42,ccr38,ccr40,ccr42,Qglym,kBPD38,kBPD40,kBPD42,Vbr,kf,Km,Vm,kgamma40)
    J[1,1] = (Vm[5]*y[1])/(Km[5]^2*(y[3]/Km[1] + y[1]/Km[5] + 1)^2) - (kgamma40+y[5]+y[6])/(Km[4]*(y[1]/Km[4] + y[4]/Km[3] + 1)) - Vm[5]/(Km[5]*(y[3]/Km[1] + y[1]/Km[5] + 1)) + (y[1]*(kgamma40+y[5]+y[6]))/(Km[4]^2*(y[1]/Km[4] + y[4]/Km[3] + 1)^2)
    J[1,2] = 0
    J[1,3] =  Vm[2]/(Km[2]*(y[3]/Km[2] + 1)) - (Vm[2]*y[3])/(Km[2]^2*(y[3]/Km[2] + 1)^2) + (Vm[5]*y[1])/(Km[1]*Km[5]*(y[3]/Km[1] + y[1]/Km[5] + 1)^2)
    J[1,4] = (y[1]*(kgamma40+y[5]+y[6]))/(Km[3]*Km[4]*(y[1]/Km[4] + y[4]/Km[3] + 1)^2)
    J[1,5] = -y[1]/(Km[4]*(y[1]/Km[4] + y[4]/Km[3] + 1))
    J[1,6] = -y[1]/(Km[4]*(y[1]/Km[4] + y[4]/Km[3] + 1))
    J[2,1] =  kgamma40/(Km[4]*(y[1]/Km[4] + y[4]/Km[3] + 1)) - (y[1]*kgamma40)/(Km[4]^2*(y[1]/Km[4] + y[4]/Km[3] + 1)^2)
    J[2,2] = 0
    J[2,3] = 0
    J[2,4] =  -(y[1]*kgamma40)/(Km[3]*Km[4]*(y[1]/Km[4] + y[4]/Km[3] + 1)^2)
    J[2,5] = 0
    J[2,6] = 0
    J[3,1] = (Vm[1]*y[3])/(Km[1]*Km[5]*(y[3]/Km[1] + y[1]/Km[5] + 1)^2)
    J[3,2] = 1
    J[3,3] = (Vm[2]*y[3])/(Km[2]^2*(y[3]/Km[2] + 1)^2) - Vm[2]/(Km[2]*(y[3]/Km[2] + 1)) - Vm[1]/(Km[1]*(y[3]/Km[1] + y[1]/Km[5] + 1)) + (Vm[1]*y[3])/(Km[1]^2*(y[3]/Km[1] + y[1]/Km[5] + 1)^2)
    J[3,4] = 0
    J[3,5] = 0
    J[3,6] = 0
    J[4,1] = Vm[5]/(Km[5]*(y[3]/Km[1] + y[1]/Km[5] + 1)) - (Vm[5]*y[1])/(Km[5]^2*(y[3]/Km[1] + y[1]/Km[5] + 1)^2) - (Vm[1]*y[3])/(Km[1]*Km[5]*(y[3]/Km[1] + y[1]/Km[5] + 1)^2) + (Vm[3]*y[4])/(Km[3]*Km[4]*(y[1]/Km[4] + y[4]/Km[3] + 1)^2)
    J[4,2] = 0
    J[4,3] = Vm[1]/(Km[1]*(y[3]/Km[1] + y[1]/Km[5] + 1)) - (Vm[1]*y[3])/(Km[1]^2*(y[3]/Km[1] + y[1]/Km[5] + 1)^2) - (Vm[5]*y[1])/(Km[1]*Km[5]*(y[3]/Km[1] + y[1]/Km[5] + 1)^2)
    J[4,4] = (Vm[3]*y[4])/(Km[3]^2*(y[1]/Km[4] + y[4]/Km[3] + 1)^2) - Vm[3]/(Km[3]*(y[1]/Km[4] + y[4]/Km[3] + 1))
    J[4,5] = 0
    J[4,6] = 0
    J[5,1] = y[5]/(Km[4]*(y[1]/Km[4] + y[4]/Km[3] + 1)) - (y[1]*y[5])/(Km[4]^2*(y[1]/Km[4] + y[4]/Km[3] + 1)^2)
    J[5,2] = 0
    J[5,3] = 0
    J[5,4] = -(y[1]*y[5])/(Km[3]*Km[4]*(y[1]/Km[4] + y[4]/Km[3] + 1)^2)
    J[5,5] = y[1]/(Km[4]*(y[1]/Km[4] + y[4]/Km[3] + 1))
    J[5,6] = 0
    J[6,1] = y[6]/(Km[4]*(y[1]/Km[4] + y[4]/Km[3] + 1)) - (y[1]*y[6])/(Km[4]^2*(y[1]/Km[4] + y[4]/Km[3] + 1)^2)
    J[6,2] = 0
    J[6,3] = 0
    J[6,4] = -(y[1]*y[6])/(Km[3]*Km[4]*(y[1]/Km[4] + y[4]/Km[3] + 1)^2)
    J[6,5] = 0
    J[6,6] = y[1]/(Km[4]*(y[1]/Km[4] + y[4]/Km[3] + 1))
end

function set_flows(t,f,VSP3,QSN,VLP,Qleak,timespan,QCSF,leak_region,GT1,LT1,Qleak2)
    CSFdrawtime = 0.1;
    floort = floor(Int,t)
    hourfraction = t-floort
    if t < timespan
        Leu = f[floort+1]+(f[floort+2]-f[floort+1])*hourfraction
    else()
        Leu = f[timespan+1]
    end


    Qrefill = 0.0;



    if VLP > 0.0
        if VLP/CSFdrawtime > QCSF
            if hourfraction < CSFdrawtime
                Qcranial = 0.0;
                QSN = 0.0;
                QLP = QCSF;
                Qrefill = -VLP/CSFdrawtime+QCSF;
            else
                Vlost = CSFdrawtime*(VLP/CSFdrawtime-QCSF);
                if hourfraction < CSFdrawtime+Vlost/QCSF
                    Qcranial = 0.0;
                    QSN = 0.0;
                    QLP = QCSF;
                    Qrefill = QCSF;
                else
                    if leak_region == 1
                        if t > GT1 && t <LT1
                            Qleak = Qleak2
                        end

                    end


                    if Qleak < QCSF - QSN
                        QLP = Qleak;

                        Qcranial = QCSF-QSN-QLP;
                    else
                        QLP = QCSF - QSN;
                        Qcranial = 0.0;
                    end
                end

            end
        else
            if hourfraction < 0.1
                QSN = 0.0
                Qcranial = QCSF-VLP/CSFdrawtime-QSN;

                QLP = VLP/CSFdrawtime;
            else

                if Qleak < QCSF - QSN
                    QLP = Qleak
                    Qcranial = QCSF-QSN-QLP;
                else
                    QLP = QCSF - QSN
                    Qcranial = 0.0;
                end
            end

        end
    else
        if Qleak < QCSF - QSN
            QLP = Qleak
            Qcranial = QCSF-QSN-QLP;
        else
            QLP = QCSF - QSN
            Qcranial = 0.0;
        end
    end

    Leu,Qcranial,QSN,QLP,Qrefill,VSP3
end


# Important parameters to adjust to fit concentration data
Qleak = 0.0
VSP = 80.0
fVCSF = 1.0
#######

filename = string(@__DIR__,"\\inital_param_estimate.mat")
vars3 = matread(filename)
initial_param_estimate = vars3["resultsmatrix"]


for subjectnum = 1:1

    println(subjectnum)
    status = subject_optimization(subjectnum,initial_param_estimate[subjectnum,:],VSP,Qleak,fVCSF)
end
