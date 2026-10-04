using CSV, DataFrames, Dates, StatsBase, ProgressMeter, DataFramesMeta, NetCDF, LinearAlgebra, Manopt, Manifolds, JuMP, JLD2, Distances, RCall
import Distances: Euclidean as EuclideanDistance
using SparseArrays
using Distributions
import Manifolds:Stiefel 
using Random
using Kronecker
using DelimitedFiles
using StableRNGs

rng = StableRNG(0)

# https://gdex.ucar.edu/dataset/371_abaker.html
# Manopt needs to be 0.5.39 to leverage JuMP

function energy_dist(x, y)

    #x is a matrix of size d x n_samples_model
    #y is a matrix of size d x n_samples_data
    x = isa(x, Vector) ? x' : x
    y = isa(y, Vector) ? y' : y
    
    n_samples_model = size(x,2)  
    n_samples_data = size(y,2)
    
    output1 = 0.0
    output2 = 0.0

    @views for i in 1:n_samples_model
        x̂ = x[:,i]
        output1 += sum(sqrt.(sum((y .- x̂).^2,dims=1)))
        output2 += sum(sqrt.(sum((x[:,(1:n_samples_model) .!= i] .- x̂).^2,dims=1)))
    end

    return output1/(n_samples_data*n_samples_model) - output2/(2.0*n_samples_model^2)

end

varnames = ["AODVIS";"BURDEN1";"BURDEN2";"BURDEN3";"BURDENBC";"BURDENPOM";"BURDENSEASALT";"BURDENSO4"; "BURDENSOA";"CDNUMC";"CLDHGH";"CLDMED";"CLDTOT";"FLDS";"FLNS";"FLNSC";"FLNT";"FLNTC";"FSDS";"FSDSC";"FSNS";"FSNSC";"FSNTC";"FSNTOA";"LHFLX";"LWCF";"PBLH";"PS";"QREFHT";"SHFLX";"SWCF";"TAUX";"TAUY";"TGCLDCWP";"TGCLDIWP";"TGCLDLWP";"TMQ";"TREFHT";"U10";"PRECT"];

n=48602;
T = 343

vartypes  = [repeat(["Aerosol"],9);repeat(["Cloud"],4);repeat(["Flux"],12);"Cloud";"Height";"Pressure";"Precipitation";"Flux";"Cloud";repeat(["Transport"],2);repeat(["Cloud"],3);"Precipitation";"Height";"Transport";"Precipitation"]

response_indices = findall([x in ["TMQ";"U10";"SWCF";"AODVIS";"FLNTC";"PBLH";"PS";"PRECT"] for x in varnames])
predictor_indices = findall((vartypes  .== "Flux" .|| vartypes .== "Cloud" .|| vartypes .== "Aerosol") .&& [x ∉ varnames[response_indices] for x in varnames])

@assert isempty(intersect(response_indices,predictor_indices))

r = length(response_indices)
p = length(predictor_indices)
u = 6
manifold = Stiefel(r,r)
ϵ = 0.05
maxIter = 50
δ = 1e-7
δinv = 1.0/δ

μ = 10.0 #enhanced penalty
lambda_W = 10.0 
λᵥ = 10000.0
alpha_targetedelastic = 1/3 #1 is glasso

println("Loading data:")

data_array = Array{Float64}(undef,40,n,T);

@showprogress for j = 1:40
        data_array[j,:,:]= ncread("Dropbox/Mizzou/envelope/data/TAMCDS.nc",varnames[j]); # assume α is zero for standardized
end

data_array_X = data_array[predictor_indices,:,:]
data_array_Y = data_array[response_indices,:,:]

locs = CSV.read("Dropbox/Mizzou/envelope/data/locs.csv",DataFrame,header=false) |> Matrix{Float64} 
locs[:,1] .-= 180
lon = locs[:,1]
lat = locs[:,2]

keeplocindices = 1:n
keeplocs = locs[keeplocindices,:]
keeplon = keeplocs[:,1]
keeplat = keeplocs[:,2]
n = length(keeplocindices)

data_array_X = data_array_X[:,keeplocindices,:]
data_array_Y = data_array_Y[:,keeplocindices,:]

lonscaled = lon./ 180
latscaled = lat./ 90

## basis setup and initial guesses

println("Basis setup")

#SVD

reshaped_data_array_Y = Array{Float64}(undef, n, length(response_indices)*T)
if true
    for i = 1:n
        reshaped_data_array_Y[i,:] .= vec(data_array_Y[:,i,:]);
    end
    @time mysvd = svd(reshaped_data_array_Y)
    dd = mysvd.S
    csumd = cumsum(dd.^2)/sum(dd.^2)
    elboval = findfirst(csumd .> 0.95)
    Φ₁ = mysvd.U[:,1:elboval]
    ddv = dd[1:elboval]
end

# LK

R"""
library(LatticeKrig)
sDomain<-apply($keeplocs , 2,"range")
LK_360 <- LKrigSetup(sDomain, NC=3 ,nlevel=4, a.wght=4.05, nu=0.5,LKGeometry="LKSphere",startingLevel=1) 
LKlevelindices = LK_360$latticeInfo$mLevel

smallPhiLK <- as.dgCMatrix.spam(LKrig.basis($keeplocs, LK_360))
smallPhiLKentries = smallPhiLK@x
smallPhiLKindices = smallPhiLK@i
smallPhiLKpointers = smallPhiLK@p
smallPhiLKdimension = smallPhiLK@Dim

BLKSAR <- as.dgCMatrix.spam(LKrig.precision(LK_360,return.B=TRUE))
BLKSARentries = BLKSAR@x
BLKSARindices = BLKSAR@i
BLKSARpointers = BLKSAR@p
BLKSARdimension = BLKSAR@Dim
"""

@rget smallPhiLKentries
@rget smallPhiLKindices
@rget smallPhiLKpointers
@rget smallPhiLKdimension
@rget BLKSARentries
@rget BLKSARindices
@rget BLKSARpointers
@rget BLKSARdimension
@rget LKlevelindices
nlevel = length(LKlevelindices)

LKlevelindices_matrix = []
for i in 1:length(LKlevelindices)
    if isequal(i,1)
        push!(LKlevelindices_matrix,1:LKlevelindices[i])
    else
        push!(LKlevelindices_matrix,((1 .+ LKlevelindices[i-1]):LKlevelindices[i]) )
    end
end

createlevelindices = function(LKlevelindices)
    empp = []
    cumsumLKlevelindices = cumsum(LKlevelindices)
    for i in 1:length(LKlevelindices)
        if isequal(i,1)
            push!(empp,1:cumsumLKlevelindices[1])
        else
            push!(empp,(cumsumLKlevelindices[i-1] + 1):(cumsumLKlevelindices[i]))
        end
    end
    return empp
end
levelindices = createlevelindices(LKlevelindices)

biglevelindices = [
    begin
        indd = (r-u)*levelindices[x]
        (indd[1]-((r-u)-1)):indd[end]
    end 
    for (x,lvlindices) in enumerate(levelindices)]

Φ₀ = SparseMatrixCSC(smallPhiLKdimension[1],smallPhiLKdimension[2],1 .+ smallPhiLKpointers, 1 .+ smallPhiLKindices,smallPhiLKentries)
map!(x-> ifelse(isequal(x,4.05), 4, x), BLKSARentries)
BLKSAR = SparseMatrixCSC(BLKSARdimension[1],BLKSARdimension[2],1 .+ BLKSARpointers, 1 .+ BLKSARindices,BLKSARentries)

## Remaining setup
L = size(Φ₁,2)
M = size(Φ₀,2)

Φ₀ᵀΦ₀ = Φ₀'*Φ₀

## initial guesses

println("Initial guesses")
# println("Variables with subscript 1 are material")
# println("Variables with subscript 0 are immaterial")

Γ = I(r)
Γ₁ = Γ[:,1:u] #[I(u); zeros(r-u,u)]
Γ₀ = Γ[:,(u+1):end] #[I(r-u); zeros(u,r-u)]

η_inv = zeros(p,p)
η_noninv = zeros(u,p)
η_old = zeros(u,p)
η = zeros(u,p)
β = zeros(r,p)

Γ₁η = Γ₁ * η

ηouterproduct = η*η'

Q₁list = [spdiagm(ones(u)) for _ in 1:L]

bigQ₁ =  blockdiag([sparse(Q₁list[kk]) for kk in 1:L]...)

Q₀list = [spdiagm(ones(r-u)) for (x,lvlindices) in enumerate(levelindices)]

S₁ = zeros(u,u,L)

data_array_Y_project_to_Ψ₁ = Array{Float64}(undef,r,L,T)
data_array_X_project_to_Ψ₁ = Array{Float64}(undef,p,L,T)
data_array_X_project_to_Ψ₀ = Array{Float64}(undef,p,M,T)
data_array_Y_project_to_Ψ₀ = Array{Float64}(undef,r,M,T)

Γ₁data_array_Y_project_to_Ψ₁ = Array{Float64}(undef,u,L,T)
Γ₀data_array_Y_project_to_Ψ₀ = Array{Float64}(undef,r-u,M,T)

data_array_X_project_to_Φ₀ = Array{Float64}(undef,p,M,T)
data_array_Y_project_to_Φ₀ = Array{Float64}(undef,r,M,T)
Φ₀ᵀΦ₀ = Φ₀'*Φ₀
Φ₀ᵀΦ₀reg = Φ₀ᵀΦ₀ + .00001 * I(M)

@showprogress for t=1:T
    data_array_Y_project_to_Φ₀[:,:,t] .= data_array_Y[:,:,t] * Φ₀
    data_array_X_project_to_Φ₀[:,:,t] .= data_array_X[:,:,t] * Φ₀
end

@showprogress for t=1:T
    data_array_Y_project_to_Ψ₁[:,:,t] .= data_array_Y[:,:,t] * Φ₁
    data_array_X_project_to_Ψ₁[:,:,t] .= data_array_X[:,:,t] * Φ₁

    data_array_Y_project_to_Ψ₀[:,:,t] .= (Φ₀ᵀΦ₀reg \ data_array_Y_project_to_Φ₀[:,:,t]')'
    data_array_X_project_to_Ψ₀[:,:,t] .= (Φ₀ᵀΦ₀reg \ data_array_X_project_to_Φ₀[:,:,t]')'
    
    Γ₁data_array_Y_project_to_Ψ₁[:,:,t] .= Γ₁'*data_array_Y_project_to_Ψ₁[:,:,t]
    Γ₀data_array_Y_project_to_Ψ₀[:,:,t] .= Γ₀'*data_array_Y_project_to_Ψ₀[:,:,t]
end

ηdata_array_X_project_to_Ψ₁ = Array{Float64}(undef,u,L,T)

@showprogress for t=1:T 
    ηdata_array_X_project_to_Ψ₁[:,:,t] .= η * data_array_X_project_to_Ψ₁[:,:,t]
end

model_tease = Model(Manopt.JuMP_Optimizer)
set_attribute(model_tease, "descent_state_type", GradientDescentState)
@variable(model_tease, γ[1:r, 1:r] in manifold, start = 1.0)

Q₁_sum_sq = Vector{Float64}(undef,L)
Q₀_sum_sq = Vector{Float64}(undef,M)
Q₁_sum_sq_diff = Vector{Float64}(undef,L)
Q₀_sum_sq_diff = Vector{Float64}(undef,M)

rmseQ₀ = 1.0
rmseQ₁ = 1.0
rmseΓ₁η = 1.0
rmseΓ₀ = 1.0
rmseΓ = 1.0
rmseη = 1.0

κ₀ = sqrt(.05)

Blist = [
    begin
        Symmetric(BLKSAR[lvlindices,lvlindices] + κ₀^2 * I(length(lvlindices)))
    end
for (x,lvlindices) in enumerate(levelindices)]

BBlist = [
    begin
        thisB = Blist[x]
        Symmetric(thisB * thisB')
    end
for (x,lvlindices) in enumerate(levelindices)]

bigQ₀ = blockdiag([
    begin
        kron(Q₀list[x],BBlist[x])
    end
    for (x,lvlindices) in enumerate(levelindices)]...)

counter = 0

## TEASE estimation
myrv = Bernoulli(0.5)

fitTEASE = true

sz1 = length.(levelindices)

while fitTEASE
    counter += 1
    println("iteration: $counter")
    println("rmseQ₀: $rmseQ₀ & rmseQ₁: $rmseQ₁ & rmseΓ₀: $rmseΓ₀ & rmseΓ₁η: $rmseΓ₁η & rmseΓ: $rmseΓ & rmseη: $rmseη")

    bigQ₀guessold = bigQ₀
    bigQ₁guessold = bigQ₁

    ## η estimation
    println("   η optimization")
    η_noninv .= 0
    for t=1:T
        η_noninv .+= Γ₁data_array_Y_project_to_Ψ₁[:,:,t] * data_array_X_project_to_Ψ₁[:,:,t]'
    end

    η_noninv ./= T

    η_inv .= 0
    for t=1:T
        η_inv .+= data_array_X_project_to_Ψ₁[:,:,t] * data_array_X_project_to_Ψ₁[:,:,t]'
    end

    η_inv ./= T

    η_inv .+= μ*L*I(p)

    η_old .= η
    norm_η_old = isone(counter) ? 1.0 : sum(abs2,η_old)
    η .= (η_inv \ η_noninv')'
    rmseη = sqrt(sum(abs2,η-η_old)/norm_η_old)
    ηouterproduct .= η*η'

    for t=1:T 
        ηdata_array_X_project_to_Ψ₁[:,:,t] .= η * data_array_X_project_to_Ψ₁[:,:,t]
    end

    ## Envelope estimation

    println("   Compiling stiefel objective")
    @time @objective(model_tease, Min, 
    sum([begin
        vecc = vec(γ[:,1:u]' * data_array_Y_project_to_Ψ₁[:,:,k] -  ηdata_array_X_project_to_Ψ₁[:,:,k]) 
        vecc' * bigQ₁ * vecc
    end for k in 1:T])/T +
    sum([begin
        vecc = vec(γ[:,(u+1):end]' * data_array_Y_project_to_Ψ₀[:,:,k]) 
        vecc' * bigQ₀ * vecc
    end for k in 1:T])/T)
    println("   Stiefel optimization")
    @time optimize!(model_tease)
    solution_summary(model_tease)

    rmseΓ= sqrt(sum(abs2,Γ - value(γ))/sum(abs2,Γ))
    Γ = value(γ)
    rmseΓ₀= sqrt(sum(abs2,Γ₀ - Γ[:,(u+1):end])/sum(abs2,Γ₀))
    norm_Γ₁η_old = isone(counter) ? 1.0 : sum(abs2,Γ₁η)
    rmseΓ₁η = sqrt(sum(abs2,Γ₁η - Γ[:,1:u]*η)/norm_Γ₁η_old)
    Γ₀ = Γ[:,(u+1):end]
    Γ₁ = Γ[:,1:u] 
    Γ₁η = Γ₁ * η

    for t=1:T
        Γ₁data_array_Y_project_to_Ψ₁[:,:,t] .= Γ₁'*data_array_Y_project_to_Ψ₁[:,:,t]
        Γ₀data_array_Y_project_to_Ψ₀[:,:,t] .= Γ₀'*data_array_Y_project_to_Ψ₀[:,:,t]
    end

    ## Q estimation

    for l=1:L
        S₁[:,:,l] .= 0
        for t=1:T
            Γ₁ᵀY_minus_ηX_times_Φ₁ = (Γ₁data_array_Y_project_to_Ψ₁[:,l,t] - ηdata_array_X_project_to_Ψ₁[:,l,t]) 
            S₁[:,:,l] .+= Γ₁ᵀY_minus_ηX_times_Φ₁ * Γ₁ᵀY_minus_ηX_times_Φ₁'
        end
        S₁[:,:,l] ./= T

        S₁[:,:,l] .+= μ * ηouterproduct
    end

    println("   Q₁ material optimization")
    
    @showprogress for l=1:L
        Target = diagm(repeat(1.0 ./ [ddv[l]],u))

        if iszero(lambda_W)
            Qguess1 = inv(S₁[:,:,l])
        else
            R"""
                Qguess1 <- GLassoElnetFast::gelnet(S=$(S₁[:,:,l]),lambda=$lambda_W,alpha=$alpha_targetedelastic,penalize.diagonal=TRUE,Target=$Target)$Theta
            """
            @rget Qguess1
        end
        if isone(u)
            Q₁list[l] .= Qguess1
        else
            Q₁list[l] .= Symmetric(Qguess1)
        end
    end

    bigQ₁ =  blockdiag([Q₁list[kk] for kk in 1:L]...)
    rmseQ₁ = sqrt(sum(abs2,bigQ₁ - bigQ₁guessold))/sqrt(sum(abs2,bigQ₁guessold))

    println("   Q₀ immaterial optimization")

    dcrmseQ₀ = 1
    dccounter = 0 
    while dcrmseQ₀ > 0.1

        dccounter += 1
        println("dccounter $dccounter")
        println("rmseQ₀ $dcrmseQ₀")

        dcrmseQ₀=0
      
        eemat = rand(rng,myrv,(size(bigQ₀,1),30))*2 .- 1
        Υ = bigQ₀ + kron(Φ₀ᵀΦ₀, δinv * I(r-u))
        cholΥ = cholesky(Υ)
        cholYlinsolvemat = cholΥ \ eemat

        for l in 1:nlevel

            lvlindices = biglevelindices[l]
            
            tracepartlinearized_S = Symmetric(
            begin
                
                innertarget = zeros(r-u,r-u)

                for (y,ee) in enumerate(eachcol(eemat))
                    vecc = cholYlinsolvemat[:,y]
                    subvecc = vecc[lvlindices]
                    matsubvecc = reshape(subvecc,(sz1[l],r-u))
                    innertarget .+= matsubvecc' * BBlist[l] * reshape(ee[lvlindices],(sz1[l],r-u))  
                end 
                innertarget
            end)/size(eemat,2)

            tracepart_S = Symmetric(
            begin
                
                innertarget = zeros(r-u,r-u)

                for k in 1:T
                    vecc = vec((Γ₀data_array_Y_project_to_Ψ₀[:,:,k]))  
                    subvecc = vecc[lvlindices]
                    matsubvecc = reshape(subvecc,(sz1[l],r-u))
                    innertarget .+= matsubvecc' * BBlist[l] * matsubvecc
                end 
                innertarget
            end
            )/T

            S_star = (tracepartlinearized_S+tracepart_S)/length(levelindices[l])
            Sigmainit = I(r-u)
            Lambda = λᵥ*I(r-u)

             R"""
            Sigmaguess <- spcov::spcov(Sigma=$Sigmainit,S=$S_star,lambda=$Lambda,step.size=1)$Sigma
            """
            @rget Sigmaguess
            Qguess = Symmetric(inv(Symmetric(Sigmaguess)))

            dcrmseQ₀ += sqrt(sum(abs2,Q₀list[l] - Qguess))/sqrt(sum(abs2,Qguess))

            Q₀list[l]=Qguess
        end

        bigQ₀ = blockdiag([
        begin
            kron(Q₀list[x],BBlist[x])
        end
        for (x,lvlindices) in enumerate(levelindices)]...)

        println(dcrmseQ₀)

    end

    rmseQ₀ = sqrt(sum(abs2,bigQ₀ - bigQ₀guessold))/sqrt(sum(abs2,bigQ₀guessold))
    
    if (rmseQ₀ < ϵ && rmseQ₁ < ϵ && rmseΓ₁η < ϵ) | (counter > maxIter)
        @save "Dropbox/Mizzou/envelope/output_TAMCDS/TEASE_TAMCDS_enhanced_targeted_spcov.jld2" Γ₁ Γ₀ η bigQ₀ lambda_W alpha_targetedelastic Φ₀ Φ₁ bigQ₁ Q₁list Q₀list κ₀ λᵥ μ
        break
    end
end

## generate simulations from estimates

loadTEASE = false
if loadTEASE
    @load "Dropbox/Mizzou/envelope/output_TAMCDS/TEASE_TAMCDS_enhanced_targeted_spcov.jld2"
    M = size(Φ₀,2)
    L = size(Φ₁,2)
end

generate = true
if generate
    choleskyQ₀list = [cholesky(x).U for x in Q₀list]
    choleskyQ₁list = [cholesky(x).U for x in Q₁list]

    Wmatlist = Array{Float64}(undef,u,L,T)

    for l in 1:L
        Wmatlist[:,l,:] .= choleskyQ₁list[l] \ randn(rng,u,T)
    end

    Vmatlist = Array{Float64}(undef,r-u,M,T)

    @showprogress for t in 1:T
        targetmat = [
        begin
            thisB = Blist[x]
            randmat = randn(rng,r-u,length(lvlindices))
            vsimmat1 = choleskyQ₀list[x] \ randmat
            (thisB \ vsimmat1')'
        end
        for (x,lvlindices) in enumerate(levelindices)]

        Vmatlist[:,:,t] = hcat(targetmat...)
    end

    data_array_Ysim_tease = Array{Float64}(undef,r,n,T)

    @showprogress for t in 1:T
        data_array_Ysim_tease[:,:,t] .= Γ₁ * η * data_array_X[:,:,t]  + Γ₀ * Vmatlist[:,:,t] * Φ₀' + Γ₁ * Wmatlist[:,:,t] * Φ₁' 
    end
end

# edists = [energy_dist(data_array_Ysim_tease[i,:,:],data_array_Y[i,:,:]) for i in 1:length(response_indices)]
# writedlm("Dropbox/Mizzou/envelope/output_TAMCDS/edist_u$(u)_mu$(μ)_lambdaV$(λᵥ)_lambdaW$(lambda_W).csv",edists)

## Plotting 

# shift all observations to right to match up with coastlines, modulo 180
keeplocshift = copy(keeplocs)
keeplocshift[:,1] .+= 180
shiftthese = findall(keeplocshift[:,1] .≥ 180) 
keeplocshift[shiftthese,1] .= (keeplocshift[shiftthese,1] .- 360)

using GeoMakie, CairoMakie, PlotUtils
using ColorSchemes

exampleplot = true
if exampleplot

    thiscolorrangesims = []
    for i in 1:r
        thiscolorrangesimsup = findmax([vec(data_array_Y[i,:,1]);vec(data_array_Ysim_tease[i,:,1])])[1]
        thiscolorrangesimslow = findmin([vec(data_array_Y[i,:,1]);vec(data_array_Ysim_tease[i,:,1])])[1]
        maxx = maximum(abs.([thiscolorrangesimsup;thiscolorrangesimslow]))
        push!(thiscolorrangesims, [-maxx,maxx])
    end

    for i in 1:r
        thisvar = varnames[response_indices[i]]
        for j in 1:2
            fig = Figure()
            ga = GeoAxis(
            fig[1, 2]; # any cell of the figure's layout
            # dest = "+proj=wintri", # the CRS in which you want to plot
            # coastlines = true, # plot coastlines from Natural Earth, as a reference
            # latlims = (minimum(lat),maximum(lat)),
            # lonlims = (minimum(lon),maximum(lon))
            )
            sc = scatter!(keeplocshift[:,1], keeplocshift[:,2]; colorrange = thiscolorrangesims[i], color = data_array_Y[i,keeplocindices,j], strokecolor = (:black, 0.2), colormap = reverse(ColorSchemes.BrBG_5.colors))
            cb1 = Colorbar(fig[1,1], sc; label = thisvar, height = Relative(0.65))
            lines!(ga, GeoMakie.coastlines(),color=:black)
            fig
            savename = "Dropbox/Mizzou/envelope/pix/TAMCDS_$(thisvar)_$j.png"
            save(savename,fig)
             R"""
            knitr::plot_crop($savename)
            """
        end
    end

    for i in 1:r
        thisvar = varnames[response_indices[i]]
        for j in 1:2
            fig = Figure()
            ga = GeoAxis(
            fig[1, 1]; # any cell of the figure's layout
            # dest = "+proj=wintri", # the CRS in which you want to plot
            # coastlines = true, # plot coastlines from Natural Earth, as a reference
            # latlims = (minimum(lat),maximum(lat)),
            # lonlims = (minimum(lon),maximum(lon))
            )
            sc = scatter!(keeplocshift[:,1], keeplocshift[:,2]; colorrange = thiscolorrangesims[i], color = data_array_Ysim_tease[i,keeplocindices,j], strokecolor = (:black, 0.2), colormap = reverse(ColorSchemes.BrBG_5.colors))
            cb1 = Colorbar(fig[1,2], sc; label = thisvar, height = Relative(0.65))
            lines!(ga, GeoMakie.coastlines(),color=:black)
            fig
            savename = "Dropbox/Mizzou/envelope/pix/TAMCDS_$(thisvar)_TEASE_enhanced_targeted_spcov$j.png"
            save(savename,fig)
             R"""
            knitr::plot_crop($savename)
            """
        end
    end

    Q₀invlist = [inv(collect(x)) for x in Q₀list]
    Q₁invlist = [inv(collect(x)) for x in Q₁list]
    BBinvlist = [inv(collect(x)) for x in BBlist]

    MarginalVariancesMat₁ = zeros(r,n)
    MarginalVariancesMat₀ = zeros(r,n)

    @showprogress for l in 1:L
        invmatdiag = diag(Γ₁ * Q₁invlist[l] * Γ₁' )
        for k in 1:n
            MarginalVariancesMat₁[:,k] .+= Φ₁[k,l]^2 * invmatdiag
        end
    end

    @showprogress for i in 1:length(levelindices)
        kronmat = (Q₀invlist[i] ⊗ BBinvlist[i])
        for k in 1:n
            kmat = kron(collect(Φ₀[k,levelindices[i]])', Γ₀)
            MarginalVariancesMat₀[:,k] .+= diag(kmat * kronmat * kmat')
        end
    end

    MarginalVariancesMat = MarginalVariancesMat₀ + MarginalVariancesMat₁

    middle_vec_indices = findall(locs[:,2] .>= 40 .&& locs[:,2] .<= 41)
    sort_middle_vec_indices = sortperm(locs[middle_vec_indices,1])
    tempvec = middle_vec_indices[sort_middle_vec_indices]
    us_plot_indices = middle_vec_indices[sort_middle_vec_indices][[1;collect(3:15:316)]]
    us_plot_index_17 = us_plot_indices[17]

    middle_vec_indices = findall(locs[:,2] .>= -6 .&& locs[:,2] .<= -5)
    sort_middle_vec_indices = sortperm(locs[middle_vec_indices,1])
    tempvec = middle_vec_indices[sort_middle_vec_indices]
    equator_plot_indices = middle_vec_indices[sort_middle_vec_indices][[1;collect(3:15:516)]]
    equator_plot_index_23 = equator_plot_indices[23]

    CovMat₁_us = zeros(n,2)
    CovMat₀_us = zeros(n,2)
    CovMat₁_equator = zeros(n,2)
    CovMat₀_equator = zeros(n,2)

    @showprogress for l in 1:L
        invmat = (Γ₁ * Q₁invlist[l] * Γ₁')
        for k in 1:n
            CovMat₁_us[k,1] += Φ₁[k,l] * Φ₁[us_plot_index_17,l] * invmat[4,8]
            CovMat₁_us[k,2] += Φ₁[k,l] * Φ₁[us_plot_index_17,l] * invmat[8,4]

            CovMat₁_equator[k,1] += Φ₁[k,l] * Φ₁[equator_plot_index_23,l] * invmat[4,8]
            CovMat₁_equator[k,2] += Φ₁[k,l] * Φ₁[equator_plot_index_23,l] * invmat[8,4]
        end
    end

    @showprogress for i in 1:length(levelindices)
        kronmat = (Q₀invlist[i] ⊗ BBinvlist[i])
        for k in 1:n
            prodmat_us = kron(collect(Φ₀[us_plot_index_17,levelindices[i]])', Γ₀) * kronmat * kron(collect(Φ₀[k,levelindices[i]])', Γ₀)'
            CovMat₀_us[k,1] += prodmat_us[4,8]
            CovMat₀_us[k,2] += prodmat_us[8,4]

            prodmat_equator = kron(collect(Φ₀[equator_plot_index_23,levelindices[i]])', Γ₀) * kronmat * kron(collect(Φ₀[k,levelindices[i]])', Γ₀)'
            CovMat₀_equator[k,1] += prodmat_equator[4,8]
            CovMat₀_equator[k,2] += prodmat_equator[8,4]

        end
    end

    CovMat_us = CovMat₀_us + CovMat₁_us
    CovMat_equator = CovMat₀_equator + CovMat₁_equator

    CorMat_us = zeros(n,2)
    CorMat_equator = zeros(n,2)

    CorMat_us[:,1] .= CovMat_us[:,1] ./ sqrt.(MarginalVariancesMat[4,us_plot_index_17].*MarginalVariancesMat[8,:])
    CorMat_us[:,2] .= CovMat_us[:,2] ./ sqrt.(MarginalVariancesMat[8,us_plot_index_17].*MarginalVariancesMat[4,:])

    CorMat_equator[:,1] .= CovMat_equator[:,1] ./ sqrt.(MarginalVariancesMat[4,equator_plot_index_23].*MarginalVariancesMat[8,:])
    CorMat_equator[:,2] .= CovMat_equator[:,2] ./ sqrt.(MarginalVariancesMat[8,equator_plot_index_23].*MarginalVariancesMat[4,:])

    sdcols = []
    sdcolormap = []
    for i in 1:r
        if maximum(sqrt.(MarginalVariancesMat[i,:])) < 1
            push!(sdcols,[minimum(sqrt.(MarginalVariancesMat[i,:]));1.0])
            push!(sdcolormap, reverse(ColorSchemes.RdBu_5.colors[3:5]))
        elseif minimum(sqrt.(MarginalVariancesMat[i,:])) > 1
            push!(sdcols,[1.0;maximum(sqrt.(MarginalVariancesMat[i,:]))])
            push!(sdcolormap, reverse(ColorSchemes.RdBu_5.colors[1:3]))
        end
    end

    for i in 1:r
        if(isodd(i))
            thisvar = varnames[response_indices[i]]
            fig = Figure()
                ga = GeoAxis(
                fig[1, 2]; # any cell of the figure's layout
                )
                sc = scatter!(keeplocshift[:,1], keeplocshift[:,2]; colorrange = sdcols[i], color = sqrt.(MarginalVariancesMat[i,:]), strokecolor = (:black, 0.2), colormap = sdcolormap[i])
                cb1 = Colorbar(fig[1,1],sc; label = thisvar, height = Relative(0.65))
                lines!(ga, GeoMakie.coastlines(),color=:black)
                fig
            savename = "Dropbox/Mizzou/envelope/pix/TAMCDS_Sd_$(thisvar).png"
            save(savename,fig)
            R"""
            knitr::plot_crop($savename)
            """
        else
            thisvar = varnames[response_indices[i]]
            fig = Figure()
                ga = GeoAxis(
                fig[1, 1]; # any cell of the figure's layout
                )
                 if i==4
                    sc = scatter!(keeplocshift[:,1], keeplocshift[:,2]; colorrange = sdcols[i], color = sqrt.(MarginalVariancesMat[i,:]), strokecolor = (:black, 0.2), colormap = sdcolormap[i])
                else
                    sc = scatter!(keeplocshift[:,1], keeplocshift[:,2]; colorrange = sdcols[i], color = sqrt.(MarginalVariancesMat[i,:]), strokecolor = (:black, 0.2), colormap = sdcolormap[i])
                end
                cb1 = Colorbar(fig[1,2],sc; label = thisvar, height = Relative(0.65))
                lines!(ga, GeoMakie.coastlines(),color=:black)
                fig
            savename = "Dropbox/Mizzou/envelope/pix/TAMCDS_Sd_$(thisvar).png"
            save(savename,fig)
            R"""
            knitr::plot_crop($savename)
            """
        end
    end

    fig = Figure()
        ga = GeoAxis(
        fig[1, 1]; # any cell of the figure's layout
        )
        sc = scatter!(keeplocshift[:,1], keeplocshift[:,2]; colorrange = (-.09,.09), color = CorMat_us[:,1], strokecolor = (:black, 0.2), colormap = reverse(ColorSchemes.RdBu_5.colors))
        cb1 = Colorbar(fig[1,2],sc;  height = Relative(0.65))
        scatter!(keeplocshift[us_plot_index_17,1],keeplocshift[us_plot_index_17,2];color=RGBf(0, 1, 0),marker=:xcross)
        lines!(ga, GeoMakie.coastlines(),color=:black)
        fig
        savename = "Dropbox/Mizzou/envelope/pix/TAMCDS_CorMat_uscol1.png"
        save(savename,fig)
        R"""
        knitr::plot_crop($savename)
        """

    fig = Figure()
        ga = GeoAxis(
        fig[1, 1]; # any cell of the figure's layout
        )
        sc = scatter!(keeplocshift[:,1], keeplocshift[:,2]; colorrange = (-.09,.09), color = CorMat_us[:,2], strokecolor = (:black, 0.2), colormap = reverse(ColorSchemes.RdBu_5.colors))
        cb1 = Colorbar(fig[1,2],sc; height = Relative(0.65))
        scatter!(keeplocshift[us_plot_index_17,1],keeplocshift[us_plot_index_17,2];color=RGBf(0, 1, 0),marker=:xcross)
        lines!(ga, GeoMakie.coastlines(),color=:black)
        fig
        savename = "Dropbox/Mizzou/envelope/pix/TAMCDS_CorMat_uscol2.png"
        save(savename,fig)
        R"""
        knitr::plot_crop($savename)
        """

    fig = Figure()
        ga = GeoAxis(
        fig[1, 1]; # any cell of the figure's layout
        )
        sc = scatter!(keeplocshift[:,1], keeplocshift[:,2]; colorrange = (-.09,.09), color = CorMat_equator[:,1], strokecolor = (:black, 0.2), colormap = reverse(ColorSchemes.RdBu_5.colors))
        cb1 = Colorbar(fig[1,2], sc; height = Relative(0.65))
        scatter!(keeplocshift[equator_plot_index_23,1],keeplocshift[equator_plot_index_23,2];color=RGBf(0, 1, 0),marker=:xcross)
        lines!(ga, GeoMakie.coastlines(),color=:black)
        fig
        savename = "Dropbox/Mizzou/envelope/pix/TAMCDS_CorMat_equatorcol1.png"
        save(savename,fig)
        R"""
        knitr::plot_crop($savename)
        """

    fig = Figure()
        ga = GeoAxis(
        fig[1, 1]; # any cell of the figure's layout
        )
        sc = scatter!(keeplocshift[:,1], keeplocshift[:,2]; colorrange = (-.09,.09), color = CorMat_equator[:,2], strokecolor = (:black, 0.2), colormap = reverse(ColorSchemes.RdBu_5.colors))
        cb1 = Colorbar(fig[1,2], sc; height = Relative(0.65))
        scatter!(keeplocshift[equator_plot_index_23,1],keeplocshift[equator_plot_index_23,2];color=RGBf(0, 1, 0),marker=:xcross)
        lines!(ga, GeoMakie.coastlines(),color=:black)
        fig
        savename = "Dropbox/Mizzou/envelope/pix/TAMCDS_CorMat_equatorcol2.png"
        save(savename,fig)
        R"""
        knitr::plot_crop($savename)
        """
end

## done

println("done")