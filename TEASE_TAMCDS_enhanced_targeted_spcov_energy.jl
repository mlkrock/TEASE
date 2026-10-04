using CSV, DataFrames, Dates, StatsBase, ProgressMeter, DataFramesMeta, NetCDF, LinearAlgebra, Manopt, Manifolds, JuMP, JLD2, Distances, RCall
import Distances: Euclidean as EuclideanDistance
using SparseArrays
using Distributions
import Manifolds:Stiefel 
using Random
using Kronecker
using DelimitedFiles
using StableRNGs

# https://gdex.ucar.edu/dataset/371_abaker.html
# Manopt needs to be 0.5.39 to leverage JuMP

rng = StableRNG(0)

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
manifold = Stiefel(r,r)
ϵ = 0.05
maxIter = 20
δ = 1e-7
δinv = 1.0/δ

println("Loading data:")

data_array = Array{Float64}(undef,40,n,T);

@showprogress for j = 1:40
        data_array[j,:,:]= ncread("data/TAMCDS.nc",varnames[j]); # assume α is zero for standardized
end

data_array_X = data_array[predictor_indices,:,:]
data_array_Y = data_array[response_indices,:,:]

locs = CSV.read("data/locs.csv",DataFrame,header=false) |> Matrix{Float64} 
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

Φ₀ = SparseMatrixCSC(smallPhiLKdimension[1],smallPhiLKdimension[2],1 .+ smallPhiLKpointers, 1 .+ smallPhiLKindices,smallPhiLKentries)
map!(x-> ifelse(isequal(x,4.05), 4, x), BLKSARentries)
BLKSAR = SparseMatrixCSC(BLKSARdimension[1],BLKSARdimension[2],1 .+ BLKSARpointers, 1 .+ BLKSARindices,BLKSARentries)

κ₀ = sqrt(0.05)

Blist = [
    begin
        # ksq = exp(κ₀*x)
        Symmetric(BLKSAR[lvlindices,lvlindices] + κ₀^2 * I(length(lvlindices)))
    end
for (x,lvlindices) in enumerate(levelindices)]

BBlist = [
    begin
        thisB = Blist[x]
        Symmetric(thisB * thisB')
    end
for (x,lvlindices) in enumerate(levelindices)]

# SVD

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

L = size(Φ₁,2)
M = size(Φ₀,2)

Φ₀ᵀΦ₀ = Φ₀'*Φ₀

lambda_Wvals = 0.0
μvals = 0.0
λᵥvals = [1000.0; 10000.0; 1000000.0; 10000000.0; 100000000.0]

μvals = [1.0; 10.0; 100.0; 1000.0; 10000.0]
lambda_Wvals = [1.0; 10.0; 100.0; 1000.0; 10000.0]
λᵥvals = 10000.0

# for u in 1:(r-2)
for u in 6

    biglevelindices = [
        begin
            indd = (r-u)*levelindices[x]
            (indd[1]-((r-u)-1)):indd[end]
        end 
        for (x,lvlindices) in enumerate(levelindices)]

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
    Φ₀ᵀΦ₀reg = Φ₀ᵀΦ₀ + .00001 * I(M)

    @showprogress for t in 1:T
        data_array_Y_project_to_Φ₀[:,:,t] .= data_array_Y[:,:,t] * Φ₀
        data_array_X_project_to_Φ₀[:,:,t] .= data_array_X[:,:,t] * Φ₀
    end

    @showprogress for t in 1:T
        data_array_Y_project_to_Ψ₁[:,:,t] .= data_array_Y[:,:,t] * Φ₁
        data_array_X_project_to_Ψ₁[:,:,t] .= data_array_X[:,:,t] * Φ₁

        data_array_Y_project_to_Ψ₀[:,:,t] .= (Φ₀ᵀΦ₀reg \ data_array_Y_project_to_Φ₀[:,:,t]')'
        data_array_X_project_to_Ψ₀[:,:,t] .= (Φ₀ᵀΦ₀reg \ data_array_X_project_to_Φ₀[:,:,t]')'
        
        Γ₁data_array_Y_project_to_Ψ₁[:,:,t] .= Γ₁'*data_array_Y_project_to_Ψ₁[:,:,t]
        Γ₀data_array_Y_project_to_Ψ₀[:,:,t] .= Γ₀'*data_array_Y_project_to_Ψ₀[:,:,t]
    end

    ηdata_array_X_project_to_Ψ₁ = Array{Float64}(undef,u,L,T)

    @showprogress for t in 1:T
        ηdata_array_X_project_to_Ψ₁[:,:,t] .= η * data_array_X_project_to_Ψ₁[:,:,t]
    end

    model_tease = Model(Manopt.JuMP_Optimizer)
    set_attribute(model_tease, "descent_state_type", GradientDescentState)
    @variable(model_tease, γ[1:r, 1:r] in manifold, start = 1.0)

    Q₁_sum_sq = Vector{Float64}(undef,L)
    Q₀_sum_sq = Vector{Float64}(undef,M)
    Q₁_sum_sq_diff = Vector{Float64}(undef,L)
    Q₀_sum_sq_diff = Vector{Float64}(undef,M)

    # rmseα = 1.0

    bigQ₀ = blockdiag([
        begin
            kron(Q₀list[x],BBlist[x])
        end
        for (x,lvlindices) in enumerate(levelindices)]...)

    for μ in μvals

        for λᵥ in λᵥvals

            for lambda_W in lambda_Wvals

                rmseQ₀ = 1.0
                rmseQ₁ = 1.0
                rmseΓ₁η = 1.0
                rmseΓ₀ = 1.0
                rmseΓ = 1.0
                rmseη = 1.0

                println("u: $(u) mu: $(μ) lambdaV: $(λᵥ) lambdaW: $(lambda_W)")

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
                    for t in 1:T
                        η_noninv .+= Γ₁data_array_Y_project_to_Ψ₁[:,:,t] * data_array_X_project_to_Ψ₁[:,:,t]'
                    end

                    η_noninv ./= T

                    η_inv .= 0
                    for t in 1:T
                        η_inv .+= data_array_X_project_to_Ψ₁[:,:,t] * data_array_X_project_to_Ψ₁[:,:,t]'
                    end

                    η_inv ./= T

                    η_inv .+= μ*L*I(p)

                    η_old .= η
                    norm_η_old = isone(counter) ? 1.0 : sum(abs2,η_old)
                    η .= (η_inv \ η_noninv')'
                    rmseη = sqrt(sum(abs2,η-η_old)/norm_η_old)
                    ηouterproduct .= η*η'

                    for t in 1:T
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

                    for t in 1:T
                        Γ₁data_array_Y_project_to_Ψ₁[:,:,t] .= Γ₁'*data_array_Y_project_to_Ψ₁[:,:,t]
                        Γ₀data_array_Y_project_to_Ψ₀[:,:,t] .= Γ₀'*data_array_Y_project_to_Ψ₀[:,:,t]
                    end

                    ## Q estimation

                    for l=1:L
                        S₁[:,:,l] .= 0
                        for t in 1:T
                            Γ₁ᵀY_minus_ηX_times_Φ₁ = (Γ₁data_array_Y_project_to_Ψ₁[:,l,t] - ηdata_array_X_project_to_Ψ₁[:,l,t]) 
                            S₁[:,:,l] .+= Γ₁ᵀY_minus_ηX_times_Φ₁ * Γ₁ᵀY_minus_ηX_times_Φ₁'
                        end
                        S₁[:,:,l] ./= T

                        S₁[:,:,l] .+= μ * ηouterproduct
                    end

                    println("   Q₁ material optimization")
                    
                    @showprogress for l=1:L
                        Target = diagm(1.0 ./repeat([ddv[l]],u))
                        if iszero(lambda_W)
                            Qguess1 = inv(S₁[:,:,l])
                        else
                            R"""
                                Qguess1 <- GLassoElnetFast::gelnet(S=$(S₁[:,:,l]),lambda=$lambda_W,alpha=1/3,penalize.diagonal=TRUE,Target=$Target)$Theta
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

                        edists = [energy_dist(data_array_Ysim_tease[i,:,:],data_array_Y[i,:,:]) for i in 1:length(response_indices)]
                                            
                        writedlm("output_TAMCDS/edists_u$(u)_mu$(μ)_lambdaV$(λᵥ)_lambdaW$(lambda_W).csv",edists)
                        break
                    end
                end
            end
        end
    end
end

# uvals = 1:(r-2)
# uvals = 6
# tab_results = Array{Float64}(undef,length(uvals),length(μvals),length(λᵥvals),length(lambda_Wvals))

# for (uenum,u) in enumerate(uvals)
#     for (i,μ) in enumerate(μvals)
#         for (j,λᵥ) in enumerate(λᵥvals)
#             for (h,lambda_W) in enumerate(lambda_Wvals)
#                 tab_results[uenum,i,j,h] = mean(readdlm("output_TAMCDS/edists_u$(u)_mu$(μ)_lambdaV$(λᵥ)_lambdaW$(lambda_W).csv"))
#             end
#         end
#     end
# end

# findmin(tab_results)