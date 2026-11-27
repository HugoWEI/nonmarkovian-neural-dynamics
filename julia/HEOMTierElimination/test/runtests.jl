using Test, LinearAlgebra
using HierarchicalEOM
using HEOMTierElimination

@testset "HEOMTierElimination" begin
    @testset "fermionic Anderson impurity (diagonal deepest block)" begin
        HBAR = 0.658211928
        d_up = tensor(destroy(2), qeye(2))
        d_dn = tensor(sigmaz(), destroy(2))
        n_up = d_up' * d_up
        H = (-0.5 / HBAR) * (n_up + d_dn' * d_dn) + (1.0 / HBAR) * n_up * (d_dn' * d_dn)
        bath(op) = Fermion_Lorentz_Pade(op, 0.1 / HBAR, 0.05 / HBAR, 5.0 / HBAR, 0.03 / HBAR, 2)
        M = M_Fermion(H, 3, [bath(d_up), bath(d_dn)]; verbose = false)

        ref = steadystate(M; verbose = false)
        for method in (:direct, :gmres)
            ados = steadystate_eliminated(M; method = method)
            @test real(expect(n_up, getRho(ados))) ≈ real(expect(n_up, getRho(ref))) atol = 1e-9
            @test real(tr(getRho(ados))) ≈ 1 atol = 1e-12
        end
    end

    @testset "bosonic spin-boson (block-diagonal deepest block)" begin
        H = 0.5 * sigmaz() + 0.5 * sigmax()
        bath = Boson_DrudeLorentz_Pade(sigmaz(), 0.1, 0.5, 0.5, 2)
        M = M_Boson(H, 4, bath; verbose = false)
        ref = steadystate(M; verbose = false)
        for method in (:direct, :gmres)
            ados = steadystate_eliminated(M; method = method)
            @test getRho(ados).data ≈ getRho(ref).data atol = 1e-8
        end
    end

    @testset "restarted GMRES" begin
        H = 0.5 * sigmaz() + 0.5 * sigmax()
        bath = Boson_DrudeLorentz_Pade(sigmaz(), 0.1, 0.5, 0.5, 2)
        M = M_Boson(H, 4, bath; verbose = false)
        a1 = steadystate_eliminated(M; method = :direct)
        a2 = steadystate_eliminated(M; method = :gmres, restart = 20)
        @test getRho(a2).data ≈ getRho(a1).data atol = 1e-8
    end
end
