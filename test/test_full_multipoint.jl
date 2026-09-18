"""
    test/test_full_multipoint.jl

Parity at three further nuisance points. Baseline-only agreement is not
sufficient: a swapped parameter or a dead branch can reproduce one point by
accident, but not a foreground move, a systematics move and both together.
"""

function multipoint_case_updates()
    _, rows = read_reference_table(MULTIPOINT_DIR, "case_parameters.tsv")
    cases = Dict{String, Dict{Symbol, Float64}}()
    for row in axes(rows, 1)
        name = String(rows[row, 1])
        entry = get!(cases, name, Dict{Symbol, Float64}())
        entry[Symbol(String(rows[row, 2]))] = Float64(rows[row, 3])
    end
    return cases
end

@testset "ACT DR6 full likelihood — nuisance multipoint parity" begin
    header, cases = read_reference_table(MULTIPOINT_DIR, "cases.tsv")
    @test header == ["name", "chi2", "data_only_loglikelihood"]
    @test size(cases, 1) == 3

    # The published references this package must not regress.
    published = Dict(
        "foregrounds" => 4044.2026791085727,
        "systematics" => 3095.611295564593,
        "combined" => 8946.136822138633,
    )

    updates = multipoint_case_updates()
    baseline = NamedTuple(REFERENCE_NUISANCE)
    checkpoint_header, checkpoints =
        read_reference_table(MULTIPOINT_DIR, "foreground_totals_checkpoints.tsv")
    @test checkpoint_header == ["case", "polarization", "i", "j", "ell", "value"]

    for row in axes(cases, 1)
        name = String(cases[row, 1])
        @testset "case $name" begin
            nuisance = merge(baseline, NamedTuple(updates[name]))
            case_foregrounds = foregrounds(FOREGROUND_MODEL, LIKE, nuisance)
            prediction = predict(LIKE, REFERENCE_CMB, case_foregrounds, nuisance)

            reference = read_reference_vector(MULTIPOINT_DIR, "model_vector_$name.txt")
            @test prediction ≈ reference rtol=1e-10 atol=1e-10

            value = chi2(LIKE, prediction)
            @test value ≈ Float64(cases[row, 2]) rtol=1e-10
            @test value ≈ published[name] rtol=1e-10
            @test loglikelihood(LIKE, prediction) ≈ Float64(cases[row, 3]) rtol=1e-10

            # The prediction must actually differ from the baseline everywhere
            # it should, so an ignored parameter cannot pass.
            @test prediction != BASELINE_PREDICTION

            selection = findall(r -> String(checkpoints[r, 1]) == name,
                                axes(checkpoints, 1))
            @test !isempty(selection)
            for r in selection
                array = foreground_block(case_foregrounds, String(checkpoints[r, 2]))
                i = Int(checkpoints[r, 3]); j = Int(checkpoints[r, 4])
                ell = Int(checkpoints[r, 5])
                @test array[i, j, ell_index(ell)] ≈ Float64(checkpoints[r, 6]) rtol=1e-10 atol=1e-12
            end
        end
    end
end

@testset "ACT DR6 full likelihood — each parameter group moves the model" begin
    # A parameter that never changes the prediction is a dead branch.
    for name in ACT_DR6_FREE_PARAMETERS
        perturbed = ACTDR6Nuisance(REFERENCE_NUISANCE;
                                   NamedTuple{(name,)}((getfield(REFERENCE_NUISANCE, name) + 0.05,))...)
        prediction = predict(LIKE, REFERENCE_CMB,
                             foregrounds(FOREGROUND_MODEL, LIKE, perturbed), perturbed)
        @test prediction != BASELINE_PREDICTION
    end
end
