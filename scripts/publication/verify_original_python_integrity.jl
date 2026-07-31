#!/usr/bin/env julia

using SHA

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const MANIFEST = joinpath(REPO, "reports", "ORIGINAL_PYTHON_INTEGRITY.csv")
const SEARCH_ROOTS = ("python_code", "notebooks", "data_preparation", "results_analysis")

normalize_path(path::AbstractString) = replace(normpath(string(path)), '\\' => '/')

function load_expected(path::AbstractString)
    isfile(path) || error("integrity manifest is missing: $(path)")
    expected = Dict{String,Tuple{Int,String}}()
    for (line_number, line) in enumerate(eachline(path))
        line_number == 1 && continue
        isempty(strip(line)) && continue
        fields = split(line, ','; limit=3)
        length(fields) == 3 || error("malformed integrity row $(line_number): $(line)")
        relative = normalize_path(fields[1])
        bytes = parse(Int, fields[2])
        digest = lowercase(strip(fields[3]))
        expected[relative] = (bytes, digest)
    end
    isempty(expected) && error("integrity manifest contains no files")
    return expected
end

function discover_python_assets()
    found = Set{String}()
    for root_name in SEARCH_ROOTS
        root = joinpath(REPO, root_name)
        isdir(root) || continue
        for (directory, _, files) in walkdir(root)
            for filename in files
                extension = lowercase(splitext(filename)[2])
                extension in (".py", ".ipynb") || continue
                relative = normalize_path(relpath(joinpath(directory, filename), REPO))
                push!(found, relative)
            end
        end
    end
    return found
end

function verify_integrity()
    expected = load_expected(MANIFEST)
    actual = discover_python_assets()
    expected_paths = Set(keys(expected))

    missing = sort!(collect(setdiff(expected_paths, actual)))
    unexpected = sort!(collect(setdiff(actual, expected_paths)))
    isempty(missing) || error("original Python assets are missing: " * join(missing, ", "))
    isempty(unexpected) || error("unexpected Python assets were introduced: " * join(unexpected, ", "))

    failures = String[]
    for relative in sort!(collect(expected_paths))
        path = joinpath(REPO, split(relative, '/')...)
        expected_bytes, expected_digest = expected[relative]
        actual_bytes = filesize(path)
        actual_digest = open(path, "r") do io
            bytes2hex(SHA.sha256(io))
        end
        actual_bytes == expected_bytes || push!(failures,
            "$(relative): bytes $(actual_bytes) != $(expected_bytes)")
        lowercase(actual_digest) == expected_digest || push!(failures,
            "$(relative): sha256 $(actual_digest) != $(expected_digest)")
    end
    isempty(failures) || error("original Python integrity check failed:\n" * join(failures, "\n"))
    println("Original Python sources and notebooks verified byte-for-byte: ", length(expected), " files.")
    return true
end

verify_integrity()
